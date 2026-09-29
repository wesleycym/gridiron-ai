'''
        Run: 
            python ml/src/scripts/predict.py                                            # next unplayed week
            python ml/src/scripts/predict.py --season 2026 --week 5                     # a specific week
            python ml/src/scripts/predict.py --season 2025 --week 10 --show-actual      # backtest a past week
'''

"""
Predict fantasy outcome ranges for an upcoming week.
For every player who's been active recently and whose team plays that
week, we add a placeholder row for the upcoming game, run the SAME feature code used
for training (which only looks at earlier games), then run the quantile models on it.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from build_dataset import (
    EXCLUDE_WEEKS, ML_ROOT, MIN_PRIOR_GAMES, STAT_COLS,
    build_features, clean, load_raw, load_schedules, team_game_lines,
)
from train_quantiles import apply_offsets, fix_crossing, prob_over, to_pandas_features

MODEL_DIR = ML_ROOT / "models" / "quantiles"
OUT_DIR = ML_ROOT / "data" / "predictions"
THRESHOLDS = [10, 15, 20, 25]
RECENT_TEAM_GAMES = 4  # player must have played within the last N weeks with games


def load_models():
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    models = [
        lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_q{int(q * 100):02d}.txt"))
        for q in meta["quantiles"]
    ]
    return meta, models


def next_unplayed_week(schedules: pl.DataFrame) -> tuple[int, int]:
    """First regular-season week that has games without a final score yet."""
    upcoming = (
        schedules
        .filter((pl.col("game_type") == "REG") & pl.col("home_score").is_null())
        .sort(["season", "week"])
    )
    if upcoming.is_empty():
        raise SystemExit("No unplayed regular-season games found. Pass --season and --week.")
    return int(upcoming["season"][0]), int(upcoming["week"][0])


def upcoming_rows(history: pl.DataFrame, schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Placeholder rows for each recently active player whose team plays this week."""
    # "Recently active" = appeared in one of the last N weeks that had games
    recent_weeks = (
        history.select("season", "week").unique()
        .sort(["season", "week"], descending=True)
        .head(RECENT_TEAM_GAMES)
    )
    recent_players = history.join(recent_weeks, on=["season", "week"], how="semi")

    # Each player's most recent team and info
    latest = (
        history.sort(["season", "week"])
        .group_by("player_id").last()
        .join(recent_players.select("player_id").unique(), on="player_id", how="semi")
        .select("player_id", "player_display_name", "position", "team")
    )

    games = (
        team_game_lines(schedules)
        .filter((pl.col("season") == season) & (pl.col("week") == week))
        .select("season", "week", "game_id", "team", "opponent_team")
    )
    rows = latest.join(games, on="team", how="inner")  # teams on bye drop out here
    # Stats unknown for the upcoming game
    return rows.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in STAT_COLS])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fantasy ranges for a week.")
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--show-actual", action="store_true",
                        help="for a past week, add what each player actually scored")
    args = parser.parse_args()

    schedules = load_schedules()
    if schedules is None:
        raise SystemExit("schedules.parquet is required. Run fetch_data.py first.")
    if args.season and args.week:
        season, week = args.season, args.week
    else:
        season, week = next_unplayed_week(schedules)
    print(f"Predicting {season} week {week}")
    if week in EXCLUDE_WEEKS:
        print(f"WARNING: week {week} is excluded from training (starters often rest). "
              "Treat these numbers with extra caution.")

    all_games = clean(load_raw())
    # Only games strictly before the target week can be used
    before = (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
    history = all_games.filter(before)

    new = upcoming_rows(history, schedules, season, week).with_columns(pl.lit(True).alias("_upcoming"))
    combined = pl.concat(
        [history.with_columns(pl.lit(False).alias("_upcoming")), new],
        how="diagonal_relaxed",
    )
    feats = (
        build_features(combined, schedules)
        .filter(pl.col("_upcoming") & (pl.col("prior_games_career") >= MIN_PRIOR_GAMES))
    )
    if feats.is_empty():
        raise SystemExit("No players to predict. Is the week right, and is the data up to date?")

    # ---- Run the models ---------------------------------------------------------------
    meta, models = load_models()
    X = to_pandas_features(feats, [f for f in meta["features"] if f != "position"])
    X = X[meta["features"]]  # exact column order used in training
    raw = fix_crossing(np.column_stack([m.predict(X) for m in models]))
    preds = apply_offsets(raw, feats["position"].to_numpy(), meta["offsets"])

    qs = meta["quantiles"]
    col = {q: i for i, q in enumerate(qs)}
    out = feats.select(
        "player_id", "player_display_name", "position", "team", "opponent_team",
        "season", "week", "vegas_implied_total",
    ).with_columns(
        pl.Series("floor", preds[:, col[0.10]]).round(1),
        pl.Series("median", preds[:, col[0.50]]).round(1),
        pl.Series("ceiling", preds[:, col[0.90]]).round(1),
        *[pl.Series(f"q{int(q * 100):02d}", preds[:, i]).round(2) for i, q in enumerate(qs)],
        *[pl.Series(f"p_{t}plus", prob_over(preds, t, qs)).round(3) for t in THRESHOLDS],
    ).sort("median", descending=True)

    if args.show_actual:
        actual = all_games.filter((pl.col("season") == season) & (pl.col("week") == week))
        out = out.join(
            actual.select("player_id", pl.col("fantasy_points_ppr").round(1).alias("actual")),
            on="player_id", how="left",
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"predictions_{season}_wk{week:02d}"
    out.write_parquet(stem.with_suffix(".parquet"))
    out.write_csv(stem.with_suffix(".csv"))

    # ---- Print a readable summary ---------------------------------------------------------
    show = ["player_display_name", "team", "opponent_team", "floor", "median", "ceiling",
            "p_15plus", "p_20plus"] + (["actual"] if args.show_actual else [])
    with pl.Config(tbl_rows=12, tbl_cols=len(show), tbl_width_chars=140):
        for pos in ["QB", "RB", "WR", "TE"]:
            top = out.filter(pl.col("position") == pos).head(10).select(show).with_columns(
                (pl.col("p_15plus") * 100).round(0), (pl.col("p_20plus") * 100).round(0),
            )
            print(f"\n{pos} (top 10 by median)")
            print(top)

    print(f"\nSaved {out.height:,} players -> {stem.relative_to(ML_ROOT)}.parquet / .csv")


if __name__ == "__main__":
    main()