# Run: { python ml/src/scripts/build_dataset.py } in terminal to build training dataset

'''
        build_dataset.py: takes raw weekly stats and turns them into training rows
            - Loads every cached season and keeps only regular-season QB/RB/WR/TE games
            - Builds player features from games before each week: averages over the last 3 and last 5 games, season-to-date averages, and recent volatility
            - Builds matchup features: how many fantasy points the opponent has recently allowed to that position
            - The game's actual stats and fantasy points, stored in columns starting with y_
'''

from pathlib import Path

import polars as pl

ML_ROOT = Path(__file__).resolve().parents[2]  # scripts -> src -> ml
RAW_DIR = ML_ROOT / "data" / "raw" / "player_stats"
OUT_PATH = ML_ROOT / "data" / "processed" / "training.parquet"

POSITIONS = ["QB", "RB", "WR", "TE"]
ROLL_WINDOWS = [3, 5]
MIN_PRIOR_GAMES = 3  # drop rows where the player has fewer career games than this

ID_COLS = [
    "player_id", "player_display_name", "position",
    "season", "week", "game_id", "team", "opponent_team",
]

# Stats we build rolling features from
FEATURE_STATS = [
    "fantasy_points_ppr",
    # usage
    "attempts", "carries", "targets", "receptions",
    "target_share", "air_yards_share", "wopr",
    # production
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds",
    # efficiency
    "passing_epa", "rushing_epa", "receiving_epa", "racr",
]

# What happened in the game itself. These are what the model predicts.
TARGET_STATS = [
    "fantasy_points_ppr", "fantasy_points",
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "fumbles_lost_total",
]


def load_raw() -> pl.DataFrame:
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw data in {RAW_DIR}. Run fetch_data.py first.")
    # diagonal_relaxed tolerates small schema differences between seasons
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def clean(raw: pl.DataFrame) -> pl.DataFrame:
    stat_cols = sorted(set(FEATURE_STATS) | set(TARGET_STATS))
    return (
        raw
        .filter(pl.col("season_type") == "REG")
        .filter(pl.col("position").is_in(POSITIONS))
        .select(ID_COLS + stat_cols)
        # A missing stat in a game the player appeared in means zero
        .with_columns(pl.col(stat_cols).cast(pl.Float64).fill_null(0.0).fill_nan(0.0))
        .sort(["player_id", "season", "week"])
    )


def add_player_features(df: pl.DataFrame) -> pl.DataFrame:
    """Rolling and season-to-date averages, always shifted by one game."""
    exprs = []

    # How many games this player has played before this one (career + this season)
    exprs.append(pl.int_range(pl.len()).over("player_id").alias("prior_games_career"))
    exprs.append(
        pl.int_range(pl.len()).over(["player_id", "season"]).alias("prior_games_season")
    )

    for stat in FEATURE_STATS:
        # Last N games played (can span into last season, which helps early weeks)
        for n in ROLL_WINDOWS:
            exprs.append(
                pl.col(stat).shift(1).rolling_mean(n).over("player_id")
                .alias(f"{stat}_avg{n}")
            )
        # Season-to-date ("szn") average before this game (null in a player's first game of a season)
        prev_sum = pl.col(stat).cum_sum().shift(1).over(["player_id", "season"])
        prev_n = pl.int_range(pl.len()).over(["player_id", "season"])
        exprs.append(
            pl.when(prev_n > 0).then(prev_sum / prev_n).otherwise(None)
            .alias(f"{stat}_szn")
        )

    # Volatility: how boom/bust has this player been lately?
    exprs.append(
        pl.col("fantasy_points_ppr").shift(1).rolling_std(5).over("player_id")
        .alias("fantasy_points_ppr_sd5")
    )

    return df.with_columns(exprs)


def add_matchup_features(df: pl.DataFrame) -> pl.DataFrame:
    """How many PPR points has this opponent allowed to this position, before this week?"""
    allowed = (
        df.group_by(["season", "week", "opponent_team", "position"])
        .agg(pl.col("fantasy_points_ppr").sum().alias("fp_allowed"))
        .sort(["opponent_team", "position", "season", "week"])
        .with_columns(
            pl.col("fp_allowed").shift(1).rolling_mean(4)
            .over(["opponent_team", "position"]).alias("opp_fp_allowed_avg4"),
            (
                pl.col("fp_allowed").cum_sum().shift(1)
                / pl.int_range(pl.len())
            ).over(["opponent_team", "position", "season"]).alias("opp_fp_allowed_szn"),
        )
        # first game of a season divides by zero -> treat as unknown
        .with_columns(
            pl.when(pl.col("opp_fp_allowed_szn").is_infinite() | pl.col("opp_fp_allowed_szn").is_nan())
            .then(None).otherwise(pl.col("opp_fp_allowed_szn")).alias("opp_fp_allowed_szn")
        )
        .drop("fp_allowed")
    )
    return df.join(allowed, on=["season", "week", "opponent_team", "position"], how="left")


def finalize(df: pl.DataFrame) -> pl.DataFrame:
    feature_cols = [
        c for c in df.columns
        if c.endswith(("_avg3", "_avg5", "_szn", "_sd5"))
        or c.startswith(("prior_games", "opp_"))
    ]
    targets = [pl.col(s).alias(f"y_{s}") for s in TARGET_STATS]

    return (
        df.filter(pl.col("prior_games_career") >= MIN_PRIOR_GAMES)
        .select(ID_COLS + feature_cols + targets)
        .sort(["season", "week", "player_id"])
    )


def main() -> None:
    raw = load_raw()
    print(f"Raw rows: {raw.height:,}")

    df = clean(raw)
    print(f"After filtering to REG season {', '.join(POSITIONS)}: {df.height:,}")

    df = add_player_features(df)
    df = add_matchup_features(df)
    df = finalize(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)

    n_features = len([c for c in df.columns if c not in ID_COLS and not c.startswith("y_")])
    print(f"Saved {df.height:,} rows, {n_features} features -> {OUT_PATH.relative_to(ML_ROOT)}")
    print(df.group_by("season").len().sort("season"))


if __name__ == "__main__":
    main()