# Script to view relevant player data from the database
'''
python ml/src/scripts/player.py "Puka Nacua"
python ml/src/scripts/player.py nacua                  # partial names work
python ml/src/scripts/player.py "Josh Allen" --games 10
python ml/src/scripts/player.py mahomes --season 2025 --week 12   # any past week
'''

import argparse
import sys

import polars as pl

from build_dataset import clean, load_raw, load_schedules
from predict import THRESHOLDS, next_unplayed_week, predict_week

# Which stats to show in the game log, by position
LOG_COLS = {
    "QB": ["completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
           "carries", "rushing_yards", "rushing_tds"],
    "RB": ["carries", "rushing_yards", "rushing_tds", "targets", "receptions",
           "receiving_yards", "receiving_tds"],
    "WR": ["targets", "target_share", "receptions", "receiving_yards", "receiving_tds",
           "air_yards_share"],
    "TE": ["targets", "target_share", "receptions", "receiving_yards", "receiving_tds",
           "air_yards_share"],
}
SHORT = {  # shorter column headers for printing
    "completions": "cmp", "attempts": "att", "passing_yards": "pass_yd", "passing_tds": "pass_td",
    "passing_interceptions": "int", "carries": "car", "rushing_yards": "rush_yd",
    "rushing_tds": "rush_td", "targets": "tgt", "target_share": "tgt_sh", "receptions": "rec",
    "receiving_yards": "rec_yd", "receiving_tds": "rec_td", "air_yards_share": "ay_sh",
    "fantasy_points_ppr": "ppr",
}


def find_player(raw: pl.DataFrame, query: str) -> pl.DataFrame:
    """Match by partial name (case-insensitive). Returns one row per matching player."""
    q = query.lower().strip()
    return (
        raw.filter(pl.col("position").is_in(list(LOG_COLS)))
        .filter(pl.col("player_display_name").str.to_lowercase().str.contains(q, literal=True))
        .sort(["season", "week"])
        .group_by("player_id")
        .agg(
            pl.col("player_display_name").last(),
            pl.col("position").last(),
            pl.col("team").last(),
            pl.col("season").last().alias("last_season"),
            pl.len().alias("games"),
        )
        .sort(["last_season", "games"], descending=True)
    )


def opponent_rank(games: pl.DataFrame, season: int, week: int, position: str, opponent: str):
    """How generous is this opponent to the position this season? 1 = allows the most points."""
    allowed = (
        games.filter((pl.col("season") == season) & (pl.col("week") < week)
                     & (pl.col("position") == position))
        .group_by(["opponent_team", "week"])
        .agg(pl.col("fantasy_points_ppr").sum().alias("pts"))
        .group_by("opponent_team")
        .agg(pl.col("pts").mean().alias("per_game"))
        .sort("per_game", descending=True)
        .with_row_index("rank", offset=1)
    )
    row = allowed.filter(pl.col("opponent_team") == opponent)
    if row.is_empty():
        return None, None, allowed.height
    return int(row["rank"][0]), float(row["per_game"][0]), allowed.height


def line(label: str, value: str) -> None:
    print(f"  {label:<22}{value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scouting report for one player.")
    parser.add_argument("name", help="player name or part of it, e.g. 'nacua'")
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--games", type=int, default=6, help="how many recent games to show")
    args = parser.parse_args()

    raw = load_raw().filter(pl.col("season_type") == "REG")
    matches = find_player(raw, args.name)
    if matches.is_empty():
        sys.exit(f"No QB/RB/WR/TE found matching '{args.name}'.")
    if matches.height > 1:
        exact = matches.filter(pl.col("player_display_name").str.to_lowercase() == args.name.lower())
        if exact.height == 1:
            matches = exact
        else:
            print(f"Several players match '{args.name}'. Be more specific:")
            for r in matches.head(10).iter_rows(named=True):
                print(f"  {r['player_display_name']:<25} {r['position']:<3} {r['team']:<4} "
                      f"(last played {r['last_season']}, {r['games']} games)")
            sys.exit(1)

    p = matches.row(0, named=True)
    pid, name, pos = p["player_id"], p["player_display_name"], p["position"]

    schedules = load_schedules()
    if args.season and args.week:
        season, week = args.season, args.week
    else:
        season, week = next_unplayed_week(schedules)

    # ---- Header -----------------------------------------------------------------------
    print(f"\n{'=' * 70}\n  {name}  |  {pos}  |  {p['team']}\n{'=' * 70}")

    # ---- This week's projection ---------------------------------------------------------
    games = clean(raw)
    preds = predict_week(season, week, schedules, games)
    me = preds.filter(pl.col("player_id") == pid)

    print(f"\nWEEK {week}, {season}")
    if me.is_empty():
        print("  No projection: his team may be on bye, he hasn't played in the last few weeks,")
        print("  or he has fewer than 3 career games.")
    else:
        r = me.row(0, named=True)
        where = "vs" if r["vegas_is_home"] == 1 else "@"
        line("Matchup", f"{where} {r['opponent_team']}")
        if r["vegas_total"] is not None:
            fav = (f"favored by {r['vegas_spread']:.1f}" if r["vegas_spread"] > 0
                   else f"underdog by {-r['vegas_spread']:.1f}" if r["vegas_spread"] < 0 else "pick'em")
            line("Vegas", f"total {r['vegas_total']:.1f}, team {fav}, "
                          f"implied {r['vegas_implied_total']:.1f} pts")
        rank, per_game, n = opponent_rank(games, season, week, pos, r["opponent_team"])
        if rank:
            label = "soft" if rank <= n / 3 else "tough" if rank > 2 * n / 3 else "average"
            line(f"{r['opponent_team']} vs {pos}s",
                 f"allows {per_game:.1f} PPR/game, #{rank} of {n} ({label} matchup)")

        # Rank among same-position players this week
        pos_rank = (preds.filter(pl.col("position") == pos)
                    .with_row_index("r", offset=1).filter(pl.col("player_id") == pid)["r"][0])
        print()
        line("Projection (median)", f"{r['median']:.1f} PPR   ({pos}{pos_rank} this week)")
        line("Range", f"floor {r['floor']:.1f}  |  ceiling {r['ceiling']:.1f}  (80% of outcomes)")
        line("Wider range", f"{r['q05']:.1f} to {r['q95']:.1f}  (90% of outcomes)")
        print()
        for t in THRESHOLDS:
            pct = r[f"p_{t}plus"] * 100
            bar = "#" * round(pct / 4)
            line(f"Chance of {t}+ pts", f"{pct:4.0f}%  {bar}")

    # ---- Recent game log ----------------------------------------------------------------
    stat_cols = LOG_COLS[pos]
    before = (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
    mine = raw.filter((pl.col("player_id") == pid) & before).sort(["season", "week"])

    log = (
        mine.tail(args.games)
        .select(
            "season", "week", "opponent_team",
            *[pl.col(c).fill_null(0).round(2) if "share" in c
              else pl.col(c).fill_null(0).round(0).cast(pl.Int64) for c in stat_cols],
            pl.col("fantasy_points_ppr").round(1),
        )
        .rename({c: SHORT.get(c, c) for c in stat_cols + ["fantasy_points_ppr"]})
        .rename({"opponent_team": "opp"})
    )
    print(f"\nLAST {log.height} GAMES")
    with pl.Config(tbl_rows=args.games, tbl_cols=20, tbl_width_chars=140,
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print(log)

    # ---- Season summaries -----------------------------------------------------------------
    summary = (
        mine.group_by("season")
        .agg(
            pl.len().alias("games"),
            pl.col("fantasy_points_ppr").mean().round(1).alias("ppr"),
            pl.col("fantasy_points_ppr").std().round(1).alias("sd"),
            (pl.col("fantasy_points_ppr") >= 15).mean().mul(100).round(0).cast(pl.Int64).alias("15+%"),
            (pl.col("fantasy_points_ppr") >= 20).mean().mul(100).round(0).cast(pl.Int64).alias("20+%"),
            *[pl.col(c).fill_null(0).mean().round(2 if "share" in c else 1).alias(SHORT.get(c, c))
              for c in stat_cols],
        )
        .sort("season", descending=True)
        .head(3)
    )
    print("\nSEASON AVERAGES (per game; sd = how much his scores swing, 15+% = share of games with 15+ pts)")
    with pl.Config(tbl_cols=20, tbl_width_chars=140,
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print(summary)
    print()


if __name__ == "__main__":
    main()