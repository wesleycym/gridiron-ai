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
SCHEDULES_PATH = ML_ROOT / "data" / "raw" / "schedules.parquet"
OUT_PATH = ML_ROOT / "data" / "processed" / "training.parquet"

POSITIONS = ["QB", "RB", "WR", "TE"]
ROLL_WINDOWS = [3, 5]
MIN_PRIOR_GAMES = 3  # drop rows where the player has fewer career games than this

# Week 18: teams with nothing to play for rest starters, so those games don't look
# like normal games. They're removed everywhere (targets AND feature history).
EXCLUDE_WEEKS = [18]

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

# Long-memory features: a player's established level, so one bad stretch or an
# injury-shortened season doesn't make the model forget who he is.
LONG_STATS = [
    "fantasy_points_ppr", "attempts", "carries", "targets", "target_share",
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds", "receiving_yards",
]
LONG_WINDOWS = [16, 48]  # ~one season, ~three seasons of games played

# What happened in the game itself. These are what the model predicts.
TARGET_STATS = [
    "fantasy_points_ppr", "fantasy_points",
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "fumbles_lost_total",
]

STAT_COLS = sorted(set(FEATURE_STATS) | set(TARGET_STATS))


# Loading
def load_raw() -> pl.DataFrame:
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw data in {RAW_DIR}. Run fetch_data.py first.")
    # diagonal_relaxed tolerates small schema differences between seasons
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def load_schedules() -> pl.DataFrame | None:
    if not SCHEDULES_PATH.exists():
        print("WARNING: no schedules.parquet found, Vegas features skipped. Run fetch_data.py.")
        return None
    return pl.read_parquet(SCHEDULES_PATH)


def clean(raw: pl.DataFrame) -> pl.DataFrame:
    return (
        raw
        .filter(pl.col("season_type") == "REG")
        .filter(pl.col("position").is_in(POSITIONS))
        .filter(~pl.col("week").is_in(EXCLUDE_WEEKS))
        .select(ID_COLS + STAT_COLS)
        # A missing stat in a game the player appeared in means zero
        .with_columns(pl.col(STAT_COLS).cast(pl.Float64).fill_null(0.0).fill_nan(0.0))
        .sort(["player_id", "season", "week"])
    )


# Features
def add_player_features(df: pl.DataFrame) -> pl.DataFrame:
    """Rolling and season-to-date averages, always shifted by one game."""
    df = df.sort(["player_id", "season", "week"])
    exprs = [
        # How many games this player has played before this one
        pl.int_range(pl.len()).over("player_id").alias("prior_games_career"),
        pl.int_range(pl.len()).over(["player_id", "season"]).alias("prior_games_season"),
    ]

    for stat in FEATURE_STATS:
        # Last N games played (can span into last season, which helps early weeks)
        for n in ROLL_WINDOWS:
            exprs.append(
                pl.col(stat).shift(1).rolling_mean(n).over("player_id")
                .alias(f"{stat}_avg{n}")
            )
        # Season-to-date ("szn") average before this game (null in first game of a season)
        prev_sum = pl.col(stat).cum_sum().shift(1).over(["player_id", "season"])
        prev_n = pl.int_range(pl.len()).over(["player_id", "season"])
        exprs.append(
            pl.when(prev_n > 0).then(prev_sum / prev_n).otherwise(None)
            .alias(f"{stat}_szn")
        )

    # Long windows: average over the last 16 / 48 games played, using however many
    # games exist if fewer (so a player with 10 career games still gets a value).
    n_prior = pl.int_range(pl.len()).over("player_id")
    for stat in LONG_STATS:
        total_before = pl.col(stat).cum_sum().shift(1).over("player_id")
        for n in LONG_WINDOWS:
            total_before_window = pl.col(stat).cum_sum().shift(n + 1).over("player_id").fill_null(0)
            count = pl.min_horizontal(n_prior, pl.lit(n))
            exprs.append(
                pl.when(count > 0).then((total_before - total_before_window) / count)
                .otherwise(None).alias(f"{stat}_avg{n}")
            )

    # Volatility: how boom/bust has this player been lately?
    exprs.append(
        pl.col("fantasy_points_ppr").shift(1).rolling_std(5).over("player_id")
        .alias("fantasy_points_ppr_sd5")
    )
    df = df.with_columns(exprs)

    # Previous season's per-game averages (whole season, already finished = no leakage)
    prev = (
        df.group_by(["player_id", "season"])
        .agg(
            pl.len().alias("prior_games_prev_season"),
            *[pl.col(stat).mean().alias(f"{stat}_prev") for stat in LONG_STATS],
        )
        .with_columns((pl.col("season") + 1).alias("season"))
    )
    return df.join(prev, on=["player_id", "season"], how="left")


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


def team_game_lines(schedules: pl.DataFrame) -> pl.DataFrame:
    """
    One row per team per regular-season game, from that team's point of view.
    nflverse convention: spread_line > 0 means the HOME team is favored.
    We flip it so vegas_spread > 0 always means "this team is favored".
    """
    games = (
        schedules
        .filter(pl.col("game_type") == "REG")
        .select("season", "week", "game_id", "home_team", "away_team", "spread_line", "total_line")
        .with_columns(pl.col("spread_line", "total_line").cast(pl.Float64))
    )
    home = games.select(
        "season", "week", "game_id",
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent_team"),
        pl.col("spread_line").alias("vegas_spread"),
        pl.col("total_line").alias("vegas_total"),
        pl.lit(1).alias("vegas_is_home"),
    )
    away = games.select(
        "season", "week", "game_id",
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent_team"),
        (-pl.col("spread_line")).alias("vegas_spread"),
        pl.col("total_line").alias("vegas_total"),
        pl.lit(0).alias("vegas_is_home"),
    )
    return pl.concat([home, away]).with_columns(
        # How many points Vegas expects this team to score
        ((pl.col("vegas_total") + pl.col("vegas_spread")) / 2).alias("vegas_implied_total")
    )


def add_vegas_features(df: pl.DataFrame, schedules: pl.DataFrame | None) -> pl.DataFrame:
    """Pre-game betting lines. These are set before kickoff, so no leakage."""
    if schedules is None:
        return df
    lines = team_game_lines(schedules).select(
        "season", "week", "team",
        "vegas_spread", "vegas_total", "vegas_is_home", "vegas_implied_total",
    )
    out = df.join(lines, on=["season", "week", "team"], how="left")
    missing = out["vegas_total"].null_count()
    if missing:
        print(f"Note: {missing:,} rows had no Vegas line (left as null)")
    return out


def build_features(games: pl.DataFrame, schedules: pl.DataFrame | None) -> pl.DataFrame:
    """All feature steps in order. Used by both this script and predict.py."""
    df = add_player_features(games)
    df = add_matchup_features(df)
    df = add_vegas_features(df, schedules)
    return df


def feature_columns(df: pl.DataFrame) -> list[str]:
    """The naming rule that decides which columns are model features."""
    return [
        c for c in df.columns
        if c.endswith(("_avg3", "_avg5", "_avg16", "_avg48", "_szn", "_sd5", "_prev"))
        or c.startswith(("prior_games", "opp_", "vegas_"))
    ]


# Training dataset
def finalize(df: pl.DataFrame) -> pl.DataFrame:
    targets = [pl.col(s).alias(f"y_{s}") for s in TARGET_STATS]
    return (
        df.filter(pl.col("prior_games_career") >= MIN_PRIOR_GAMES)
        .select(ID_COLS + feature_columns(df) + targets)
        .sort(["season", "week", "player_id"])
    )


def main() -> None:
    raw = load_raw()
    print(f"Raw rows: {raw.height:,}")

    games = clean(raw)
    print(f"After filtering to REG season {', '.join(POSITIONS)}, "
          f"excluding week(s) {EXCLUDE_WEEKS}: {games.height:,}")

    df = finalize(build_features(games, load_schedules()))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)

    n_features = len(feature_columns(df))
    print(f"Saved {df.height:,} rows, {n_features} features -> {OUT_PATH.relative_to(ML_ROOT)}")
    print(df.group_by("season").len().sort("season"))


if __name__ == "__main__":
    main()