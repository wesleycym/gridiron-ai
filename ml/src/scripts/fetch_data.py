"""
Fetch weekly NFL player stats from nflverse and cache them locally as parquet.

Usage (run from the project root):
    python scripts/fetch_data.py                 # last 5 seasons (default)
    python scripts/fetch_data.py --start 2018    # 2018 through the current season
    python scripts/fetch_data.py --force         # re-download everything
"""

import argparse
from pathlib import Path

import nflreadpy as nfl

# Resolve paths
ML_ROOT = Path(__file__).resolve().parents[2]   # scripts -> src -> ml
RAW_DIR = ML_ROOT / "data" / "raw" / "player_stats"

def cache_path(season: int) -> Path:
    """One file per season, e.g. data/raw/player_stats/player_stats_2024.parquet"""
    return RAW_DIR / f"player_stats_{season}.parquet"


def fetch_season(season: int):
    """Download one season of player stats, one row per player per game."""
    return nfl.load_player_stats(seasons=[season], summary_level="week")


def update_cache(seasons: list[int], force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    current_season = nfl.get_current_season()

    for season in seasons:
        path = cache_path(season)

        # Refresh for every new game | Skip already cached games
        is_current = season == current_season
        if path.exists() and not is_current and not force:
            print(f"{season}: already cached, skipping")
            continue

        print(f"{season}: downloading...")
        try:
            df = fetch_season(season)
        except Exception as e:
            # If seasons hasn't started yet
            print(f"{season}: failed ({e})")
            continue

        df.write_parquet(path)
        print(f"{season}: saved {df.height:,} rows -> {path.relative_to(ML_ROOT)}")


def main() -> None:
    current = nfl.get_current_season()

    parser = argparse.ArgumentParser(description="Fetch and cache weekly player stats.")
    parser.add_argument("--start", type=int, default=current - 4, help="first season to fetch")
    parser.add_argument("--end", type=int, default=current, help="last season to fetch")
    parser.add_argument("--force", action="store_true", help="re-download cached seasons")
    args = parser.parse_args()

    seasons = list(range(args.start, args.end + 1))
    print(f"Fetching seasons {seasons[0]}-{seasons[-1]}\n")
    update_cache(seasons, force=args.force)


if __name__ == "__main__":
    main()