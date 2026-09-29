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

ML_ROOT = Path(__file__).resolve().parents[2]  # scripts -> src -> ml
RAW_DIR = ML_ROOT / "data" / "raw" / "player_stats"
SCHEDULES_PATH = ML_ROOT / "data" / "raw" / "schedules.parquet"
INJURIES_PATH = ML_ROOT / "data" / "raw" / "injuries.parquet"


def cache_path(season: int) -> Path:
    return RAW_DIR / f"player_stats_{season}.parquet"


def update_player_stats(seasons: list[int], force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    current_season = nfl.get_current_season()

    for season in seasons:
        path = cache_path(season)
        # Finished seasons never change; the current one gets new games weekly.
        if path.exists() and season != current_season and not force:
            print(f"{season}: already cached, skipping")
            continue

        print(f"{season}: downloading...")
        try:
            df = nfl.load_player_stats(seasons=[season], summary_level="week")
        except Exception as e:
            print(f"{season}: failed ({e})")
            continue

        df.write_parquet(path)
        print(f"{season}: saved {df.height:,} rows -> {path.relative_to(ML_ROOT)}")


def update_schedules(seasons: list[int]) -> None:
    """Schedules are small, and lines for upcoming games change, so always refresh."""
    print("schedules: downloading...")
    df = nfl.load_schedules(seasons=seasons)
    SCHEDULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(SCHEDULES_PATH)
    print(f"schedules: saved {df.height:,} games -> {SCHEDULES_PATH.relative_to(ML_ROOT)}")


def update_injuries(season: int) -> None:
    """Official injury reports (Out / Doubtful / Questionable + practice status)."""
    print("injuries: downloading...")
    try:
        df = nfl.load_injuries(seasons=[season])
    except Exception as e:
        # Not fatal: reports may not be published yet, especially early in the week
        print(f"injuries: not available ({e})")
        return
    df.write_parquet(INJURIES_PATH)
    print(f"injuries: saved {df.height:,} rows -> {INJURIES_PATH.relative_to(ML_ROOT)}")


def main() -> None:
    current = nfl.get_current_season()

    parser = argparse.ArgumentParser(description="Fetch and cache NFL data.")
    parser.add_argument("--start", type=int, default=current - 4, help="first season to fetch")
    parser.add_argument("--end", type=int, default=current, help="last season to fetch")
    parser.add_argument("--force", action="store_true", help="re-download cached seasons")
    args = parser.parse_args()

    seasons = list(range(args.start, args.end + 1))
    print(f"Fetching seasons {seasons[0]}-{seasons[-1]}\n")
    update_player_stats(seasons, force=args.force)
    update_schedules(seasons)
    update_injuries(current)


if __name__ == "__main__":
    main()