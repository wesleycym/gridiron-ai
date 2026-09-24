# Readable terminal output [debugging]
import sys
sys.stdout.reconfigure(encoding="utf-8")

# Imports
import nflreadpy as nfl
import pandas as pd
import argparse
from pathlib import Path

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "player_stats"

teamStats = nfl.load_team_stats(None, 'week').to_pandas()
print("Team Columns:")
for column in teamStats.columns:
    print(column)
print('\n\n')
playerStats = nfl.load_player_stats(None, 'week').to_pandas()
print("Player Columns:")
for column in playerStats.columns:
    print(column)