# Readable terminal output
import sys
sys.stdout.reconfigure(encoding="utf-8")

# APIs
import nflreadpy as nfl
import pandas as pd

pbp = nfl.load_pbp() # Load current season play-by-play data
pbp_pandas = pbp.to_pandas() # Convert to pandas

stats = nfl.load_team_stats(None, 'week').to_pandas() # Load current season team stats & convert to pandas

bills = stats[stats['team'] == 'BUF'] # Filter for Buffalo Bills stats


for column in bills.columns:
    print(column)