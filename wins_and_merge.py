# ─────────────────────────────────────────────────────────────────────────────
# wins_and_merge.py
# PURPOSE: Pull win/loss records for every team and year that exists in our
# recruiting CSV, then merge it with the recruiting data into one clean file.
# This gives us a single table with both recruiting rankings AND game results.
# ─────────────────────────────────────────────────────────────────────────────

import cfbd
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Authenticate with the CFBD API using the key from .env
configuration = cfbd.Configuration(
    access_token=os.getenv('CFBD_API_KEY')
)

# Load the recruiting CSV we built in data_pull.py
recruiting_df = pd.read_csv('recruiting_classes.csv')

# Get the unique list of years in our dataset, sorted oldest to newest
# We only pull win/loss data for years we actually have recruiting data for
years = sorted(recruiting_df['year'].unique())

# This list will hold one dictionary per team per year (win/loss record)
records = []

with cfbd.ApiClient(configuration) as api_client:

    # GamesApi handles game results and records
    games_api = cfbd.GamesApi(api_client)

    for year in years:
        try:
            # get_records() returns season win/loss records for all teams in a year
            # We cast year to int() because pandas stores numbers as numpy int64,
            # which the cfbd library doesn't accept — it needs a native Python int
            team_records = games_api.get_records(year=int(year))

            for r in team_records:
                # Some teams have incomplete records — skip them
                if r.total is None:
                    continue

                records.append({
                    'year':   year,
                    'team':   r.team,
                    'wins':   r.total.wins,    # total wins for the season
                    'losses': r.total.losses,  # total losses for the season
                })

            print(f"pulled {year} — {len(team_records)} teams")

        except Exception as e:
            print(f"error on {year}: {e}")

# Convert the win/loss records into a DataFrame
wins_df = pd.DataFrame(records)

# Merge the recruiting data with the win/loss data
# 'on' means match rows where both year AND team name are the same
# 'how=inner' means only keep rows that exist in BOTH tables
# (drops any team-year that has recruiting data but no win record, or vice versa)
merged = recruiting_df.merge(wins_df, on=['year', 'team'], how='inner')

# Save the merged table — this is the main dataset for all modeling
merged.to_csv('merged_data.csv', index=False)

print(f"\nmerged_data.csv: {len(merged)} rows, {merged['team'].nunique()} unique teams")
print(merged.head(10).to_string(index=False))
