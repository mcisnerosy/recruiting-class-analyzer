"""
prepare_data.py
PURPOSE: Run this script ONCE before launching the Streamlit app.
It pulls all game data and returning production from the API, builds
the same features used in game_model.py, and saves two CSV files:
  - games_with_features.csv  → training data for the model inside app.py
  - team_features.csv        → per-team stats the app uses for lookups

Why pre-compute? The Streamlit app would be slow and fragile if it called
the API every time someone opened it. Caching to CSV means the app loads
instantly and works even if the API is temporarily down.
"""
import cfbd
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Power 4 only — same filter as all other scripts in the pipeline
POWER4 = {
    'Alabama', 'Arkansas', 'Auburn', 'Florida', 'Georgia', 'Kentucky',
    'LSU', 'Mississippi State', 'Missouri', 'Ole Miss', 'South Carolina',
    'Tennessee', 'Texas', 'Texas A&M', 'Vanderbilt',
    'Illinois', 'Indiana', 'Iowa', 'Maryland', 'Michigan', 'Michigan State',
    'Minnesota', 'Nebraska', 'Northwestern', 'Ohio State', 'Oregon',
    'Penn State', 'Purdue', 'Rutgers', 'UCLA', 'USC', 'Washington', 'Wisconsin',
    'Baylor', 'BYU', 'Cincinnati', 'Colorado', 'Houston', 'Iowa State',
    'Kansas', 'Kansas State', 'Oklahoma', 'Oklahoma State', 'TCU',
    'Texas Tech', 'UCF', 'Utah', 'West Virginia',
    'Boston College', 'Clemson', 'Duke', 'Florida State', 'Georgia Tech',
    'Louisville', 'Miami', 'NC State', 'North Carolina', 'Notre Dame',
    'Pittsburgh', 'Stanford', 'Syracuse', 'Virginia', 'Virginia Tech', 'Wake Forest',
}

# ── Build 4-year rolling average recruiting rank ──────────────────────────────
# Same logic as game_model.py — averages the prior 4 years of recruiting classes
# to approximate the talent currently on a team's roster.
recruiting = pd.read_csv('recruiting_classes.csv')
recruiting = recruiting.sort_values(['team', 'year'])

roster_talent = []
for team, grp in recruiting.groupby('team'):
    grp = grp.set_index('year')  # index by year for easy slice lookups
    for year in grp.index:
        # The window is the 4 classes already playing (not the current year's recruits)
        window = [y for y in [year-1, year-2, year-3, year-4] if y in grp.index]
        if not window:
            continue
        roster_talent.append({
            'team': team, 'year': year,
            'avg_rank':   grp.loc[window, 'rank'].mean(),
            'avg_points': grp.loc[window, 'points'].mean(),
        })
talent_df = pd.DataFrame(roster_talent)

# ── Pull game results and returning production from the API ───────────────────
configuration = cfbd.Configuration(access_token=os.getenv('CFBD_API_KEY'))
game_records      = []
returning_records = []

with cfbd.ApiClient(configuration) as api_client:
    games_api   = cfbd.GamesApi(api_client)
    players_api = cfbd.PlayersApi(api_client)

    for year in range(2014, 2024):
        # Pull regular season games for this year
        try:
            for g in games_api.get_games(year=int(year), season_type='regular'):
                # Skip games without final scores or non-Power 4 matchups
                if g.home_points is None or g.away_points is None:
                    continue
                if g.home_team not in POWER4 or g.away_team not in POWER4:
                    continue
                game_records.append({
                    'year': year,
                    'home_team': g.home_team, 'away_team': g.away_team,
                    'home_points': g.home_points, 'away_points': g.away_points,
                    'neutral': g.neutral_site,
                })
        except Exception as e:
            print(f"error games {year}: {e}")

        # Pull returning production — percent of last season's value that's back
        try:
            for r in players_api.get_returning_production(year=int(year)):
                if r.team not in POWER4:
                    continue
                returning_records.append({
                    'year': year, 'team': r.team,
                    'ppa_return': r.percent_ppa,   # % of Predicted Points Added returning
                    'usage_return': r.usage,        # % of total usage returning
                })
        except Exception as e:
            print(f"error returning {year}: {e}")

        print(f"pulled {year}")

games_df     = pd.DataFrame(game_records)
returning_df = pd.DataFrame(returning_records)

# ── Merge everything into one game-level table ────────────────────────────────
# inner join on talent = only keep games where we have recruiting data for both teams
# left join on returning = keep all games even if returning data is missing
df = games_df.merge(
    talent_df.rename(columns={'team': 'home_team', 'avg_rank': 'home_avg_rank', 'avg_points': 'home_avg_points'}),
    on=['home_team', 'year'], how='inner'
).merge(
    talent_df.rename(columns={'team': 'away_team', 'avg_rank': 'away_avg_rank', 'avg_points': 'away_avg_points'}),
    on=['away_team', 'year'], how='inner'
).merge(
    returning_df.rename(columns={'team': 'home_team', 'ppa_return': 'home_ppa_return', 'usage_return': 'home_usage_return'}),
    on=['home_team', 'year'], how='left'
).merge(
    returning_df.rename(columns={'team': 'away_team', 'ppa_return': 'away_ppa_return', 'usage_return': 'away_usage_return'}),
    on=['away_team', 'year'], how='left'
)

# ── Compute differential features ─────────────────────────────────────────────
# The model predicts based on differences between teams, not raw values.
# Positive means home team advantage; negative means away team advantage.
df['rank_diff']         = df['away_avg_rank']    - df['home_avg_rank']
df['points_diff']       = df['home_avg_points']  - df['away_avg_points']
df['ppa_return_diff']   = df['home_ppa_return']  - df['away_ppa_return']
df['usage_return_diff'] = df['home_usage_return'] - df['away_usage_return']
df['home_win']          = (df['home_points'] > df['away_points']).astype(int)
df['neutral']           = df['neutral'].astype(int)

# ── Save the two output files ──────────────────────────────────────────────────
# team_features.csv: one row per team per year — the app uses this to look up
# each team's stats when a user selects teams in the dropdown menus.
# We join talent and returning production into one table for easy lookup.
talent_df.merge(
    returning_df, on=['team', 'year'], how='left'
).to_csv('team_features.csv', index=False)

# games_with_features.csv: one row per game — the app trains the model on this.
# Includes all the differential features the model needs to make predictions.
df.to_csv('games_with_features.csv', index=False)
print(f"\nSaved {len(df)} games to games_with_features.csv")
print(f"Saved {len(talent_df)} team-years to team_features.csv")
