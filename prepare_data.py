"""
Run this once to pull game data + returning production and cache to
games_with_features.csv. The Streamlit app loads from that file.
"""
import cfbd
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

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

# 4-year rolling avg recruiting rank per team-year
recruiting = pd.read_csv('recruiting_classes.csv')
recruiting = recruiting.sort_values(['team', 'year'])

roster_talent = []
for team, grp in recruiting.groupby('team'):
    grp = grp.set_index('year')
    for year in grp.index:
        window = [y for y in [year-1, year-2, year-3, year-4] if y in grp.index]
        if not window:
            continue
        roster_talent.append({
            'team': team, 'year': year,
            'avg_rank':   grp.loc[window, 'rank'].mean(),
            'avg_points': grp.loc[window, 'points'].mean(),
        })
talent_df = pd.DataFrame(roster_talent)

configuration = cfbd.Configuration(access_token=os.getenv('CFBD_API_KEY'))
game_records      = []
returning_records = []

with cfbd.ApiClient(configuration) as api_client:
    games_api   = cfbd.GamesApi(api_client)
    players_api = cfbd.PlayersApi(api_client)

    for year in range(2014, 2024):
        try:
            for g in games_api.get_games(year=int(year), season_type='regular'):
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

        try:
            for r in players_api.get_returning_production(year=int(year)):
                if r.team not in POWER4:
                    continue
                returning_records.append({
                    'year': year, 'team': r.team,
                    'ppa_return': r.percent_ppa, 'usage_return': r.usage,
                })
        except Exception as e:
            print(f"error returning {year}: {e}")

        print(f"pulled {year}")

games_df     = pd.DataFrame(game_records)
returning_df = pd.DataFrame(returning_records)

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

df['rank_diff']         = df['away_avg_rank']    - df['home_avg_rank']
df['points_diff']       = df['home_avg_points']  - df['away_avg_points']
df['ppa_return_diff']   = df['home_ppa_return']  - df['away_ppa_return']
df['usage_return_diff'] = df['home_usage_return'] - df['away_usage_return']
df['home_win']          = (df['home_points'] > df['away_points']).astype(int)
df['neutral']           = df['neutral'].astype(int)

# Also save team features for the app to look up individual team stats
talent_df.merge(
    returning_df, on=['team', 'year'], how='left'
).to_csv('team_features.csv', index=False)

df.to_csv('games_with_features.csv', index=False)
print(f"\nSaved {len(df)} games to games_with_features.csv")
print(f"Saved {len(talent_df)} team-years to team_features.csv")
