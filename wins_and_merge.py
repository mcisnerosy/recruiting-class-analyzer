import cfbd
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

configuration = cfbd.Configuration(
    access_token=os.getenv('CFBD_API_KEY')
)

recruiting_df = pd.read_csv('recruiting_classes.csv')
years = sorted(recruiting_df['year'].unique())

records = []

with cfbd.ApiClient(configuration) as api_client:
    games_api = cfbd.GamesApi(api_client)

    for year in years:
        try:
            team_records = games_api.get_records(year=int(year))
            for r in team_records:
                if r.total is None:
                    continue
                records.append({
                    'year': year,
                    'team': r.team,
                    'wins': r.total.wins,
                    'losses': r.total.losses,
                })
            print(f"pulled {year} — {len(team_records)} teams")
        except Exception as e:
            print(f"error on {year}: {e}")

wins_df = pd.DataFrame(records)

merged = recruiting_df.merge(wins_df, on=['year', 'team'], how='inner')
merged.to_csv('merged_data.csv', index=False)

print(f"\nmerged_data.csv: {len(merged)} rows, {merged['team'].nunique()} unique teams")
print(merged.head(10).to_string(index=False))
