import cfbd
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

configuration = cfbd.Configuration(
    access_token=os.getenv('CFBD_API_KEY')
)

records = []

with cfbd.ApiClient(configuration) as api_client:
    recruiting_api = cfbd.RecruitingApi(api_client)
    
    for year in range(2010, 2024):
        try:
            classes = recruiting_api.get_team_recruiting_rankings(year=year)
            for team in classes:
                records.append({
                    'year': year,
                    'team': team.team,
                    'points': team.points,
                    'rank': team.rank
                })
            print(f"pulled {year}")
        except Exception as e:
            print(f"error on {year}: {e}")

df = pd.DataFrame(records)
df.to_csv('recruiting_classes.csv', index=False)
print(df.head(20))