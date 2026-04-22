# ─────────────────────────────────────────────────────────────────────────────
# data_pull.py
# PURPOSE: Pull 14 years of college football recruiting class rankings from the
# CFBD API and save them to a CSV file. This is the first step in the pipeline
# — everything else depends on this data.
# ─────────────────────────────────────────────────────────────────────────────

# cfbd is the Python library for the College Football Data API
# pandas is used to organize and save the data as a table
# os lets us read environment variables (like the API key)
# dotenv reads the .env file so the API key is available via os.getenv()
import cfbd
import pandas as pd
import os
from dotenv import load_dotenv

# Load the .env file so CFBD_API_KEY becomes available as an environment variable
load_dotenv()

# Set up authentication — the API requires a key to allow requests
# os.getenv() reads the key from the .env file without hardcoding it in the code
configuration = cfbd.Configuration(
    access_token=os.getenv('CFBD_API_KEY')
)

# This list will collect one dictionary per team per year
records = []

# Open a connection to the CFBD API using our credentials
with cfbd.ApiClient(configuration) as api_client:

    # RecruitingApi is the specific part of the API that handles recruiting data
    recruiting_api = cfbd.RecruitingApi(api_client)

    # Loop through every year from 2010 to 2023 (range stops before 2024)
    for year in range(2010, 2024):
        try:
            # Pull the recruiting class rankings for this year
            # Returns a list of team objects, one per school
            classes = recruiting_api.get_team_recruiting_rankings(year=year)

            # Loop through each team's recruiting data for this year
            for team in classes:
                records.append({
                    'year':   year,
                    'team':   team.team,    # school name (e.g. 'Alabama')
                    'points': team.points,  # composite recruiting score (higher = better)
                    'rank':   team.rank     # national recruiting rank (lower = better)
                })
            print(f"pulled {year}")

        except Exception as e:
            # If the API call fails for any reason, print the error and keep going
            print(f"error on {year}: {e}")

# Convert the list of dictionaries into a pandas DataFrame (a table)
df = pd.DataFrame(records)

# Save the table to a CSV file — index=False means don't write row numbers
df.to_csv('recruiting_classes.csv', index=False)

# Preview the first 20 rows to confirm it looks right
print(df.head(20))
