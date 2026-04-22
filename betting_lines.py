# ─────────────────────────────────────────────────────────────────────────────
# betting_lines.py
# PURPOSE: Benchmark our game-level model against Vegas. Vegas sets spreads
# using enormous amounts of data, sharp money, and professional analysts —
# if our model can't beat Vegas, it means the signals we're using (recruiting
# rank, returning production) are already priced into the betting market.
# ─────────────────────────────────────────────────────────────────────────────

import cfbd
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
# LogisticRegression is the same classifier used in game_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

load_dotenv()

# Same Power 4 filter as all other scripts — keeps competition level consistent
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

# ── Rebuild 4-year rolling average recruiting rank (same logic as game_model.py) ─
# We re-derive the talent metric here so this script is self-contained.
# A college roster is roughly 4 years of recruiting classes, so averaging the
# last 4 years' recruiting rank approximates overall team talent.
recruiting = pd.read_csv('recruiting_classes.csv')
recruiting = recruiting.sort_values(['team', 'year'])

roster_talent = []
for team, grp in recruiting.groupby('team'):
    grp = grp.set_index('year')  # index by year so we can look up prior years easily
    for year in grp.index:
        # Pull the 4 prior years (not the current year — those recruits are already on campus)
        window = [y for y in [year-1, year-2, year-3, year-4] if y in grp.index]
        if not window:
            continue
        roster_talent.append({
            'team':       team,
            'year':       year,
            'avg_rank':   grp.loc[window, 'rank'].mean(),
            'avg_points': grp.loc[window, 'points'].mean(),
        })
talent_df = pd.DataFrame(roster_talent)

# ── Pull three data sources from the API ─────────────────────────────────────
configuration = cfbd.Configuration(access_token=os.getenv('CFBD_API_KEY'))
game_records      = []  # game results (scores, teams)
returning_records = []  # returning production (how experienced each roster is)
lines_records     = []  # Vegas betting lines (spread = point advantage)

with cfbd.ApiClient(configuration) as api_client:
    games_api   = cfbd.GamesApi(api_client)
    players_api = cfbd.PlayersApi(api_client)
    betting_api = cfbd.BettingApi(api_client)  # BettingApi is new — not in game_model.py

    for year in range(2014, 2024):
        # ── Pull game results ──────────────────────────────────────────────
        try:
            games = games_api.get_games(year=int(year), season_type='regular')
            for g in games:
                # Skip games without final scores or non-Power 4 matchups
                if g.home_points is None or g.away_points is None:
                    continue
                if g.home_team not in POWER4 or g.away_team not in POWER4:
                    continue
                game_records.append({
                    'year': year, 'game_id': g.id,  # game_id links this to the betting line
                    'home_team': g.home_team, 'away_team': g.away_team,
                    'home_points': g.home_points, 'away_points': g.away_points,
                    'neutral': g.neutral_site,
                })
        except Exception as e:
            print(f"error pulling games {year}: {e}")

        # ── Pull returning production ──────────────────────────────────────
        try:
            for r in players_api.get_returning_production(year=int(year)):
                if r.team not in POWER4:
                    continue
                returning_records.append({
                    'year': year, 'team': r.team,
                    'ppa_return': r.percent_ppa,  # % of last year's production returning
                    'usage_return': r.usage,
                })
        except Exception as e:
            print(f"error pulling returning production {year}: {e}")

        # ── Pull Vegas betting lines ───────────────────────────────────────
        # The spread tells us who Vegas thinks will win:
        #   spread < 0 → home team is favored (they "give" points)
        #   spread > 0 → home team is the underdog (they "receive" points)
        # Example: spread = -7.0 means the home team is favored by 7 points
        try:
            raw_lines = betting_api.get_lines(year=int(year), season_type='regular')
            count = 0
            for g in raw_lines:
                if g.home_team not in POWER4 or g.away_team not in POWER4:
                    continue
                if not g.lines:
                    continue
                # Multiple sportsbooks may report lines — prefer the consensus line
                # because it averages across all books, giving the most reliable signal.
                # If no consensus exists, fall back to the first available provider.
                line = next((l for l in g.lines if l.provider == 'consensus'), g.lines[0])
                if line.spread is None:
                    continue
                lines_records.append({
                    'year': year, 'game_id': g.id,  # matches the game_id in game_records
                    'spread': line.spread,
                    'provider': line.provider,
                    'over_under': line.over_under,
                })
                count += 1
            print(f"{year}: {count} games with lines")
        except Exception as e:
            print(f"error pulling lines {year}: {e}")

games_df      = pd.DataFrame(game_records)
returning_df  = pd.DataFrame(returning_records)
lines_df      = pd.DataFrame(lines_records)

# Quick sanity check on what we pulled
print(f"\nLines pulled: {len(lines_df)} | Games pulled: {len(games_df)}")
print("\nSample lines data:")
print(lines_df.head(10).to_string(index=False))
print(f"\nSpread distribution:\n{lines_df['spread'].describe().round(2)}")
print(f"\nProviders used:\n{lines_df['provider'].value_counts().to_string()}")

# ── Merge all three data sources into one table ───────────────────────────────
# Step 1: Join game results with betting lines (inner = only keep games with both)
df = games_df.merge(lines_df[['game_id', 'spread', 'over_under']], on='game_id', how='inner')

# Step 2: Add home team recruiting talent (inner = need both sides to build features)
# Step 3: Add away team recruiting talent
# Steps 4-5: Add returning production for home and away (left = keep game even if missing)
df = df.merge(
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

# ── Build differential features (same as game_model.py) ──────────────────────
# Positive rank_diff = home team has better recruiting (lower rank number = better)
df['rank_diff']         = df['away_avg_rank']    - df['home_avg_rank']
df['points_diff']       = df['home_avg_points']  - df['away_avg_points']
df['ppa_return_diff']   = df['home_ppa_return']  - df['away_ppa_return']
df['usage_return_diff'] = df['home_usage_return'] - df['away_usage_return']
df['home_win']          = (df['home_points'] > df['away_points']).astype(int)
df['neutral']           = df['neutral'].astype(int)

features = ['rank_diff', 'points_diff', 'neutral', 'ppa_return_diff', 'usage_return_diff']
df = df.dropna(subset=features).copy()

# ── Time-based train/test split ───────────────────────────────────────────────
# Same split as game_model.py — train on 2014–2020, evaluate on 2021–2023.
# We never touch the test set during training to avoid cheating.
train = df[df['year'] <= 2020]
test  = df[df['year'] >  2020]

# Train the same logistic regression classifier as game_model.py
lr = LogisticRegression()
lr.fit(train[features], train['home_win'])

# Generate predictions on the test set (years 2021-2023)
test = test.copy()
test['model_home_win_prob'] = lr.predict_proba(test[features])[:, 1]  # probability home team wins
test['model_pick_home']     = (test['model_home_win_prob'] > 0.5).astype(int)  # 1 if model picks home

# ── Derive Vegas's implied pick from the spread ───────────────────────────────
# If spread < 0, Vegas made the home team the favorite → Vegas "picks" home
# If spread > 0, Vegas made the away team the favorite → Vegas "picks" away
test['vegas_pick_home'] = (test['spread'] < 0).astype(int)

# Check whether each pick was correct
test['model_correct'] = (test['model_pick_home'] == test['home_win']).astype(int)
test['vegas_correct'] = (test['vegas_pick_home'] == test['home_win']).astype(int)

# ── Print the benchmark comparison ───────────────────────────────────────────
print("\n--- Benchmark: Model vs Vegas (2021–2023 test set) ---")
print(f"Total games:        {len(test)}")
print(f"Model accuracy:     {test['model_correct'].mean():.4f}")
print(f"Vegas accuracy:     {test['vegas_correct'].mean():.4f}")
print(f"Model AUC-ROC:      {roc_auc_score(test['home_win'], test['model_home_win_prob']):.4f}")

# ── Analyze disagreements — the most interesting cases ───────────────────────
# When our model and Vegas agree, there's no "edge" — both are saying the same thing.
# Disagreements are the only games where our model could potentially add value.
# If our model is right on disagreements more than 50% of the time, it has some edge.
disagree = test[test['model_pick_home'] != test['vegas_pick_home']].copy()
print(f"\nGames where model disagrees with Vegas: {len(disagree)}")
print(f"Model correct on disagreements:  {disagree['model_correct'].mean():.4f}")
print(f"Vegas correct on disagreements:  {disagree['vegas_correct'].mean():.4f}")

# Break down disagreements into two subcases:
# Case 1: Our model likes the home team but Vegas likes the away team
# Case 2: Our model likes the away team but Vegas likes the home team
model_likes_home = disagree[disagree['model_pick_home'] == 1]
model_likes_away = disagree[disagree['model_pick_home'] == 0]
print(f"\n  Model picks home, Vegas picks away ({len(model_likes_home)} games): model wins {model_likes_home['model_correct'].mean():.4f}")
print(f"  Model picks away, Vegas picks home ({len(model_likes_away)} games): model wins {model_likes_away['model_correct'].mean():.4f}")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Vegas spread vs model probability
# This shows where the two systems agree and where they diverge.
# Green zone (top-left): spread < 0 AND model prob > 0.5 → both pick home
# Red zone (bottom-right): spread > 0 AND model prob < 0.5 → both pick away
# Orange zones: they disagree — these are the potentially interesting games
axes[0].scatter(test['spread'], test['model_home_win_prob'], alpha=0.3, s=12,
                c=test['home_win'], cmap='RdYlGn')  # color = actual result (green=home won)
axes[0].axhline(0.5, color='black', linestyle='--', linewidth=0.8)  # model decision line
axes[0].axvline(0,   color='black', linestyle='--', linewidth=0.8)  # Vegas decision line
axes[0].set_xlabel('Vegas Spread (positive = home underdog)')
axes[0].set_ylabel('Model Home Win Probability')
axes[0].set_title('Model vs Vegas: Where Do They Disagree?')

# Shade the four quadrants to make them readable
axes[0].fill_between([-60, 0], 0.5, 1.0, alpha=0.05, color='green', label='Both pick home')
axes[0].fill_between([0,  60], 0.0, 0.5, alpha=0.05, color='red',   label='Both pick away')
axes[0].fill_between([-60, 0], 0.0, 0.5, alpha=0.08, color='orange', label='Disagree')
axes[0].fill_between([0,  60], 0.5, 1.0, alpha=0.08, color='orange')
axes[0].legend(fontsize=8)

# Plot 2: On disagreements, does model accuracy improve when Vegas is MORE confident?
# A large spread means Vegas is very confident — if our model still disagrees on those,
# it's likely wrong. Grouping by spread magnitude shows whether model accuracy varies.
disagree['spread_abs'] = disagree['spread'].abs()  # strip sign — just care about size
disagree['spread_bin'] = pd.cut(disagree['spread_abs'], bins=5)  # bucket by magnitude
model_by_bin = disagree.groupby('spread_bin')['model_correct'].mean()
vegas_by_bin = disagree.groupby('spread_bin')['vegas_correct'].mean()
x = range(len(model_by_bin))
axes[1].bar([i - 0.2 for i in x], model_by_bin, width=0.4, label='Model', color='steelblue')
axes[1].bar([i + 0.2 for i in x], vegas_by_bin, width=0.4, label='Vegas', color='coral')
axes[1].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
axes[1].set_xticks(list(x))
axes[1].set_xticklabels([str(b) for b in model_by_bin.index], rotation=30, ha='right', fontsize=8)
axes[1].set_title('Model vs Vegas Accuracy on Disagreements\n(by spread magnitude)')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('betting_lines_results.png', dpi=150)
print("\nPlots saved to betting_lines_results.png")

# Save all disagreement games to CSV so we can manually review specific cases
disagree[['year', 'home_team', 'away_team', 'home_points', 'away_points',
          'spread', 'model_home_win_prob', 'model_pick_home', 'vegas_pick_home',
          'model_correct', 'home_win']].sort_values('year').to_csv('disagreements.csv', index=False)
print("Disagreement games saved to disagreements.csv")
