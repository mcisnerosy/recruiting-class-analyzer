import cfbd
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

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

# --- Rebuild model features (same as game_model.py) ---
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
            'team':       team,
            'year':       year,
            'avg_rank':   grp.loc[window, 'rank'].mean(),
            'avg_points': grp.loc[window, 'points'].mean(),
        })
talent_df = pd.DataFrame(roster_talent)

# --- Pull games, returning production, and betting lines ---
configuration = cfbd.Configuration(access_token=os.getenv('CFBD_API_KEY'))
game_records     = []
returning_records = []
lines_records    = []

with cfbd.ApiClient(configuration) as api_client:
    games_api   = cfbd.GamesApi(api_client)
    players_api = cfbd.PlayersApi(api_client)
    betting_api = cfbd.BettingApi(api_client)

    for year in range(2014, 2024):
        # Games
        try:
            games = games_api.get_games(year=int(year), season_type='regular')
            for g in games:
                if g.home_points is None or g.away_points is None:
                    continue
                if g.home_team not in POWER4 or g.away_team not in POWER4:
                    continue
                game_records.append({
                    'year': year, 'game_id': g.id,
                    'home_team': g.home_team, 'away_team': g.away_team,
                    'home_points': g.home_points, 'away_points': g.away_points,
                    'neutral': g.neutral_site,
                })
        except Exception as e:
            print(f"error pulling games {year}: {e}")

        # Returning production
        try:
            for r in players_api.get_returning_production(year=int(year)):
                if r.team not in POWER4:
                    continue
                returning_records.append({
                    'year': year, 'team': r.team,
                    'ppa_return': r.percent_ppa, 'usage_return': r.usage,
                })
        except Exception as e:
            print(f"error pulling returning production {year}: {e}")

        # Betting lines
        try:
            raw_lines = betting_api.get_lines(year=int(year), season_type='regular')
            count = 0
            for g in raw_lines:
                if g.home_team not in POWER4 or g.away_team not in POWER4:
                    continue
                if not g.lines:
                    continue
                # Prefer consensus, fall back to first available provider
                line = next((l for l in g.lines if l.provider == 'consensus'), g.lines[0])
                if line.spread is None:
                    continue
                lines_records.append({
                    'year': year, 'game_id': g.id,
                    'spread': line.spread,       # positive = home team underdog
                    'provider': line.provider,
                    'over_under': line.over_under,
                })
                count += 1
            print(f"{year}: {count} games with lines")
        except Exception as e:
            print(f"error pulling lines {year}: {e}")

games_df     = pd.DataFrame(game_records)
returning_df = pd.DataFrame(returning_records)
lines_df     = pd.DataFrame(lines_records)

print(f"\nLines pulled: {len(lines_df)} | Games pulled: {len(games_df)}")
print("\nSample lines data:")
print(lines_df.head(10).to_string(index=False))
print(f"\nSpread distribution:\n{lines_df['spread'].describe().round(2)}")
print(f"\nProviders used:\n{lines_df['provider'].value_counts().to_string()}")

# --- Merge everything ---
df = games_df.merge(lines_df[['game_id', 'spread', 'over_under']], on='game_id', how='inner')

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

df['rank_diff']         = df['away_avg_rank']    - df['home_avg_rank']
df['points_diff']       = df['home_avg_points']  - df['away_avg_points']
df['ppa_return_diff']   = df['home_ppa_return']  - df['away_ppa_return']
df['usage_return_diff'] = df['home_usage_return'] - df['away_usage_return']
df['home_win']          = (df['home_points'] > df['away_points']).astype(int)
df['neutral']           = df['neutral'].astype(int)

features = ['rank_diff', 'points_diff', 'neutral', 'ppa_return_diff', 'usage_return_diff']
df = df.dropna(subset=features).copy()

# --- Train model and generate predictions ---
train = df[df['year'] <= 2020]
test  = df[df['year'] >  2020]

lr = LogisticRegression()
lr.fit(train[features], train['home_win'])
test = test.copy()
test['model_home_win_prob'] = lr.predict_proba(test[features])[:, 1]
test['model_pick_home']     = (test['model_home_win_prob'] > 0.5).astype(int)

# Vegas pick: spread < 0 → home team favored → Vegas picks home
test['vegas_pick_home'] = (test['spread'] < 0).astype(int)
test['model_correct']   = (test['model_pick_home'] == test['home_win']).astype(int)
test['vegas_correct']   = (test['vegas_pick_home'] == test['home_win']).astype(int)

# --- Benchmark ---
print("\n--- Benchmark: Model vs Vegas (2021–2023 test set) ---")
print(f"Total games:        {len(test)}")
print(f"Model accuracy:     {test['model_correct'].mean():.4f}")
print(f"Vegas accuracy:     {test['vegas_correct'].mean():.4f}")
print(f"Model AUC-ROC:      {roc_auc_score(test['home_win'], test['model_home_win_prob']):.4f}")

# Games where model and Vegas disagree
disagree = test[test['model_pick_home'] != test['vegas_pick_home']].copy()
print(f"\nGames where model disagrees with Vegas: {len(disagree)}")
print(f"Model correct on disagreements:  {disagree['model_correct'].mean():.4f}")
print(f"Vegas correct on disagreements:  {disagree['vegas_correct'].mean():.4f}")

# Disagreements where model favors home but Vegas favors away
model_likes_home = disagree[disagree['model_pick_home'] == 1]
model_likes_away = disagree[disagree['model_pick_home'] == 0]
print(f"\n  Model picks home, Vegas picks away ({len(model_likes_home)} games): model wins {model_likes_home['model_correct'].mean():.4f}")
print(f"  Model picks away, Vegas picks home ({len(model_likes_away)} games): model wins {model_likes_away['model_correct'].mean():.4f}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1. Spread vs model probability
axes[0].scatter(test['spread'], test['model_home_win_prob'], alpha=0.3, s=12,
                c=test['home_win'], cmap='RdYlGn')
axes[0].axhline(0.5, color='black', linestyle='--', linewidth=0.8)
axes[0].axvline(0,   color='black', linestyle='--', linewidth=0.8)
axes[0].set_xlabel('Vegas Spread (positive = home underdog)')
axes[0].set_ylabel('Model Home Win Probability')
axes[0].set_title('Model vs Vegas: Where Do They Disagree?')

# Shade disagreement zones
axes[0].fill_between([-60, 0], 0.5, 1.0, alpha=0.05, color='green', label='Both pick home')
axes[0].fill_between([0,  60], 0.0, 0.5, alpha=0.05, color='red',   label='Both pick away')
axes[0].fill_between([-60, 0], 0.0, 0.5, alpha=0.08, color='orange', label='Disagree')
axes[0].fill_between([0,  60], 0.5, 1.0, alpha=0.08, color='orange')
axes[0].legend(fontsize=8)

# 2. Model accuracy vs Vegas accuracy on disagreements by spread magnitude
disagree['spread_abs'] = disagree['spread'].abs()
disagree['spread_bin'] = pd.cut(disagree['spread_abs'], bins=5)
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

# Save disagreement games for inspection
disagree[['year', 'home_team', 'away_team', 'home_points', 'away_points',
          'spread', 'model_home_win_prob', 'model_pick_home', 'vegas_pick_home',
          'model_correct', 'home_win']].sort_values('year').to_csv('disagreements.csv', index=False)
print("Disagreement games saved to disagreements.csv")
