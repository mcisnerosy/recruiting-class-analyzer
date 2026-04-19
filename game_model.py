import cfbd
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt

load_dotenv()

POWER4 = {
    # SEC
    'Alabama', 'Arkansas', 'Auburn', 'Florida', 'Georgia', 'Kentucky',
    'LSU', 'Mississippi State', 'Missouri', 'Ole Miss', 'South Carolina',
    'Tennessee', 'Texas', 'Texas A&M', 'Vanderbilt',
    # Big Ten
    'Illinois', 'Indiana', 'Iowa', 'Maryland', 'Michigan', 'Michigan State',
    'Minnesota', 'Nebraska', 'Northwestern', 'Ohio State', 'Oregon',
    'Penn State', 'Purdue', 'Rutgers', 'UCLA', 'USC', 'Washington', 'Wisconsin',
    # Big 12
    'Baylor', 'BYU', 'Cincinnati', 'Colorado', 'Houston', 'Iowa State',
    'Kansas', 'Kansas State', 'Oklahoma', 'Oklahoma State', 'TCU',
    'Texas Tech', 'UCF', 'Utah', 'West Virginia',
    # ACC
    'Boston College', 'Clemson', 'Duke', 'Florida State', 'Georgia Tech',
    'Louisville', 'Miami', 'NC State', 'North Carolina', 'Notre Dame',
    'Pittsburgh', 'Stanford', 'Syracuse', 'Virginia', 'Virginia Tech', 'Wake Forest',
}

# --- Build 4-year rolling avg recruiting rank per team-year ---
recruiting = pd.read_csv('recruiting_classes.csv')
recruiting = recruiting.sort_values(['team', 'year'])

roster_talent = []
for team, grp in recruiting.groupby('team'):
    grp = grp.set_index('year')
    for year in grp.index:
        window = [y for y in [year-1, year-2, year-3, year-4] if y in grp.index]
        if len(window) == 0:
            continue
        avg_rank   = grp.loc[window, 'rank'].mean()
        avg_points = grp.loc[window, 'points'].mean()
        roster_talent.append({'team': team, 'year': year, 'avg_rank': avg_rank, 'avg_points': avg_points})

talent_df = pd.DataFrame(roster_talent)

# --- Pull games and returning production from cfbd ---
configuration = cfbd.Configuration(access_token=os.getenv('CFBD_API_KEY'))
game_records = []
returning_records = []

with cfbd.ApiClient(configuration) as api_client:
    games_api   = cfbd.GamesApi(api_client)
    players_api = cfbd.PlayersApi(api_client)

    for year in range(2014, 2024):
        # Games
        try:
            games = games_api.get_games(year=int(year), season_type='regular')
            count = 0
            for g in games:
                if g.home_points is None or g.away_points is None:
                    continue
                if g.home_team not in POWER4 or g.away_team not in POWER4:
                    continue
                game_records.append({
                    'year':        year,
                    'home_team':   g.home_team,
                    'away_team':   g.away_team,
                    'home_points': g.home_points,
                    'away_points': g.away_points,
                    'neutral':     g.neutral_site,
                })
                count += 1
            print(f"{year}: {count} Power 4 games")
        except Exception as e:
            print(f"error pulling games {year}: {e}")

        # Returning production
        try:
            ret = players_api.get_returning_production(year=int(year))
            for r in ret:
                if r.team not in POWER4:
                    continue
                returning_records.append({
                    'year':        year,
                    'team':        r.team,
                    'ppa_return':  r.percent_ppa,
                    'usage_return': r.usage,
                })
        except Exception as e:
            print(f"error pulling returning production {year}: {e}")

games_df     = pd.DataFrame(game_records)
returning_df = pd.DataFrame(returning_records)

# --- Join recruiting talent ---
games_df = games_df.merge(
    talent_df.rename(columns={'team': 'home_team', 'avg_rank': 'home_avg_rank', 'avg_points': 'home_avg_points'}),
    on=['home_team', 'year'], how='inner'
).merge(
    talent_df.rename(columns={'team': 'away_team', 'avg_rank': 'away_avg_rank', 'avg_points': 'away_avg_points'}),
    on=['away_team', 'year'], how='inner'
)

# --- Join returning production ---
games_df = games_df.merge(
    returning_df.rename(columns={'team': 'home_team', 'ppa_return': 'home_ppa_return', 'usage_return': 'home_usage_return'}),
    on=['home_team', 'year'], how='left'
).merge(
    returning_df.rename(columns={'team': 'away_team', 'ppa_return': 'away_ppa_return', 'usage_return': 'away_usage_return'}),
    on=['away_team', 'year'], how='left'
)

# --- Features and target ---
games_df['rank_diff']        = games_df['away_avg_rank']    - games_df['home_avg_rank']
games_df['points_diff']      = games_df['home_avg_points']  - games_df['away_avg_points']
games_df['ppa_return_diff']  = games_df['home_ppa_return']  - games_df['away_ppa_return']
games_df['usage_return_diff']= games_df['home_usage_return']- games_df['away_usage_return']
games_df['home_win']         = (games_df['home_points'] > games_df['away_points']).astype(int)
games_df['neutral']          = games_df['neutral'].astype(int)

features_v1 = ['rank_diff', 'points_diff', 'neutral']
features_v2 = ['rank_diff', 'points_diff', 'neutral', 'ppa_return_diff', 'usage_return_diff']

df_model = games_df.dropna(subset=features_v2).copy()

train = df_model[df_model['year'] <= 2020]
test  = df_model[df_model['year'] >  2020]

print(f"\nTrain: {len(train)} games | Test: {len(test)} games")
print(f"Home win rate — train: {train['home_win'].mean():.3f} | test: {test['home_win'].mean():.3f}\n")

def evaluate(name, features, train, test):
    X_tr, y_tr = train[features], train['home_win']
    X_te, y_te = test[features],  test['home_win']
    lr = LogisticRegression()
    lr.fit(X_tr, y_tr)
    preds = lr.predict(X_te)
    proba = lr.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, preds)
    auc = roc_auc_score(y_te, proba)
    print(f"  {name:<35} Accuracy: {acc:.4f}  AUC: {auc:.4f}")
    return lr, proba, y_te, auc

print("Logistic Regression results:")
_, proba_v1, y_te, auc_v1 = evaluate("Recruiting only",          features_v1, train, test)
lr_v2, proba_v2, y_te, auc_v2 = evaluate("Recruiting + Returning Production", features_v2, train, test)

print(f"\nAUC improvement from returning production: +{auc_v2 - auc_v1:.4f}")

print(f"\nFull classification report (Recruiting + Returning):")
print(classification_report(y_te, lr_v2.predict(test[features_v2]), target_names=['Away Win', 'Home Win']))

# Feature coefficients
coef_df = pd.Series(lr_v2.coef_[0], index=features_v2).sort_values(key=abs, ascending=False)
print("Feature coefficients (larger abs = more influential):")
print(coef_df.to_string())

# --- Plots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC comparison
for proba, auc, label in [(proba_v1, auc_v1, 'Recruiting Only'),
                           (proba_v2, auc_v2, 'Recruiting + Returning')]:
    fpr, tpr, _ = roc_curve(y_te, proba)
    axes[0].plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
axes[0].plot([0,1],[0,1],'k--', linewidth=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve Comparison')
axes[0].legend()

# Returning production diff vs win rate (binned)
df_model['ppa_return_bin'] = pd.cut(df_model['ppa_return_diff'], bins=8)
win_by_ret = df_model.groupby('ppa_return_bin')['home_win'].mean()
win_by_ret.plot(kind='bar', ax=axes[1], color='steelblue')
axes[1].axhline(0.5, color='red', linestyle='--', linewidth=1)
axes[1].set_title('Home Win Rate by Returning Production Differential')
axes[1].set_xlabel('Returning PPA Diff (home advantage = positive)')
axes[1].set_ylabel('Home Win Rate')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('game_model_results.png', dpi=150)
print("\nPlots saved to game_model_results.png")
