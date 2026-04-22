# ─────────────────────────────────────────────────────────────────────────────
# model.py
# PURPOSE: Build a season-level model that tries to predict a team's win
# percentage using their recruiting class rankings. This is the first model —
# it works at the team-season level (one row = one team's full season).
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
# Three different model types to compare
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
# Metrics to evaluate how good each model is
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the merged dataset built by wins_and_merge.py
df = pd.read_csv('merged_data.csv')

# Power 4 conferences only — SEC, Big Ten, Big 12, ACC
# Mixing all FBS teams kills the signal because a team ranked 80th in recruiting
# could go 9-3 in the MAC or 4-8 in the SEC. Same rank, totally different wins.
# Filtering to Power 4 makes the comparison fair.
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

# Keep only Power 4 teams
df = df[df['team'].isin(POWER4)].copy()
print(f"Power 4 teams in dataset: {df['team'].nunique()}")

# Calculate total games played and win percentage
# We use win_pct instead of raw wins because seasons have different game counts
# (especially 2020 COVID year where teams played 8-10 games instead of 12-13)
df['games']   = df['wins'] + df['losses']
df['win_pct'] = df['wins'] / df['games']

# Drop any rows where a team played 0 games (cancelled seasons)
df = df[df['games'] > 0].copy()

# Sort by team then year — required for the lag feature calculation below
df = df.sort_values(['team', 'year'])

# Create lag features — the recruiting rank/points from 1, 2, and 3 years ago
# This represents the prior recruiting classes that are now sophomores, juniors, seniors
# groupby('team').shift(1) means "for each team, look at the previous year's value"
for lag in [1, 2, 3]:
    df[f'points_lag{lag}'] = df.groupby('team')['points'].shift(lag)
    df[f'rank_lag{lag}']   = df.groupby('team')['rank'].shift(lag)

# Drop rows where we don't have 3 full years of prior data
# (e.g. a team's 2010 and 2011 rows won't have all lag values)
df = df.dropna(subset=[f'points_lag{lag}' for lag in [1, 2, 3]] +
                       [f'rank_lag{lag}'   for lag in [1, 2, 3]])

# Define which columns are inputs (features) and what we're predicting (target)
features = ['points', 'rank',           # current year's recruiting class
            'points_lag1', 'rank_lag1', # last year's class (now sophomores)
            'points_lag2', 'rank_lag2', # 2 years ago (now juniors)
            'points_lag3', 'rank_lag3'] # 3 years ago (now seniors)
target = 'win_pct'

# Time-based train/test split — train on older data, test on newer data
# This is important: if we split randomly we'd be "cheating" by training on
# future data. Real predictions only use past data.
train = df[df['year'] <= 2020]  # train on 2013-2020
test  = df[df['year'] >  2020]  # test on 2021-2023

X_train, y_train = train[features], train[target]
X_test,  y_test  = test[features],  test[target]

# Three models to compare:
# LinearRegression — basic straight-line fit, simplest possible model
# Ridge — like linear regression but penalizes large coefficients (prevents overfitting)
# RandomForest — builds many decision trees and averages their predictions (captures non-linear patterns)
models = {
    'Linear Regression': LinearRegression(),
    'Ridge':             Ridge(alpha=1.0),
    'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=42),
}

print(f"Train: {len(train)} rows ({train['year'].min()}–{train['year'].max()})")
print(f"Test:  {len(test)} rows ({test['year'].min()}–{test['year'].max()})\n")
print(f"{'Model':<22} {'RMSE':>8} {'R²':>8} {'MAE':>8}")
print("-" * 50)

results = {}
for name, model in models.items():
    # .fit() trains the model on the training data
    model.fit(X_train, y_train)

    # .predict() generates predictions on the unseen test data
    preds = model.predict(X_test)

    # RMSE — Root Mean Squared Error: average prediction error (in win% units)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # R² — how much of the variance in win% the model explains (0 = nothing, 1 = perfect)
    r2   = r2_score(y_test, preds)

    # MAE — Mean Absolute Error: simpler average of how far off predictions are
    mae  = np.mean(np.abs(y_test - preds))

    results[name] = {'model': model, 'preds': preds, 'rmse': rmse, 'r2': r2, 'mae': mae}
    print(f"{name:<22} {rmse:>8.4f} {r2:>8.4f} {mae:>8.4f}")

# Random Forest can tell us which features it used most heavily
# Higher importance = that feature drove more of the predictions
rf = results['Random Forest']['model']
importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nRandom Forest feature importances:")
for feat, imp in importance.items():
    print(f"  {feat:<16} {imp:.4f}")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Does same-year recruiting rank correlate with win%?
axes[0].scatter(df['rank'], df['win_pct'], alpha=0.3, s=10)
axes[0].set_xlabel('Same-Year Recruiting Rank')
axes[0].set_ylabel('Win %')
axes[0].set_title('Same-Year Rank vs Win %')

# Plot 2: Does recruiting rank from 2 years ago correlate with win%?
# (Tests whether juniors — recruited 2 years ago — drive results more)
axes[1].scatter(df['rank_lag2'], df['win_pct'], alpha=0.3, s=10)
axes[1].set_xlabel('Recruiting Rank (2 Years Prior)')
axes[1].set_ylabel('Win %')
axes[1].set_title('Lag-2 Rank vs Win %')

# Plot 3: How close are the best model's predictions to reality?
# Points near the diagonal red line = accurate predictions
best_name  = min(results, key=lambda k: results[k]['rmse'])
best_preds = results[best_name]['preds']
axes[2].scatter(y_test, best_preds, alpha=0.4, s=10)
axes[2].plot([0, 1], [0, 1], 'r--', linewidth=1)  # perfect prediction line
axes[2].set_xlabel('Actual Win %')
axes[2].set_ylabel('Predicted Win %')
axes[2].set_title(f'{best_name}: Actual vs Predicted')

plt.tight_layout()
plt.savefig('model_results.png', dpi=150)
print(f"\nBest model: {best_name} (RMSE={results[best_name]['rmse']:.4f})")
print("Plots saved to model_results.png")

# Save the test set with predictions added, for further inspection
test = test.copy()
test['predicted_win_pct'] = best_preds
test[['year', 'team', 'rank', 'points', 'wins', 'losses', 'win_pct', 'predicted_win_pct']].to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
