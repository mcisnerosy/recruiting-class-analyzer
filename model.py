import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('merged_data.csv')

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

df = df[df['team'].isin(POWER4)].copy()
print(f"Power 4 teams in dataset: {df['team'].nunique()}")

df['games'] = df['wins'] + df['losses']
df['win_pct'] = df['wins'] / df['games']
df = df[df['games'] > 0].copy()
df = df.sort_values(['team', 'year'])

# Lag recruiting points and rank by 1, 2, 3 years per team
for lag in [1, 2, 3]:
    df[f'points_lag{lag}'] = df.groupby('team')['points'].shift(lag)
    df[f'rank_lag{lag}']   = df.groupby('team')['rank'].shift(lag)

# Drop rows missing any lag (teams without 3 prior years of data)
df = df.dropna(subset=[f'points_lag{lag}' for lag in [1, 2, 3]] +
                       [f'rank_lag{lag}'   for lag in [1, 2, 3]])

features = ['points', 'rank',
            'points_lag1', 'rank_lag1',
            'points_lag2', 'rank_lag2',
            'points_lag3', 'rank_lag3']
target = 'win_pct'

train = df[df['year'] <= 2020]
test  = df[df['year'] >  2020]

X_train, y_train = train[features], train[target]
X_test,  y_test  = test[features],  test[target]

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
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mae  = np.mean(np.abs(y_test - preds))
    results[name] = {'model': model, 'preds': preds, 'rmse': rmse, 'r2': r2, 'mae': mae}
    print(f"{name:<22} {rmse:>8.4f} {r2:>8.4f} {mae:>8.4f}")

# Feature importance from Random Forest
rf = results['Random Forest']['model']
importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nRandom Forest feature importances:")
for feat, imp in importance.items():
    print(f"  {feat:<16} {imp:.4f}")

# --- Plots ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Same-year rank vs win_pct
axes[0].scatter(df['rank'], df['win_pct'], alpha=0.3, s=10)
axes[0].set_xlabel('Same-Year Recruiting Rank')
axes[0].set_ylabel('Win %')
axes[0].set_title('Same-Year Rank vs Win %')

# 2. 2-year lagged rank vs win_pct
axes[1].scatter(df['rank_lag2'], df['win_pct'], alpha=0.3, s=10)
axes[1].set_xlabel('Recruiting Rank (2 Years Prior)')
axes[1].set_ylabel('Win %')
axes[1].set_title('Lag-2 Rank vs Win %')

# 3. Best model: actual vs predicted
best_name = min(results, key=lambda k: results[k]['rmse'])
best_preds = results[best_name]['preds']
axes[2].scatter(y_test, best_preds, alpha=0.4, s=10)
axes[2].plot([0, 1], [0, 1], 'r--', linewidth=1)
axes[2].set_xlabel('Actual Win %')
axes[2].set_ylabel('Predicted Win %')
axes[2].set_title(f'{best_name}: Actual vs Predicted')

plt.tight_layout()
plt.savefig('model_results.png', dpi=150)
print(f"\nBest model: {best_name} (RMSE={results[best_name]['rmse']:.4f})")
print("Plots saved to model_results.png")

test = test.copy()
test['predicted_win_pct'] = best_preds
test[['year', 'team', 'rank', 'points', 'wins', 'losses', 'win_pct', 'predicted_win_pct']].to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
