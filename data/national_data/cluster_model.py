import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 30)

# ============================================================
# STEP 0: Load the cleaned data
# ============================================================
df = pd.read_csv("clean_dataset.csv", dtype={'COUNTY_FIPS': str}, low_memory=False)
print("Loaded:", df.shape)

# Only cluster rows that matched an institution (the other 2,108 have no school features)
model_df = df[df['HAS_INSTITUTION']].copy()
print("Rows used for clustering:", model_df.shape[0])

# ============================================================
# STEP 1: Build the features (context only, NO outcome columns)
# ============================================================
# Log-transform columns with huge ranges so a few giant values don't dominate
for c in ['IPEDSCOUNT1', 'UGDS', 'B01003_001E', 'TOT_EMP']:
    model_df[c + '_LOG'] = np.log1p(model_df[c])

# LOCALE codes: 11-13 city, 21-23 suburb, 31-33 town, 41-43 rural -> 1 to 4
model_df['LOCALE_GROUP'] = model_df['LOCALE'] // 10

# Public / private nonprofit / for-profit as separate yes/no columns
control_dummies = pd.get_dummies(model_df['CONTROL_inst'].astype('string'),
                                 prefix='CTRL', dtype=float)

feature_cols = [
    'IPEDSCOUNT1_LOG', 'UGDS_LOG', 'PCTPELL', 'PCTFLOAN', 'UG25ABV',
    'LOCALE_GROUP', 'HBCU', 'HSI',
    'B01003_001E_LOG', 'B19013_001E', 'B20004_003E',
    'S1501_C02_010E', 'S1501_C02_014E', 'S1501_C02_015E',
    'S2301_C02_001E', 'S2301_C04_001E',
    'A_MEDIAN', 'LOC_QUOTIENT', 'TOT_EMP_LOG', 'NO_OEWS_WAGE'
]
X = pd.concat([model_df[feature_cols], control_dummies], axis=1)
X['NO_OEWS_WAGE'] = X['NO_OEWS_WAGE'].astype(float)

# ============================================================
# STEP 2: Fill blanks with the median, then put everything on the same scale
# ============================================================
X_filled = SimpleImputer(strategy='median').fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_filled)

# ============================================================
# STEP 3: Try a few cluster counts and keep the best one
# ============================================================
best_k, best_score = None, -1
for k in range(4, 9):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
    print(f"k={k}: silhouette={score:.3f}")
    if score > best_score:
        best_k, best_score = k, score
print(f"Using k={best_k}")

model_df['CLUSTER'] = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X_scaled)

# ============================================================
# STEP 4: Describe each cluster, using outcomes the model never saw
# ============================================================
profile = model_df.groupby('CLUSTER').agg(
    programs=('CIPCODE', 'size'),
    avg_completion_rate=('C150_L4', 'mean'),
    share_at_70=('C150_L4', lambda s: (s.dropna() >= 0.70).mean()),
    completion_coverage=('C150_L4', lambda s: s.notna().mean()),
    avg_placement=('PLACEMENT_RATE', 'mean'),
    avg_earnings_1yr=('EARN_MDN_1YR_CLEAN', 'mean'),
    median_county_income=('B19013_001E', 'median'),
    avg_local_wage=('A_MEDIAN', 'mean'),
    avg_pct_pell=('PCTPELL', 'mean'),
    median_enrollment=('UGDS', 'median'),
).round(3)

# ============================================================
# STEP 5: Opportunity Score = how well the cluster's schools complete students
# ============================================================
profile['TIER'] = pd.qcut(profile['avg_completion_rate'].rank(method='first'),
                          3, labels=['Low', 'Medium', 'High'])
model_df['OPPORTUNITY_SCORE'] = (model_df['CLUSTER'].map(profile['avg_completion_rate']) * 100).round(1)
model_df['OPPORTUNITY_TIER'] = model_df['CLUSTER'].map(profile['TIER']).astype(str)

print("\nCluster profiles:")
print(profile.to_string())

# ============================================================
# STEP 6: Save everything
# ============================================================
df = df.merge(model_df[['CLUSTER', 'OPPORTUNITY_SCORE', 'OPPORTUNITY_TIER']],
              left_index=True, right_index=True, how='left')
df.to_csv("clustered_dataset.csv", index=False)
profile.to_csv("cluster_profiles.csv")
print("\nSaved clustered_dataset.csv and cluster_profiles.csv")