import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_validate

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(REPO_ROOT / "data" / "victoria_datasets" / "clean_dataset.csv",
                 dtype={'COUNTY_FIPS': str}, low_memory=False)

# Label: completion rate at or above 70% (Beth's threshold)
df = df.dropna(subset=['C150_L4', 'UNITID']).copy()
y = (df['C150_L4'] >= 0.70).astype(int)
groups = df['UNITID']   # keeps each school entirely in train or test

# Features (no outcome columns, no C150_L4)
df['CIP_FAMILY'] = df['CIPCODE'] // 100
df['LOCALE_GROUP'] = df['LOCALE'] // 10
for c in ['IPEDSCOUNT1', 'UGDS', 'B01003_001E', 'TOT_EMP']:
    df[c + '_LOG'] = np.log1p(df[c])

ctrl = pd.get_dummies(df['CONTROL_inst'].astype('string'), prefix='CTRL', dtype=float)
feature_cols = [
    'IPEDSCOUNT1_LOG', 'UGDS_LOG', 'PCTPELL', 'PCTFLOAN', 'UG25ABV',
    'LOCALE_GROUP', 'HBCU', 'HSI', 'CIP_FAMILY',
    'B01003_001E_LOG', 'B19013_001E', 'B20004_003E',
    'S1501_C02_010E', 'S1501_C02_014E', 'S1501_C02_015E',
    'S2301_C02_001E', 'S2301_C04_001E',
    'A_MEDIAN', 'LOC_QUOTIENT', 'TOT_EMP_LOG', 'NO_OEWS_WAGE', 'NO_ACS'
]
X = pd.concat([df[feature_cols], ctrl], axis=1).astype(float)

print("Rows:", len(X), "| Schools:", groups.nunique())
print("Share labeled high completion:", y.mean().round(3))

# Same cross-validation and scoring for every model
cv = GroupKFold(n_splits=5)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

def evaluate(model, name):
    res = cross_validate(model, X, y, cv=cv, groups=groups, scoring=scoring)
    out = {m: res[f'test_{m}'].mean().round(3) for m in scoring}
    print(name, out)
    return out