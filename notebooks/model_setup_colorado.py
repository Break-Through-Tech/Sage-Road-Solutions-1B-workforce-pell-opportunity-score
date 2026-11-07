import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_validate

REPO_ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(REPO_ROOT / "data" / "colorado_data" / "colorado_model_table.csv",
                 dtype={"COUNTY_FIPS": str}, low_memory=False)

# Label: Year 1 median earnings beat the county high school baseline
y = df["LABEL"].astype(int)
groups = df["UNITID"]   # keeps each school entirely in train or test

df["CIP_FAMILY"] = df["CIPCODE"] // 100
df["LOCALE_GROUP"] = df["LOCALE"] // 10
for c in ["IPEDSCOUNT1", "UGDS", "B01003_001E"]:
    df[c + "_LOG"] = np.log1p(df[c])

ctrl = pd.get_dummies(df["CONTROL_inst"].astype("string"), prefix="CTRL", dtype=float)

# Left out on purpose: B20004_003E (it defines the label), the earnings and placement
# columns (outcomes), the OEWS wage columns (kept separate for the demand score),
# and DEGREE (Colorado only, so it can't be used to score other states).
feature_cols = [
    "IPEDSCOUNT1_LOG", "UGDS_LOG", "PCTPELL", "PCTFLOAN", "UG25ABV",
    "LOCALE_GROUP", "HBCU", "HSI", "CIP_FAMILY",
    "B01003_001E_LOG", "B19013_001E",
    "S1501_C02_010E", "S1501_C02_014E", "S1501_C02_015E",
    "S2301_C02_001E", "S2301_C04_001E", "NO_ACS",
]
X = pd.concat([df[feature_cols], ctrl], axis=1).astype(float)

print("Rows:", len(X), "| Schools:", groups.nunique())
print("Share labeled 1:", y.mean().round(3))

cv = GroupKFold(n_splits=5)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

def evaluate(model, name):
    res = cross_validate(model, X, y, cv=cv, groups=groups, scoring=scoring)
    out = {m: res[f"test_{m}"].mean().round(3) for m in scoring}
    print(name, out)
    return out