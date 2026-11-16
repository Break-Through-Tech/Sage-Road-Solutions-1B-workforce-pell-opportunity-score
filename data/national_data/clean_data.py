import pandas as pd
import numpy as np

# ============================================================
# STEP 0: Pick the columns we want
# (the full file is 785MB and 3,510 columns, we only need a few dozen)
# ============================================================
id_cols = ['UNITID', 'CIPCODE', 'CIPDESC', 'INSTNM_inst', 'STABBR',
           'CBSA Title', 'COUNTY_FIPS', 'PRIMARY_SOC_CODE']

# Context features: what the program, school, and area look like
school_cols = ['IPEDSCOUNT1', 'UGDS', 'PCTPELL', 'PCTFLOAN', 'UG25ABV',
               'LOCALE', 'HBCU', 'HSI', 'CONTROL_inst']
acs_cols = ['B01003_001E', 'B19013_001E', 'B20004_003E', 'S1501_C02_010E',
            'S1501_C02_014E', 'S1501_C02_015E', 'S2301_C02_001E', 'S2301_C04_001E']
oews_cols = ['TOT_EMP', 'A_MEDIAN', 'LOC_QUOTIENT']

# Outcome columns: used to JUDGE the clusters later, NOT to build them
outcome_cols = ['EARN_MDN_1YR_CLEAN', 'PLACEMENT_RATE', 'C150_L4', 'DEBT_ALL_STGP_EVAL_MDN']

wanted = id_cols + school_cols + acs_cols + oews_cols + outcome_cols

# Only ask for columns that actually exist, and tell us which ones don't
header = pd.read_csv("merged_dataset_v1.csv", nrows=0).columns.tolist()
use = [c for c in wanted if c in header]
print("Columns not found in the merged file:", [c for c in wanted if c not in header])

df = pd.read_csv("merged_dataset_v1.csv", usecols=use, low_memory=False,
                 dtype={'COUNTY_FIPS': str})
print("Loaded:", df.shape)

# ============================================================
# STEP 1: Turn text columns into real numbers
# ('PS' and other junk become blank instead of breaking things)
# ============================================================
numeric_cols = [c for c in school_cols + acs_cols + oews_cols + outcome_cols
                if c in df.columns and c != 'CONTROL_inst']
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# ============================================================
# STEP 2: Fix Census "missing" codes
# The Census API uses huge negative numbers (like -666666666) to mean "no data".
# Income and percentages can't be negative, so any negative value is really missing.
# ============================================================
for c in acs_cols:
    if c in df.columns:
        df[c] = df[c].where(df[c] >= 0)

# ============================================================
# STEP 3: Add flags so missingness itself becomes information
# ============================================================
df['HAS_INSTITUTION'] = df['STABBR'].notna()          # False = the 2,108 unmatched rows
df['NO_OEWS_WAGE'] = df['A_MEDIAN'].isna() if 'A_MEDIAN' in df.columns else True
df['NO_ACS'] = df['B19013_001E'].isna() if 'B19013_001E' in df.columns else True

# ============================================================
# STEP 4: Check the result and save
# ============================================================
print("\nShare of each column that is blank:")
print(df.isna().mean().sort_values(ascending=False).round(3).to_string())

print("\nRows with an institution match:", df['HAS_INSTITUTION'].sum(), "of", len(df))

df.to_csv("clean_dataset.csv", index=False)
print("\nSaved clean_dataset.csv with shape", df.shape)