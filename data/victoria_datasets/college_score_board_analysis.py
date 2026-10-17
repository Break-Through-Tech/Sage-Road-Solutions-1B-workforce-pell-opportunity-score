import pandas as pd

# ─── LOAD THE TWO SCORECARD FILES ───────────────────────────────
# Field of Study file was split into 3 parts to fit GitHub's size limit,
# so we load each part and stick them back together into one dataframe.
part1 = pd.read_csv("Field-of-Study_part1.csv", low_memory=False)
part2 = pd.read_csv("Field-of-Study_part2.csv", low_memory=False)
part3 = pd.read_csv("Field-of-Study_part3.csv", low_memory=False)
df1 = pd.concat([part1, part2, part3], ignore_index=True)

print(df1.shape)
print(df1.columns.tolist())
print(df1.head())
print(df1.isna().mean().sort_values(ascending=False).head(20))

# Institution file: one row per college.
df2 = pd.read_csv("Most-Recent-Cohorts-Institution.csv", low_memory=False)
print(df2.shape)
print(df2.columns.tolist())
print(df2.head())
print(df2.isna().mean().sort_values(ascending=False).head(20))

# ─── UNDERSTAND CREDLEV (CREDENTIAL LEVEL) ──────────────────────
# CREDLEV is a numeric code for credential type (1 = certificate, 3 = bachelor's, etc).
# We need to know which code means "certificate" since that's our target population.
print(df1['CREDLEV'].value_counts().sort_index())  # row count per credential level
print(df1['CREDDESC'].unique())                     # plain-language labels for each code

# ─── CHECK HOW SCORECARD MARKS SUPPRESSED EARNINGS DATA ─────────
# Scorecard hides earnings data for small groups instead of leaving it blank.
# First guess: maybe it uses the word "PrivacySuppressed".
print(df1['EARN_MDN_1YR'].apply(type).value_counts())        # are values stored as str or float?
print((df1['EARN_MDN_1YR'] == 'PrivacySuppressed').sum())    # count of that exact suppression code

# That search came back 0, so check for the actual code instead ('PS').
print(df1['EARN_MDN_1YR'].apply(type).value_counts())
print((df1['EARN_MDN_1YR'] == 'PS').sum())                   # count of real suppression code
print(df1['EARN_MDN_1YR'].unique()[:20])                      # sample of raw values to confirm

# ─── CLEAN THE EARNINGS COLUMN INTO USABLE NUMBERS ──────────────
# pd.to_numeric with errors='coerce' turns any non-numeric text (like 'PS')
# into NaN, so the column becomes safe to do math on.
df1['EARN_MDN_1YR_CLEAN'] = pd.to_numeric(df1['EARN_MDN_1YR'], errors='coerce')

print(df1['EARN_MDN_1YR_CLEAN'].describe())                      # distribution of real earnings values
print(df1['EARN_MDN_1YR_CLEAN'].isna().sum(), "missing after cleaning")
print((df1['EARN_MDN_1YR'] == 'PS').sum(), "suppressed (PS)")    # how many were suppressed specifically
print(df1['EARN_MDN_1YR'].isna().sum(), "true NaN (blank)")      # how many were genuinely blank to begin with

# ─── FOCUS ON CERTIFICATES ONLY (CREDLEV == 1) ──────────────────
# Our project's unit of analysis is certificate-level programs, so from here on
# we filter to just those rows and re-check missingness within that subset.
cert_only = df1[df1['CREDLEV'] == 1]
print(cert_only['EARN_MDN_1YR_CLEAN'].isna().mean(), "missing rate for certificates only")
print(cert_only['EARN_MDN_1YR_CLEAN'].describe())

# ─── COMPARE MISSINGNESS ACROSS LONGER EARNINGS WINDOWS (ALL ROWS) ──
# Checking whether earnings measured 4 or 5 years after completion
# are suppressed less often than the 1-year figure, across the whole file.
for col in ['EARN_MDN_4YR', 'EARN_MDN_5YR']:
    clean = pd.to_numeric(df1[col], errors='coerce')
    print(col, "missing rate:", clean.isna().mean())

# ─── SAME COMPARISON, BUT RESTRICTED TO CERTIFICATES ────────────
# This is the version that actually matters for our project, since certificates
# get suppressed more heavily than the dataset average.
for col in ['EARN_MDN_1YR', 'EARN_MDN_4YR', 'EARN_MDN_5YR']:
    clean = pd.to_numeric(cert_only[col], errors='coerce')
    print(col, "missing rate for certificates:", clean.isna().mean())

# ─── CHECK IF DEBT DATA IS LESS SUPPRESSED THAN EARNINGS ────────
# Testing whether debt columns could serve as a better label source
# than earnings, for certificates specifically.
for col in ['DEBT_ALL_STGP_EVAL_MDN', 'DEBT_ALL_PP_EVAL_MDN']:
    clean = pd.to_numeric(cert_only[col], errors='coerce')
    print(col, "missing rate for certificates:", clean.isna().mean())

# ─── CHECK COMPLETION COUNTS AS A MORE RELIABLE FALLBACK ────────
# IPEDSCOUNT1/2 are completer counts, not outcomes, but they're far less
# suppressed than earnings or debt. Useful as a feature, and a sanity check
# for whether institution-level aggregation would avoid the suppression problem.
print(cert_only['IPEDSCOUNT1'].isna().mean(), "IPEDSCOUNT1 missing rate")
print(cert_only['IPEDSCOUNT2'].isna().mean(), "IPEDSCOUNT2 missing rate")



# Does Scorecard track "did this person work at all" separately from "how much did they earn"?
# If so, that's a usable placement signal even though earnings is too sparse.

for col in ['EARN_COUNT_WNE_1YR', 'EARN_COUNT_NWNE_1YR']:
    clean = pd.to_numeric(cert_only[col], errors='coerce')
    print(col, "missing rate for certificates:", clean.isna().mean())

# If these are well-populated, build a placement rate:
# placement rate = WNE / (WNE + NWNE)
cert_only['PLACEMENT_RATE'] = pd.to_numeric(cert_only['EARN_COUNT_WNE_1YR'], errors='coerce') / (
    pd.to_numeric(cert_only['EARN_COUNT_WNE_1YR'], errors='coerce') +
    pd.to_numeric(cert_only['EARN_COUNT_NWNE_1YR'], errors='coerce')
)
print(cert_only['PLACEMENT_RATE'].isna().mean(), "missing rate for placement rate")
print(cert_only['PLACEMENT_RATE'].describe())




print("NEW THINGS THAT WE NEED")
print(cert_only['CIPCODE'].head(10))
print(cert_only['CIPCODE'].dtype)

print("NEW THINGS THAT WE NEED")
# Confirm the pattern on your Scorecard side
cert_only['CIP_FAMILY'] = cert_only['CIPCODE'] // 100
cert_only['CIP_SERIES'] = cert_only['CIPCODE'] % 100
print(cert_only[['CIPCODE', 'CIP_FAMILY', 'CIP_SERIES']].head(10))

# On the crosswalk side, truncate the 6-digit code down to the 4-digit series level
crosswalk = pd.read_csv("CIP-SOC.csv")
crosswalk['CIP_4DIGIT'] = crosswalk['CIP2020Code'].astype(str).str.split('.').str[0].astype(int) * 100 + \
                           crosswalk['CIP2020Code'].astype(str).str.split('.').str[1].str[:2].astype(int)

print(crosswalk[['CIP2020Code', 'CIP_4DIGIT']].head(10))

print("NEWNEWNEW")
# Remove rows with no real occupation match
crosswalk_clean = crosswalk[crosswalk['SOC2018Code'] != '99-9999'].copy()

# Collapse the crosswalk to one row per 4-digit CIP code,
# picking the SOC code that appears most often as a reasonable default.
# (This is the team decision point — flag this choice when you document it.)
crosswalk_collapsed = (
    crosswalk_clean.groupby('CIP_4DIGIT')['SOC2018Code']
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
)
crosswalk_collapsed.columns = ['CIPCODE', 'PRIMARY_SOC_CODE']

print(crosswalk_collapsed.shape)
print(crosswalk_collapsed.head())

# Join to your certificate data
merged_with_soc = cert_only.merge(crosswalk_collapsed, on='CIPCODE', how='left')
print(merged_with_soc.shape)
print(merged_with_soc['PRIMARY_SOC_CODE'].isna().sum(), "certificate rows with no SOC match")

print("LOTS OF NEW THIGNS")
# ============================================================
# STEP 1: Join Scorecard certificates to Institution data
# ============================================================
merged = cert_only.merge(
    inst,
    on='UNITID',
    how='left',
    suffixes=('_program', '_inst')
)
print("After Scorecard + Institution join:", merged.shape)
print(merged['UNITID'].isna().sum(), "rows with no institution match")

# ============================================================
# STEP 2a: Check the ZIP-county crosswalk file structure first
# ============================================================
zip_county = pd.read_csv("path/to/your_zip_county_file.csv")  # update this path
print(zip_county.columns.tolist())
print(zip_county.head())