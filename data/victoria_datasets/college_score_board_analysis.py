import pandas as pd

# ─── LOAD THE TWO SCORECARD FILES ───────────────────────────────
# Field of Study file: one row per institution + CIP code + credential level.
# This is the file our certificate-level analysis is built on.
df1 = pd.read_csv("data/Most-Recent-Cohorts-Field-of-Study.csv", low_memory=False)
print(df1.shape)                      # (rows, columns)
print(df1.columns.tolist())           # full list of column names
print(df1.head())                     # first 5 rows, to sanity-check it loaded right
print(df1.isna().mean().sort_values(ascending=False).head(20))  # top 20 columns by % missing

# Institution file: one row per college. Used for institution-level context
# (size, control type, etc.) rather than program-level outcomes.
df2 = pd.read_csv("data/Most-Recent-Cohorts-Institution.csv", low_memory=False)
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