import pandas as pd

# ============================================================
# STEP 0: Load Data Files
# ============================================================
print("Loading Field of Study files...")
part1 = pd.read_csv("Field-of-Study_part1.csv", low_memory=False)
part2 = pd.read_csv("Field-of-Study_part2.csv", low_memory=False)
part3 = pd.read_csv("Field-of-Study_part3.csv", low_memory=False)
df1 = pd.concat([part1, part2, part3], ignore_index=True)
print(f"Field of Study shape: {df1.shape}")

print("Loading Institution file...")
inst = pd.read_csv("Most-Recent-Cohorts-Institution.csv", low_memory=False)
print(f"Institution shape: {inst.shape}")

# ─── CLEAN THE EARNINGS COLUMN ──────────────────────────────
df1['EARN_MDN_1YR_CLEAN'] = pd.to_numeric(df1['EARN_MDN_1YR'], errors='coerce')

# ─── FOCUS ON CERTIFICATES ONLY (CREDLEV == 1) ──────────────────
cert_only = df1[df1['CREDLEV'] == 1].copy()
print(f"Certificates-only shape: {cert_only.shape}")

cert_only['PLACEMENT_RATE'] = pd.to_numeric(cert_only['EARN_COUNT_WNE_1YR'], errors='coerce') / (
    pd.to_numeric(cert_only['EARN_COUNT_WNE_1YR'], errors='coerce') +
    pd.to_numeric(cert_only['EARN_COUNT_NWNE_1YR'], errors='coerce')
)

# ============================================================
# STEP 1: CIP-to-SOC Crosswalk
# ============================================================
cert_only['CIP_FAMILY'] = cert_only['CIPCODE'] // 100
cert_only['CIP_SERIES'] = cert_only['CIPCODE'] % 100

crosswalk = pd.read_csv("CIP-SOC.csv")
crosswalk['CIP_4DIGIT'] = (
    crosswalk['CIP2020Code'].astype(str).str.split('.').str[0].astype(int) * 100 +
    crosswalk['CIP2020Code'].astype(str).str.split('.').str[1].str[:2].astype(int)
)

crosswalk_clean = crosswalk[crosswalk['SOC2018Code'] != '99-9999'].copy()
crosswalk_collapsed = (
    crosswalk_clean.groupby('CIP_4DIGIT')['SOC2018Code']
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
)
crosswalk_collapsed.columns = ['CIPCODE', 'PRIMARY_SOC_CODE']

merged_with_soc = cert_only.merge(crosswalk_collapsed, on='CIPCODE', how='left')
print(f"After SOC match join: {merged_with_soc.shape}")
print(merged_with_soc['PRIMARY_SOC_CODE'].isna().sum(), "certificate rows with no SOC match")

# ============================================================
# STEP 2: Join Certificates to Institution Data
# ============================================================
print("Joining certificates to institution data...")

# UNITID dtype fix (cert data was float64, institution was int64)
merged_with_soc['UNITID'] = merged_with_soc['UNITID'].astype('Int64')
inst['UNITID'] = inst['UNITID'].astype('Int64')

merged = merged_with_soc.merge(
    inst,
    on='UNITID',
    how='left',
    suffixes=('_program', '_inst')
)
print("After Scorecard + Institution join:", merged.shape)
print(merged['HBCU'].isna().sum(), "rows with no institution match")
# Known gap: 2,108 rows (903 missing UNITID + 1,205 UNITIDs not in institution file,
# 378 unique institutions likely closed or added after this Scorecard snapshot). Documented, not dropped.

# ============================================================
# STEP 3: ZIP-to-County Join
# ============================================================
zip_county = pd.read_csv("Zip-County.csv")

# For ZIPs spanning multiple counties, keep the county with the highest residential ratio
zip_county_primary = (
    zip_county.sort_values('res_ratio', ascending=False)
    .drop_duplicates(subset='zip', keep='first')
)
zip_county_primary = zip_county_primary[['zip', 'geoid']].rename(columns={'geoid': 'COUNTY_FIPS'})
zip_county_primary['zip'] = zip_county_primary['zip'].astype(str).str.zfill(5)

# Clean institution ZIP to 5-digit format
inst['ZIP_5'] = inst['ZIP'].astype(str).str.split('-').str[0].str.zfill(5)

inst_with_county = inst.merge(
    zip_county_primary,
    left_on='ZIP_5',
    right_on='zip',
    how='left'
)
print("After ZIP-to-county join:", inst_with_county.shape)
print(inst_with_county['COUNTY_FIPS'].isna().sum(), "institutions with no county match")
# Known gap: 341 institutions (5.4%), ZIPs not found in HUD crosswalk
# (likely institutional/PO box ZIPs, e.g. Puerto Rico and campus-specific codes). Documented, not dropped.

# ============================================================
# STEP 4: County-to-CBSA (Metro Area) Join
# ============================================================
cbsa = pd.read_excel("list1_2023.xlsx", skiprows=2)

cbsa['COUNTY_FIPS'] = (
    cbsa['FIPS State Code'].astype('Int64').astype(str).str.zfill(2) +
    cbsa['FIPS County Code'].astype('Int64').astype(str).str.zfill(3)
)
cbsa_lookup = cbsa[['COUNTY_FIPS', 'CBSA Code', 'CBSA Title']].drop_duplicates(subset='COUNTY_FIPS')
cbsa_lookup['COUNTY_FIPS'] = cbsa_lookup['COUNTY_FIPS'].astype(str).str.zfill(5)

# Fix institution COUNTY_FIPS dtype (float -> clean string) before this join
inst_with_county['COUNTY_FIPS'] = inst_with_county['COUNTY_FIPS'].astype('Int64').astype(str).str.zfill(5)

inst_with_cbsa = inst_with_county.merge(
    cbsa_lookup,
    on='COUNTY_FIPS',
    how='left'
)
print("After county-to-CBSA join:", inst_with_cbsa.shape)
print(inst_with_cbsa['CBSA Title'].isna().sum(), "institutions with no CBSA match")
# Expected: rural counties with no defined metro/micro area won't match here.

no_cbsa = inst_with_cbsa[inst_with_cbsa['CBSA Title'].isna()]
print(no_cbsa[['INSTNM', 'STABBR', 'COUNTY_FIPS']].head(10))

# ============================================================
# STEP 5: Bring CBSA info into the main merged dataframe
# ============================================================
# inst_with_cbsa has one row per institution; pull just the columns we need
cbsa_cols = inst_with_cbsa[['UNITID', 'COUNTY_FIPS', 'CBSA Code', 'CBSA Title']]

merged_full = merged.merge(
    cbsa_cols,
    on='UNITID',
    how='left',
    suffixes=('', '_cbsa')
)
print("After bringing in CBSA info:", merged_full.shape)

# ============================================================
# STEP 6: Join ACS county data
# ============================================================
acs = pd.read_csv("acs_county_data.csv", dtype={'GEOID': str})
acs['GEOID'] = acs['GEOID'].str.zfill(5)

merged_full = merged_full.merge(
    acs,
    left_on='COUNTY_FIPS',
    right_on='GEOID',
    how='left'
)
print("After ACS join:", merged_full.shape)
print(merged_full['B19013_001E'].isna().sum(), "rows with no ACS match")

# ============================================================
# STEP 6.5: Load and clean OEWS data (needed before Step 7)
# ============================================================
oews = pd.read_csv("us_labor_statistics_BLS.csv", low_memory=False)

# Filter to regional data only (metro + nonmetro areas)
oews_regional = oews[oews['AREA_TYPE'].isin([4, 6])].copy()

# Clean comma-formatted numeric columns
for col in ['TOT_EMP', 'A_MEDIAN']:
    oews_regional[col] = oews_regional[col].astype(str).str.replace(',', '')
    oews_regional[col] = pd.to_numeric(oews_regional[col], errors='coerce')

# Keep only the most granular occupation level
oews_detailed = oews_regional[oews_regional['O_GROUP'] == 'detailed'].copy()
print("OEWS detailed shape:", oews_detailed.shape)

# ============================================================
# STEP 7: Join OEWS regional wage data
# ============================================================
# Join on BOTH SOC code and region together, so each row only pulls in
# wage data for its own institution's specific metro area, not every region nationally.
merged_full = merged_full.merge(
    oews_detailed[['AREA_TITLE', 'OCC_CODE', 'TOT_EMP', 'A_MEDIAN', 'LOC_QUOTIENT']],
    left_on=['PRIMARY_SOC_CODE', 'CBSA Title'],
    right_on=['OCC_CODE', 'AREA_TITLE'],
    how='left'
)
print("After OEWS join:", merged_full.shape)
print(merged_full['A_MEDIAN'].isna().sum(), "rows with no OEWS wage match")

# ============================================================
# FINAL CHECK
# ============================================================
print("\nFinal merged dataset shape:", merged_full.shape)
print(merged_full.isna().mean().sort_values(ascending=False).head(15))

merged_full.to_csv("merged_dataset_v1.csv", index=False)
print("\nSaved to merged_dataset_v1.csv")