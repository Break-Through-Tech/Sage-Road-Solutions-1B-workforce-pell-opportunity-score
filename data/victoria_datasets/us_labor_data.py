import pandas as pd

oews = pd.read_csv("us_labor_statistics_BLS.csv")
print(oews['AREA'].nunique())
print(oews['AREA_TYPE'].unique())
print(oews[oews['AREA'] != 99]['AREA_TITLE'].unique()[:20])

print(oews.groupby('AREA_TYPE')['AREA_TITLE'].apply(lambda x: x.unique()[:3]))

oews_regional = oews[oews['AREA_TYPE'].isin([4, 6])].copy()
print(oews_regional.shape)
print(oews_regional['AREA_TITLE'].nunique())

# Clean the comma-formatted numeric columns
for col in ['TOT_EMP', 'A_MEDIAN']:
    oews_regional[col] = oews_regional[col].astype(str).str.replace(',', '')
    oews_regional[col] = pd.to_numeric(oews_regional[col], errors='coerce')

# Keep only the most granular occupation level, matching individual SOC codes
oews_detailed = oews_regional[oews_regional['O_GROUP'] == 'detailed'].copy()
print(oews_detailed.shape)
print(oews_detailed[['AREA_TITLE', 'OCC_CODE', 'OCC_TITLE', 'TOT_EMP', 'A_MEDIAN']].head())