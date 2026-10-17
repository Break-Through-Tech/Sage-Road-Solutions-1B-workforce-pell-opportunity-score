import requests
import pandas as pd

API_KEY = "bf321550aa3ed9b79098bdcd5e29a9c1b1b61b5b"  # paste your actual key

detail_vars = "B01003_001E,B19013_001E,B20004_003E"
detail_url = f"https://api.census.gov/data/2024/acs/acs5?get=NAME,{detail_vars}&for=county:*&in=state:*&key={API_KEY}"

detail_resp = requests.get(detail_url)
print("Status code:", detail_resp.status_code)
detail_data = detail_resp.json()
detail_df = pd.DataFrame(detail_data[1:], columns=detail_data[0])

subject_vars = "S1501_C02_010E,S1501_C02_014E,S1501_C02_015E,S2301_C02_001E,S2301_C04_001E"
subject_url = f"https://api.census.gov/data/2024/acs/acs5/subject?get=NAME,{subject_vars}&for=county:*&in=state:*&key={API_KEY}"

subject_resp = requests.get(subject_url)
subject_data = subject_resp.json()
subject_df = pd.DataFrame(subject_data[1:], columns=subject_data[0])

detail_df['GEOID'] = detail_df['state'] + detail_df['county']
subject_df['GEOID'] = subject_df['state'] + subject_df['county']

acs = detail_df.merge(subject_df.drop(columns=['NAME', 'state', 'county']), on='GEOID', how='left')

print(acs.shape)
print(acs.head())

acs.to_csv("acs_county_data.csv", index=False)