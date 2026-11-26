
# Feature Dictionary and Data Lineage

This document traces the inputs, transformations, targets, and outputs of the Workforce Pell Opportunity Score. It covers the Colorado outcome model, the separate national completion model, and the final readiness-scoring pipeline.

The project estimates promising certificate programs; its scores do not determine official Workforce Pell eligibility.

## 1. Data Sources

| Source | Data Used | Reference |
|---|---|---|
| College Scorecard — Field of Study | Certificate programs, award counts, earnings, program identifiers | [College Scorecard](https://collegescorecard.ed.gov/data/) |
| College Scorecard — Institution | Enrollment, demographics, institution characteristics, completion rates | [College Scorecard](https://collegescorecard.ed.gov/data/) |
| NCES CIP–SOC Crosswalk | Program-to-occupation mapping | [NCES](https://nces.ed.gov/ipeds/cipcode) |
| HUD ZIP–County Crosswalk | Institution ZIP-to-county mapping | [HUD](https://www.huduser.gov/portal/datasets/usps_crosswalk.html) |
| Census CBSA Delineations | County-to-metropolitan-area mapping | [Census](https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html) |
| Census ACS 2024 5-Year | County population, income, education, and employment | [ACS](https://www.census.gov/programs-surveys/acs/data.html) |
| BLS OEWS May 2025 | Regional occupational employment, wages, and concentration | [BLS](https://www.bls.gov/oes/) |
| Colorado CDHE ROI | Certificate earnings and wage-record coverage | [CDHE](https://cdhe.colorado.gov/return-on-investment-report) |

IPEDS-derived variables enter through College Scorecard rather than separately downloaded IPEDS files. The exact release of the team's original College Scorecard input CSVs remains unconfirmed.

## 2. Data Processing and Lineage

| Stage | Implementation | Output |
|---|---|---|
| National merging | [`merged.py`](../data/national_data/merged.py) | `merged_dataset_v1.csv` |
| National cleaning | [`clean_data.py`](../data/national_data/clean_data.py) | `clean_dataset.csv` |
| Colorado cleaning | [`clean_colorado_data.ipynb`](../data/colorado_data/clean_colorado_data.ipynb) | `year_1_clean.csv` |
| Colorado labeling | [`colorado_label.ipynb`](../data/colorado_data/colorado_label.ipynb) | `colorado_model_table.csv` |
| National model preparation | [`model_setup.py`](../notebooks/model_setup.py) | National feature matrix and target |
| Colorado model preparation | [`model_setup_colorado.py`](../notebooks/model_setup_colorado.py) | Colorado feature matrix and target |
| Readiness scoring | [`readiness_score.ipynb`](../notebooks/readiness_score.ipynb) | `readiness_scores.csv` |

### National Merge Logic

1. Combine three College Scorecard Field of Study files and retain `CREDLEV == 1`.
2. Map each four-digit `CIPCODE` to the most frequent valid SOC code in the CIP–SOC crosswalk.
3. Join institution information using `UNITID`.
4. Select the county with the highest residential ratio for each ZIP code.
5. Join county and CBSA information, then ACS county estimates.
6. Filter OEWS to regional area types 4 and 6 and detailed occupations.
7. Join OEWS using `PRIMARY_SOC_CODE` and `CBSA Title`.
8. Retain unmatched observations and handle missing values during cleaning.

The selected occupation is a crosswalk-based approximation; programs may lead to multiple occupations.

## 3. Model Feature Dictionary

The Colorado model explicitly lists 17 features and the national model lists 22. Both additionally generate institution-control indicators.

**CO:** Colorado outcome model  
**US:** National completion model

| Feature | CO | US | Original Variable / Source | Definition and Units | Transformation |
|---|:---:|:---:|---|---|---|
| `IPEDSCOUNT1_LOG` | Yes | Yes | `IPEDSCOUNT1`, Scorecard Field of Study | Awards to students in year 1 of the pooled debt cohort; count | `np.log1p(IPEDSCOUNT1)` |
| `UGDS_LOG` | Yes | Yes | `UGDS`, Scorecard Institution | Undergraduate certificate/degree-seeking enrollment; count | `np.log1p(UGDS)` |
| `PCTPELL` | Yes | Yes | `PCTPELL`, Scorecard Institution | Undergraduate Pell Grant recipient share; proportion | Numeric conversion |
| `PCTFLOAN` | Yes | Yes | `PCTFLOAN`, Scorecard Institution | Undergraduate federal loan recipient share; proportion | Numeric conversion |
| `UG25ABV` | Yes | Yes | `UG25ABV`, Scorecard Institution | Undergraduate share aged 25 or older; proportion | Numeric conversion |
| `LOCALE_GROUP` | Yes | Yes | `LOCALE`, Scorecard Institution | Institution locale category | `LOCALE // 10` |
| `HBCU` | Yes | Yes | `HBCU`, Scorecard Institution | Historically Black College or University indicator | Numeric conversion |
| `HSI` | Yes | Yes | `HSI`, Scorecard Institution | Hispanic-serving institution indicator | Numeric conversion |
| `CIP_FAMILY` | Yes | Yes | `CIPCODE`, Scorecard Field of Study | Two-digit instructional-program family | `CIPCODE // 100` |
| `B01003_001E_LOG` | Yes | Yes | `B01003_001E`, ACS | Total county population; people | `np.log1p(B01003_001E)` |
| `B19013_001E` | Yes | Yes | `B19013_001E`, ACS | Median household income; 2024 inflation-adjusted dollars | Negative values become missing |
| `S1501_C02_010E` | Yes | Yes | ACS S1501 | Population 25+ with some college, no degree; percent | Negative values become missing |
| `S1501_C02_014E` | Yes | Yes | ACS S1501 | Population 25+ with high school diploma or higher; percent | Negative values become missing |
| `S1501_C02_015E` | Yes | Yes | ACS S1501 | Population 25+ with bachelor's degree or higher; percent | Negative values become missing |
| `S2301_C02_001E` | Yes | Yes | ACS S2301 | Labor force participation rate, population 16+; percent | Negative values become missing |
| `S2301_C04_001E` | Yes | Yes | ACS S2301 | Unemployment rate, population 16+; percent | Negative values become missing |
| `NO_ACS` | Yes | Yes | Derived from `B19013_001E` | Missing median household income indicator | `B19013_001E.isna()` |
| `B20004_003E` | No | Yes | ACS B20004 | Median earnings of high school graduates, including equivalency, population 25+; 2024 inflation-adjusted dollars | Negative values become missing |
| `A_MEDIAN` | No | Yes | `A_MEDIAN`, BLS OEWS | Occupational annual median wage; dollars | Remove commas; numeric conversion |
| `LOC_QUOTIENT` | No | Yes | `LOC_QUOTIENT`, BLS OEWS | Regional occupational concentration relative to national concentration; ratio | Numeric conversion |
| `TOT_EMP_LOG` | No | Yes | `TOT_EMP`, BLS OEWS | Estimated occupational employment; jobs | Remove commas; `np.log1p(TOT_EMP)` |
| `NO_OEWS_WAGE` | No | Yes | Derived from `A_MEDIAN` | Missing occupational median wage indicator | `A_MEDIAN.isna()` |

### Generated Features

Both models generate `CTRL_*` columns from the institution-control variable `CONTROL_inst`:

```python
ctrl = pd.get_dummies(
    df["CONTROL_inst"].astype("string"),
    prefix="CTRL",
    dtype=float
)
```

The number and names of generated columns depend on observed institution-control categories.

Log-transformed variables use `np.log1p(x)`, equivalent to `ln(1 + x)`. The model setup scripts convert feature matrices to floating-point values.

`CIP_FAMILY` and `LOCALE_GROUP` remain numerically represented categories rather than one-hot encoded features.

### ACS Metadata

Definitions were checked against the official 2024 Census API metadata:

- [ACS Detailed Variables](https://api.census.gov/data/2024/acs/acs5/variables.json)
- [ACS Subject Variables](https://api.census.gov/data/2024/acs/acs5/subject/variables.json)

ACS subject-table percentages are on a 0–100 scale, while College Scorecard proportion variables are generally on a 0–1 scale.

## 4. Missing-Value Handling

[`clean_data.py`](../data/national_data/clean_data.py) selects available required columns, converts numeric values using `pd.to_numeric(errors="coerce")`, and replaces negative ACS values with missing values.

| Indicator | Definition | Meaning |
|---|---|---|
| `NO_ACS` | `B19013_001E.isna()` | County median household income unavailable |
| `NO_OEWS_WAGE` | `A_MEDIAN.isna()` | Occupational median wage unavailable |
| `HAS_INSTITUTION` | `STABBR.notna()` | Institution state information available |

These indicators describe selected variables, not complete coverage of their respective datasets.

Missing predictors are retained in the cleaned datasets. The model setup scripts do not independently impute missing feature values; handling depends on the estimator or its preprocessing pipeline.

### Colorado Cleaning

[`clean_colorado_data.ipynb`](../data/colorado_data/clean_colorado_data.ipynb):

- Retains `Certificate(<1 year)` and `Certificate(>1 year)`
- Excludes statewide totals and school-wide program rollups
- Converts earnings and counts to numeric values
- Converts `PCT_WITH_WAGE` from percentage strings to 0–1 proportions
- Creates `COMPLETERS_SUPPRESSED` and `EARNINGS_SUPPRESSED` indicators
- Converts suppressed values to missing rather than zero

The Colorado labeling notebook matches institutions and programs, retains records with usable earnings and an available county earnings baseline, and constructs the outcome label.

## 5. Target Variables

### Colorado Outcome Label

| Variable | Source | Definition |
|---|---|---|
| `EARN_MEDIAN` | CDHE | Median Year 1 earnings; dollars |
| `PCT_WITH_WAGE` | CDHE | Share of completers found in Colorado wage records; proportion |
| `PLACEMENT` | Derived from `PCT_WITH_WAGE` | Placement proxy; proportion |
| `BASELINE` | ACS `B20004_003E` | County high school graduate median earnings; dollars |
| `LABEL_EARNINGS` | Derived | Earnings exceed baseline |
| `LABEL` | Derived | Earnings exceed baseline and placement is at least 50% |

```python
LABEL = (
    (EARN_MEDIAN > BASELINE)
    & (PLACEMENT >= 0.50)
).astype(int)
```

Verified against the committed Colorado modeling table:

- 343 records across 22 institutions
- 100 positive labels
- 0 mismatches with the implemented formula

The placement proxy does not capture all employment, including some out-of-state and self-employment activity.

### National Completion Label

`C150_L4` is the College Scorecard institution-level completion rate at 150% of expected time for first-time, full-time students at less-than-four-year institutions.

```python
y = (df["C150_L4"] >= 0.70).astype(int)
```

Records missing `C150_L4` or `UNITID` are excluded.

Verified:

- 26,857 labeled program records
- 4,178 positive labels

The completion label is institution-level, not program-level. The national model is evaluated separately and does not produce the final readiness score.

## 6. Feature Exclusions and Leakage

The Colorado model excludes `EARN_MEDIAN`, `PLACEMENT`, `BASELINE`, `LABEL`, and `LABEL_EARNINGS` because they directly define or represent its target.

It also excludes `B20004_003E`, the source of the earnings baseline; `DEGREE`, which is not available in the same form for national scoring; and OEWS wage variables reserved for the separate demand component.

The national model excludes `C150_L4`, its target source. It includes `B20004_003E` because that variable does not directly define the national completion label.

Other retained outcome variables are absent from the explicit predictor lists.

Both model setup scripts use feature allowlists and five-fold `GroupKFold` cross-validation grouped by `UNITID`, keeping institutions separate across training and validation folds.

This prevents institution overlap between folds but does not independently establish the absence of every possible source of leakage.

## 7. Identifiers and Supporting Variables

| Variable | Source | Usage |
|---|---|---|
| `UNITID` | College Scorecard | Institution identifier, joins, grouped validation |
| `CIPCODE` | College Scorecard | Program code, joins, subject-family derivation |
| `CIPDESC` | College Scorecard | Program description |
| `CREDLEV` | College Scorecard | Certificate filtering |
| `INSTNM_inst` | College Scorecard | Institution name |
| `STABBR` | College Scorecard | Institution state |
| `COUNTY_FIPS` | HUD geographic mapping | County identifier |
| `CBSA Title` | Census geographic mapping | Regional OEWS matching |
| `PRIMARY_SOC_CODE` | CIP–SOC crosswalk | Selected occupation |
| `EARN_MDN_1YR_CLEAN` | Scorecard `EARN_MDN_1YR` | Numeric earnings for analysis |
| `PLACEMENT_RATE` | Scorecard working/not-working counts | Descriptive employment proxy |
| `DEBT_ALL_STGP_EVAL_MDN` | College Scorecard | Retained debt variable; not a model predictor |
| `ENTITY_NAME` | CDHE | Colorado institution name |
| `DEGREE` | CDHE | Colorado certificate-length category |

The national `PLACEMENT_RATE` is calculated as:

```python
PLACEMENT_RATE = (
    EARN_COUNT_WNE_1YR
    / (EARN_COUNT_WNE_1YR + EARN_COUNT_NWNE_1YR)
)
```

It is distinct from the Colorado `PLACEMENT` variable derived from CDHE wage-record coverage.

## 8. Readiness Scoring

[`readiness_score.ipynb`](../notebooks/readiness_score.ipynb) trains Colorado XGBoost and applies it to national certificate-program records.

| Variable | Meaning |
|---|---|
| `OUTCOME_PROB` | Colorado XGBoost predicted positive-class probability |
| `SUBJECT_COVERED` | Program's subject family has at least five Colorado training records |
| `IN_COLORADO_TRAINING` | School/program key appears in Colorado training data |
| `DEMAND_SCORE` | Average percentile of occupational employment and location quotient, multiplied by 100 |
| `SCHOOL_COMPLETION` | Copy of `C150_L4` for display |
| `COMBINED` | Outcome percentile combined with occupational demand |
| `TIER` | Relative Low, Medium, or High score category |
| `TIER_BASIS` | Outcome only, outcome plus demand, or not scored |

### Outcome Score

```python
OUTCOME_PROB = model.predict_proba(Xnat)[:, 1]
```

Programs whose subject family appears fewer than five times in Colorado training receive no outcome score.

The predicted probabilities have not been verified as calibrated.

### Occupational Demand

```python
DEMAND_SCORE = (
    TOT_EMP.rank(pct=True)
    + LOC_QUOTIENT.rank(pct=True)
) / 2 * 100
```

These percentiles are computed across the national scoring records, not independently within each geographic region.

### Combined Score and Tiers

The outcome probability is converted to a percentile rank. The combined score averages that percentile and `DEMAND_SCORE` when both exist.

If demand is unavailable, the outcome percentile is used alone. If the outcome score is unavailable, the combined score remains missing.

```python
pd.qcut(
    COMBINED,
    3,
    labels=["Low", "Medium", "High"]
)
```

Tiers divide scored observations into relative thirds. They are not official eligibility thresholds.

## 9. Verified Data Quality

The following checks were run against the local model-ready CSV files.

| Check | National | Colorado |
|---|---:|---:|
| Total records | 43,465 | 343 |
| Unique institutions | 4,723 | 22 |
| Missing `UNITID` | 903 | 0 |
| Missing `IPEDSCOUNT1` | 6,139 | 12 |
| Missing `UGDS` | 2,114 | 0 |
| Missing `PCTPELL` | 2,123 | 0 |
| Missing `PCTFLOAN` | 2,123 | 0 |
| Missing `UG25ABV` | 2,142 | 0 |
| Missing `B19013_001E` | 2,395 | 0 |
| Missing `B20004_003E` | 2,395 | 0 |
| Missing `TOT_EMP` | 19,149 | 116 |
| Missing `A_MEDIAN` | 18,843 | 111 |
| Missing `LOC_QUOTIENT` | 19,149 | 116 |
| `NO_ACS` mismatches | 0 | 0 |
| `NO_OEWS_WAGE` mismatches | 0 | 0 |
| Colorado label mismatches | — | 0 |

### Program-Key Integrity

**National dataset**

- 805 extra rows share `(UNITID, CIPCODE)` keys with earlier rows.
- All 805 have missing `UNITID`.
- Among 42,562 records with valid `UNITID`, there are zero repeated `(UNITID, CIPCODE)` keys.
- Missing identifiers prevent reliable differentiation of the affected institution/program records.
- Repeated incomplete keys do not establish that the underlying programs are duplicates.

**Colorado dataset**

- 62 `(UNITID, CIPCODE)` combinations have two observations each.
- Each pair represents different certificate-length categories.
- Earnings differ in all 62 pairs; composite labels differ in 21.
- `(UNITID, CIPCODE, DEGREE)` uniquely identifies all 343 committed Colorado records.

Both certificate-length observations are retained because they represent distinct outcomes.

No records were removed during these verification checks.

## 10. Limitations

- **Geographic generalization:** The Colorado outcome model uses only 22 institutions. National predictions extrapolate beyond the training geography.
- **Outcome availability:** Suppressed or missing earnings prevent some programs from receiving labels.
- **Placement measurement:** Colorado wage-record coverage is not a complete employment measure.
- **Completion measurement:** `C150_L4` measures institution-level completion rather than program-level completion.
- **Program mapping:** The CIP–SOC crosswalk reduces multiple possible occupations to one selected occupation.
- **Geographic matching:** Missing ZIP, county, CBSA, or occupational matches reduce feature coverage.
- **Missingness indicators:** `NO_ACS` and `NO_OEWS_WAGE` represent selected variables rather than complete source coverage.
- **Institution identifiers:** National records with missing `UNITID` cannot be uniquely identified using the school/program key.
- **Certificate granularity:** Colorado records distinguish certificate lengths, while national Scorecard records use a different program representation.
- **Source vintage:** The precise College Scorecard input release remains unconfirmed.
- **Probability calibration:** The outcome probabilities have not been verified as calibrated.
- **Eligibility:** The readiness score does not establish official Workforce Pell eligibility.

## 11. References

### Repository Code

- [National Merge](../data/national_data/merged.py)
- [National Cleaning](../data/national_data/clean_data.py)
- [ACS Retrieval](../data/national_data/pull_acs_data.py)
- [Colorado Cleaning](../data/colorado_data/clean_colorado_data.ipynb)
- [Colorado Labeling](../data/colorado_data/colorado_label.ipynb)
- [National Model Setup](../notebooks/model_setup.py)
- [Colorado Model Setup](../notebooks/model_setup_colorado.py)
- [Readiness Scoring](../notebooks/readiness_score.ipynb)

### Official Data Documentation

- [College Scorecard](https://collegescorecard.ed.gov/data/)
- [NCES CIP](https://nces.ed.gov/ipeds/cipcode)
- [HUD ZIP Crosswalk](https://www.huduser.gov/portal/datasets/usps_crosswalk.html)
- [Census CBSA Delineations](https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html)
- [2024 ACS Detailed Variables](https://api.census.gov/data/2024/acs/acs5/variables.json)
- [2024 ACS Subject Variables](https://api.census.gov/data/2024/acs/acs5/subject/variables.json)
- [BLS OEWS](https://www.bls.gov/oes/)
- [Colorado CDHE ROI](https://cdhe.colorado.gov/return-on-investment-report)
