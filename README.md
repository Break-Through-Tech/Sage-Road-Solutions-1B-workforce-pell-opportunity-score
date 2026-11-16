# Workforce Pell Opportunity Score

An explainable machine learning model and readiness score built to help identify short-term certificate programs that are promising candidates for Workforce Pell, developed for Sage Road Solutions as part of the Break Through Tech AI Studio (Fall 2026).

Public data does not say which programs qualify for Workforce Pell, and the ideal requirements (program length, program-level completion and job placement rates, and state approval) are not publicly available. The closest program-level outcome data we found covers Colorado only: Year 1 earnings and the share of completers found in state wage records. So the model scores how likely a program is to lead to good labor market outcomes (graduates who earn more than a high school baseline and show up working), trained on Colorado and applied to programs in every state. We combine that with local demand for the occupation the program leads to, as a stand-in for Workforce Pell readiness. Treat the score as a way to prioritize where to look, not as a decision about eligibility.

---

### 👥 **Team Members**

| Name | GitHub Handle | Contribution |
|------|---------------|--------------|
| Gabby Bui | [@gabby-bui](https://github.com/gabby-bui) | Census ACS analysis, exploratory data analysis on all data (EDA notebook, Colorado EDA script, figures) |
| Mometa Uddin | [@CosmicJellifish](https://github.com/CosmicJellifish) | Gathered initial datasets, IPEDS data search, placement data search (found the Census PSEO Colorado file), dashboard |
| Sam Ruvalcaba | [@glorpToro](https://github.com/glorpToro) | BLS OEWS analysis, placement data search (found the Census PSEO Colorado file), feature dictionary, dashboard |
| Victoria Johnson | [@victoriamjohnson](https://github.com/victoriamjohnson) | Merged national dataset, composite label, model training and selection, readiness score |

---

## 🎯 **Project Highlights**

- Merged College Scorecard, a program-to-occupation crosswalk, county and metro area files, Census ACS, and BLS wages into one table of certificate programs (about 43,000 programs).
- Built a Colorado label from Colorado Department of Higher Education earnings and placement data: a program is labeled 1 when graduates' Year 1 median earnings beat the county high school baseline and at least 50% of completers appear in Colorado wage records. 343 labeled programs at 22 schools, 29% positive.
- Compared logistic regression, random forest, XGBoost and histogram gradient boosting with cross-validation grouped by school. XGBoost was best on Colorado (ROC-AUC 0.73, F1 0.48).
- Found that program subject drives most of the Colorado result, and that the national completion label mostly reflects school size and type.
- Scored 41,357 programs in 59 states and territories with an outcome part, a local demand part, and a tier, and used SHAP to explain the model.
- Cross-checked our earnings and placement numbers against Census PSEO data.

---

## 👩🏽‍💻 **Setup and Installation**

You need Python 3.10 or newer.

```bash
git clone https://github.com/Break-Through-Tech/Sage-Road-Solutions-1B-workforce-pell-opportunity-score.git
cd Sage-Road-Solutions-1B-workforce-pell-opportunity-score
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Quick start (no large downloads)

The analysis tables are in the repository, so you can run the models and the dashboard right away:

- `data/national_data/clean_dataset.csv` (national model table)
- `data/colorado_data/colorado_model_table.csv` (Colorado model table with the label)
- `data/national_data/readiness_scores.csv` (final scores)

```bash
streamlit run dashboard/app.py
```

Run the model notebooks from the `notebooks/` folder in Jupyter or VS Code.

### Full rebuild from raw data

Some files are too large for GitHub. They are in the team Google Drive folder:
https://drive.google.com/drive/folders/16enmBpnOXgOu0QUwYHx6VoRRiUkFMkRh

| File | What it is | Where to save it |
|------|------------|------------------|
| `merged_dataset_v1.csv` (about 820 MB) | Output of `merged.py`. Download it to skip the merge step. | `data/national_data/` |
| `us labor statistics BLS` (xlsx) | BLS OEWS data, May 2025. Open it, save as CSV, and name the CSV `us_labor_statistics_BLS.csv`. Needed only to rerun `merged.py`. | `data/national_data/` |
| `ROI2026_Earnings_1_5_10_CIP4-suppressed.xlsx` | Original Colorado earnings workbook. The Year 1 sheet exported to CSV is `data/colorado_data/year_1_wage.csv`. | for reference |
| `pseo_co.xlsx` | Census PSEO Colorado earnings and flows (two sheets), used for the cross-check only. | for reference |

Run order for a full rebuild. The scripts in `data/national_data/` must be run from inside that folder, because they read files by name.

1. `data/national_data/pull_acs_data.py` (optional, `acs_county_data.csv` is already in the repository). It needs a free Census API key: `export CENSUS_API_KEY=your_key`, from https://api.census.gov/data/key_signup.html.
2. `data/national_data/merged.py` builds `merged_dataset_v1.csv`.
3. `data/national_data/clean_data.py` builds `clean_dataset.csv`.
4. `data/national_data/cluster_model.py` builds `cluster_profiles.csv` (optional clustering).
5. `data/colorado_data/clean_colorado_data.ipynb` builds `year_1_clean.csv`.
6. `data/colorado_data/colorado_label.ipynb` builds `colorado_model_table.csv`.
7. `notebooks/national_models.ipynb` and `notebooks/colorado_models.ipynb` train and compare the models.
8. `notebooks/model_comparison.ipynb` makes the SHAP charts.
9. `notebooks/readiness_score.ipynb` builds `readiness_scores.csv`.
10. `streamlit run dashboard/app.py`.

If you change `notebooks/model_setup.py` or `notebooks/model_setup_colorado.py`, restart the notebook kernel before rerunning, so the new version is loaded.

### Repository structure

```
dashboard/            Streamlit app (app.py) and its requirements
data/
  national_data/      Scorecard, crosswalk and ACS inputs, merge and clean scripts, readiness scores, EDA figures
  colorado_data/      Colorado earnings files, label notebook, model table, Colorado EDA script and figures
notebooks/            Model notebooks, shared model setup files, results tables, SHAP charts
requirements.txt      Python packages
```

---

## 🏗️ **Project Overview**

This project is part of the Break Through Tech AI Studio. Our host is Sage Road Solutions, and the project objective is an explainable machine learning model, a program-level Workforce Pell Opportunity Score, and an interactive dashboard.

Workforce Pell grants cover short programs, but no public dataset says which programs qualify, and the outcomes states care about most (did graduates get jobs, was the program worth the cost) are not in the usual public files. We used public education, wage and labor market data to estimate which certificate programs resemble programs with good outcomes, and which of those are in places with local demand for the occupation. The goal is to help colleges, workforce agencies and policymakers decide where to look first.

---

## 📊 **Data Exploration**

### Data sources

| Source | What we use | Link |
|--------|-------------|------|
| College Scorecard (Field of Study and Institution files) | Program and school features, program award counts (IPEDS-sourced), national completion rate | https://collegescorecard.ed.gov/data/ |
| NCES CIP to SOC crosswalk | Links program codes (CIP) to occupation codes (SOC) | https://nces.ed.gov/ipeds/cipcode |
| HUD ZIP to county crosswalk | School ZIP code to county | https://www.huduser.gov/portal/datasets/usps_crosswalk.html |
| Census metro area definitions | County to metro area (CBSA) | https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html |
| Census American Community Survey, 2024 5-year | County population, income, education and employment | https://www.census.gov/programs-surveys/acs/data.html |
| BLS Occupational Employment and Wage Statistics, May 2025 | Local employment, wages and job concentration by occupation and metro area | https://www.bls.gov/oes/ |
| Colorado Department of Higher Education, Return on Investment report | Year 1 median earnings and share of completers with wage records, by school and program (the Colorado label) | https://cdhe.colorado.gov/return-on-investment-report |
| Census PSEO, Colorado | Cross-check of earnings and placement | https://lehd.ces.census.gov/data/pseo_experimental.html |

IPEDS data reaches the project through College Scorecard (for example the award counts). We did not use IPEDS files directly.

### How the tables are joined

- Programs link to occupations through the CIP to SOC crosswalk. A program that maps to several occupations uses the most common one.
- Schools link to a county through ZIP code, then to a metro area. ACS is joined by county, and OEWS by occupation and metro area.
- Missing values are kept and flagged, not dropped (for example `NO_ACS`, `NO_OEWS_WAGE`).

### EDA

Exploratory analysis by Gabby Bui: `notebooks/EDA_merged_dataset.ipynb`, `data/colorado_data/colorado_eda.py`, and figures in `data/national_data/figures/` and `data/colorado_data/figures/`.

### Challenges and assumptions

- There is no public program-level completion label, and the only program-level placement data we found is a proxy for Colorado programs (the share of completers found in state wage records). We tested several sources and built the label from Colorado program-level earnings and that proxy.
- Placement is a proxy: the share of completers found in Colorado wage records. Graduates who work out of state, are self-employed, or earn under a full-time minimum wage look unplaced. Census PSEO suggests about 81% of employed Colorado certificate graduates work in Colorado.
- Colorado suppresses groups with fewer than 10 completers with wage data, so 400 of 747 certificate programs have no earnings and are left unlabeled (not set to 0).

---

## 🧠 **Model Development**

- **Colorado label:** 1 if Year 1 median earnings are above the county high school baseline (ACS B20004_003E) and at least 50% of completers have wage records, otherwise 0. Both parts are required. The 50% cutoff is a judgment call.
- **National label:** school completion rate at 70% or higher (College Scorecard C150_L4). It is school-level, so every program at a school shares one value.
- **Models:** logistic regression, random forest, XGBoost and histogram gradient boosting, compared with a dummy baseline. Model settings were chosen by hand, with no tuning search.
- **Validation:** 5-fold cross-validation grouped by school (GroupKFold), so no school is in both training and test.
- **Leakage:** label-building columns and other outcome columns are excluded from the predictors. See `notebooks/model_setup_colorado.py` for the exact list.
- **Readiness score:** the outcome part comes from the Colorado XGBoost model applied to all programs. The demand part is the average percentile of local employment and job concentration for the occupation linked to the program. The overall score and tiers (High, Medium, Low, in thirds) combine the two. Programs without a demand score use the outcome part alone.
- **Clustering:** k-means with 4 clusters (silhouette 0.185) in `data/national_data/cluster_model.py`.

---

## 📈 **Results & Key Findings**

### Colorado model (343 programs, 22 schools)

| Model | F1 | ROC-AUC |
|-------|----|---------|
| Dummy baseline | 0.00 | 0.50 |
| Logistic regression | 0.14 | 0.58 |
| Random forest | 0.24 | 0.73 |
| **XGBoost** | **0.48** | **0.73** |
| Histogram gradient boosting | 0.39 | 0.69 |

XGBoost accuracy 0.75, precision 0.62, recall 0.40. Fold scores vary a lot (ROC-AUC 0.60 to 0.84) because each fold holds out only 4 or 5 schools.

### National model (26,857 programs, 2,789 schools)

XGBoost reaches ROC-AUC 0.95 and F1 0.69, but the national label mostly reflects school size and type, not program quality. Small schools and private schools meet the 70% completion bar far more often than large public ones, which is a fairness concern for any score built on it.

### What drives the scores

- **Program subject is the main driver in Colorado.** Subject alone gets ROC-AUC 0.75, and everything except subject gets 0.52 (close to a coin flip). Subject also explains about 55% of the variation in the outcome score.
- Subjects that push toward a positive label include criminal justice, electrical and power installation, environmental control technologies, and legal support. Education, culinary arts and computer science push the other way.
- SHAP charts are in `notebooks/shap_colorado_bar.png` and `notebooks/shap_national_beeswarm.png`.

### Cross-check with Census PSEO (Colorado)

- Earnings agree with the CDHE numbers (rank correlation 0.70 on matched programs).
- PSEO employment rates, available by school and 2-digit subject, track our placement proxy (rank correlation 0.64) and run higher, because they count work anywhere in the US.
- PSEO has no school-level 4-digit rates we could use for individual programs, so the CDHE placement proxy stays as the placement part of the label.

### Readiness scores

`data/national_data/readiness_scores.csv` has 41,357 programs. 6,949 are not scored (no usable outcome data). Of the rest, 20,245 have both an outcome and a demand score and 14,163 have an outcome score only.

---

## 🚀 **Next Steps**

**Limitations**

- Only 22 Colorado schools, so scores are noisy and the model separates subjects better than schools within a subject.
- The outcome model is trained on Colorado only. Scores for other states are an extrapolation we could not validate.
- The placement part of the label is a proxy and runs low, as described above.
- Program length (8 to 15 weeks for Workforce Pell) is rarely available.
- The probabilities have not been checked for calibration, so treat them as a relative score.

**Next**

- Let dashboard users choose the placement cutoff (50%, 60% or 70%, with 70% matching the Workforce Pell placement standard) by training one model per cutoff, with a warning that results at 70% are noisier.
- Check probability calibration and validate the tiers with expert review.
- Submit the data request to the Colorado Department of Higher Education for program-level starters, completers and placement.
- Add more states' data if program-level outcomes become available.

---

## 📝 **License**

[License to be confirmed with the Challenge Advisor.]

---

## 📄 **References**

See the data sources table above.

---

## 🙏 **Acknowledgements**

Thank you to our Challenge Advisor, Beth Davis of Sage Road Solutions, and our AI Studio Coach, Bhavya Gopal, for their guidance throughout the project.
