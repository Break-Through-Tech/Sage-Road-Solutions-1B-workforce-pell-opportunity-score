from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Workforce Pell Readiness Explorer", layout="wide")

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "victoria_datasets" / "readiness_scores.csv"
NOTEBOOKS = REPO / "notebooks"

CIP_FAMILIES = {
    1: "Agricultural and Veterinary Science", 3: "Natural Resources", 4: "Architecture",
    5: "Area and Cultural Studies", 9: "Communication and Journalism",
    10: "Communications Technologies", 11: "Computer and Information Sciences",
    12: "Culinary and Personal Services", 13: "Education", 14: "Engineering",
    15: "Engineering Technologies", 16: "Foreign Languages", 19: "Family and Consumer Sciences",
    22: "Legal Studies", 23: "English", 24: "Liberal Arts and General Studies",
    25: "Library Science", 26: "Biological Sciences", 27: "Mathematics and Statistics",
    29: "Military Technologies", 30: "Interdisciplinary Studies", 31: "Parks, Recreation, and Fitness",
    38: "Philosophy and Religion", 39: "Theology", 40: "Physical Sciences",
    41: "Science Technologies", 42: "Psychology", 43: "Protective Services and Law Enforcement",
    44: "Public Administration and Social Service", 45: "Social Sciences",
    46: "Construction Trades", 47: "Mechanic and Repair Technologies", 48: "Precision Production",
    49: "Transportation and Materials Moving", 50: "Visual and Performing Arts",
    51: "Health Professions", 52: "Business and Management", 54: "History",
}

SORT_OPTIONS = {
    "Overall score": "COMBINED",
    "Score within subject": "SUBJECT_PERCENTILE",
    "Outcome score": "OUTCOME",
    "Demand score": "DEMAND_SCORE",
    "School completion rate": "COMPLETION_PCT",
}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA, low_memory=False)
    df["SUBJECT"] = df["CIP_FAMILY"].map(CIP_FAMILIES).fillna("Other")
    df["SUBJECT_LABEL"] = df["CIP_FAMILY"].astype(int).astype(str).str.zfill(2) + " " + df["SUBJECT"]
    df["OUTCOME"] = df["OUTCOME_PROB"] * 100
    df["COMPLETION_PCT"] = df["SCHOOL_COMPLETION"] * 100
    df["SUBJECT_PERCENTILE"] = df.groupby("CIP_FAMILY")["COMBINED"].rank(pct=True) * 100
    df["SCORED"] = df["TIER"] != "Not scored"
    return df


if not DATA.exists():
    st.error(f"Could not find the scores file at {DATA}. Run the readiness score notebook first.")
    st.stop()

df = load_data()

st.title("Workforce Pell Readiness Explorer")
st.caption(
    "A readiness score for certificate programs, built from public data. "
    "It is a planning tool, not a Workforce Pell eligibility determination."
)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
view = st.sidebar.radio(
    "Programs to include",
    ["Both score parts (recommended)", "All scored programs"],
    help="Programs with both an outcome score and a demand score are ranked on the same recipe. "
         "The others are ranked on the outcome score alone, so they are not directly comparable.",
)
states = st.sidebar.multiselect("State", sorted(df["STABBR"].dropna().unique()))
subjects = st.sidebar.multiselect("Subject", sorted(df["SUBJECT_LABEL"].unique()))
tiers = st.sidebar.multiselect("Tier", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
school_text = st.sidebar.text_input("School name contains")
sort_label = st.sidebar.selectbox("Rank by", list(SORT_OPTIONS))
top_n = st.sidebar.slider("Rows to show", 25, 500, 100, step=25)

f = df[df["SCORED"]]
if view.startswith("Both"):
    f = f[f["TIER_BASIS"] == "outcome + demand"]
if states:
    f = f[f["STABBR"].isin(states)]
if subjects:
    f = f[f["SUBJECT_LABEL"].isin(subjects)]
f = f[f["TIER"].isin(tiers)]
if school_text:
    f = f[f["INSTNM_inst"].str.contains(school_text, case=False, na=False, regex=False)]
f = f.sort_values(SORT_OPTIONS[sort_label], ascending=False, na_position="last")

if view.startswith("All"):
    st.warning(
        "This view mixes two recipes: programs with both parts and programs scored on outcome alone. "
        "Check the 'Based on' column before comparing programs."
    )

(tab_table, tab_subject, tab_state, tab_overview, tab_models,
 tab_fair, tab_about) = st.tabs(
    ["Ranked programs", "By subject", "By state", "Project overview",
     "Models and explainability", "Fairness and data limits", "About this score"]
)

# ---------- Ranked programs ----------
with tab_table:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Programs shown", f"{len(f):,}")
    c2.metric("Share rated High", f"{(f['TIER'] == 'High').mean():.0%}" if len(f) else "n/a")
    c3.metric("Median outcome score", f"{f['OUTCOME'].median():.0f}" if f["OUTCOME"].notna().any() else "n/a")
    c4.metric("Median demand score", f"{f['DEMAND_SCORE'].median():.0f}" if f["DEMAND_SCORE"].notna().any() else "n/a")

    table = pd.DataFrame({
        "School": f["INSTNM_inst"],
        "State": f["STABBR"],
        "Metro area": f["CBSA Title"],
        "Program": f["CIPDESC"],
        "Subject": f["SUBJECT"],
        "Tier": f["TIER"],
        "Overall": f["COMBINED"],
        "Outcome": f["OUTCOME"],
        "Demand": f["DEMAND_SCORE"],
        "Within subject": f["SUBJECT_PERCENTILE"],
        "School completion %": f["COMPLETION_PCT"],
        "Based on": f["TIER_BASIS"],
        "In Colorado training data": f["IN_COLORADO_TRAINING"],
    })

    bar = lambda label: st.column_config.ProgressColumn(label, min_value=0, max_value=100, format="%.0f")
    st.dataframe(
        table.head(top_n),
        hide_index=True,
        column_config={
            "Overall": bar("Overall"),
            "Outcome": bar("Outcome"),
            "Demand": bar("Demand"),
            "Within subject": bar("Within subject"),
            "School completion %": st.column_config.NumberColumn("School completion %", format="%.0f"),
        },
    )
    st.caption(
        "Demand is blank where no local wage data matched. School completion is the whole school's rate, "
        "shown for context and not part of the score. Programs marked True in the last column were used to "
        "train the outcome model, so their outcome scores look better than they would for unseen programs."
    )
    st.download_button(
        "Download these results as CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="readiness_scores_filtered.csv",
        mime="text/csv",
    )

# ---------- By subject ----------
with tab_subject:
    st.subheader("Average outcome score by subject")
    st.write(
        "Subject explains about 61% of the variation in the outcome score, so comparing programs "
        "within the same subject is usually fairer than comparing across subjects."
    )
    s = (df[df["OUTCOME"].notna()]
         .groupby("SUBJECT_LABEL")
         .agg(programs=("OUTCOME", "size"), avg_outcome=("OUTCOME", "mean"))
         .reset_index())
    s = s[s["programs"] >= 50].sort_values("avg_outcome")
    if len(s):
        fig = px.bar(s, x="avg_outcome", y="SUBJECT_LABEL", orientation="h", hover_data=["programs"],
                     labels={"avg_outcome": "Average outcome score (0 to 100)", "SUBJECT_LABEL": ""})
        fig.update_layout(height=max(400, 28 * len(s)))
        st.plotly_chart(fig)
        st.caption("Subjects with at least 50 scored programs.")

# ---------- By state ----------
with tab_state:
    st.subheader("Median overall score by state")
    both = df[(df["TIER_BASIS"] == "outcome + demand") & df["STABBR"].notna()]
    st_df = both.groupby("STABBR").agg(programs=("COMBINED", "size"), median_score=("COMBINED", "median")).reset_index()
    st_df = st_df[st_df["programs"] >= 20]
    if len(st_df):
        fig = px.choropleth(st_df, locations="STABBR", locationmode="USA-states", scope="usa",
                            color="median_score", hover_data=["programs"],
                            labels={"median_score": "Median overall score"})
        st.plotly_chart(fig)
        st.caption("Programs with both score parts, in states with at least 20 such programs. "
                   "State averages mostly reflect which subjects each state's programs teach.")


# ---------- Helpers for the project tabs ----------
def read_results(name):
    path = NOTEBOOKS / name
    if not path.exists():
        return None
    res = pd.read_csv(path, index_col=0)
    res.insert(0, "Model", res.index)
    return res.reset_index(drop=True)


def show_image(name, caption):
    path = NOTEBOOKS / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Image not found: notebooks/{name}")


def model_chart(res, title):
    metrics = [m for m in ["f1", "roc_auc"] if m in res.columns]
    long = res.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")
    long["Metric"] = long["Metric"].map({"f1": "F1", "roc_auc": "ROC-AUC"})
    fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1], title=title)
    st.plotly_chart(fig)


def simple_bar(labels, values, title, x_title, horizontal=False, fmt=None):
    d = pd.DataFrame({"label": labels, "value": values})
    if horizontal:
        fig = px.bar(d, x="value", y="label", orientation="h", title=title,
                     labels={"value": x_title, "label": ""}, text="value")
        fig.update_yaxes(autorange="reversed")
    else:
        fig = px.bar(d, x="label", y="value", title=title,
                     labels={"value": x_title, "label": ""}, text="value")
    if fmt:
        fig.update_traces(texttemplate=fmt)
    st.plotly_chart(fig)


CHECKLIST = [
    ("An explainable ML model, comparing Random Forest, XGBoost, Gradient Boosting, and Logistic Regression with cross-validation",
     "All of them, on two labels (national school completion, Colorado earnings), cross-validated by school",
     "Done", "Models and explainability tab; notebooks 06 and 07"),
    ("Accuracy, precision, recall, F1, ROC-AUC", "In the results tables", "Done",
     "Models and explainability tab"),
    ("SHAP explainability", "National and Colorado", "Done",
     "Models and explainability tab; notebook 05"),
    ("An interpretable Opportunity Score for programs",
     "readiness_scores.csv: 41,357 programs with an outcome part, a demand part, and a tier",
     "Done, with caveats", "Ranked programs tab"),
    ("Regional workforce demand", "The demand score from OEWS (blank for about 41% of programs)",
     "Partly", "Ranked programs tab, Demand column"),
    ("An interactive dashboard", "This Streamlit app", "In progress", "This app"),
    ("Data source quality and selection", "Documented gaps and a handoff doc", "Done",
     "Fairness and data limits tab"),
    ("A reproducible open-source workflow",
     "The pipeline and notebooks are in the repo, but there is no README yet", "Partly",
     "GitHub repository"),
    ("Actionable recommendations", "A draft is below, for team and advisor review", "Draft",
     "Project overview tab"),
    ("Validation against Workforce Pell criteria and expert review", "Needs the challenge advisor",
     "Not yet", ""),
    ("Feature dictionary, technical documentation, final presentation", "Not done", "Not yet", ""),
    ("Fairness across institution types (a stretch goal)",
     "A start: school size and type strongly shape the national results", "Partly",
     "Fairness and data limits tab"),
]

# ---------- Project overview ----------
with tab_overview:
    st.subheader("What we built")
    st.write(
        "The Workforce Pell Opportunity Score project asked for an explainable model, a score, and a dashboard. "
        "Public data cannot say which programs qualify for Workforce Pell, so we built a readiness score instead: "
        "it estimates how closely a program resembles programs with strong earnings outcomes, alongside local job demand."
    )
    st.subheader("Checklist against the project overview")
    st.dataframe(
        pd.DataFrame(CHECKLIST, columns=["The overview asks for", "What we have", "Status", "Where to see it"]),
        hide_index=True,
    )

    st.subheader("From certificate programs to scored programs")
    funnel = pd.DataFrame({
        "Stage": ["All certificate programs", "With school information", "With an outcome score",
                  "With a demand score", "With both scores"],
        "Programs": [43465, len(df), int(df["OUTCOME"].notna().sum()), int(df["DEMAND_SCORE"].notna().sum()),
                     int((df["OUTCOME"].notna() & df["DEMAND_SCORE"].notna()).sum())],
    })
    fig = px.bar(funnel, x="Programs", y="Stage", orientation="h", text="Programs")
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(texttemplate="%{text:,}")
    st.plotly_chart(fig)
    st.caption("Programs drop out when school, subject, or local wage information is missing. "
               "The total comes from the merged dataset, and the rest from the scores file.")

    st.subheader("How the pieces fit together")
    st.markdown(
        "1. **Merge** College Scorecard, a program-to-occupation crosswalk, county and metro area files, "
        "Census ACS, and BLS wages into one table of certificate programs.\n"
        "2. **Build two labels:** school completion at 70% or higher (national), and Year 1 earnings above the "
        "high school baseline (Colorado).\n"
        "3. **Train and compare models** with cross-validation that keeps each school in either training or testing.\n"
        "4. **Explain them** with SHAP.\n"
        "5. **Score every program** with an outcome part and a demand part, and show school completion alongside.\n"
        "6. **Explore** the results in this dashboard."
    )

    st.subheader("Draft recommendations")
    st.caption("Drafts for the team and the challenge advisor to review.")
    st.markdown(
        "- **Publish program-level completion and placement.** This was the biggest gap we hit. Any state that "
        "releases starting cohorts and completers by program would let a score like this get much closer to the "
        "actual Workforce Pell tests.\n"
        "- **Compare programs within the same subject and school type.** Subject and school size explain a large "
        "share of our scores, so national rankings can be misleading.\n"
        "- **Use the score to shortlist, then verify locally.** Check program length, completion, placement, and "
        "state approval before making funding decisions.\n"
        "- **Extend the framework state by state.** Colorado's published program earnings made the outcome model "
        "possible, and other states that publish similar data could be added the same way."
    )

# ---------- Models and explainability ----------
with tab_models:
    st.subheader("Two labels, two sets of models")
    st.write(
        "We trained the same five models on two different labels. The results use different labels and "
        "sample sizes, so the two sets of numbers should not be compared directly."
    )

    st.markdown("### National model: school completion at 70% or higher")
    st.write("26,857 programs at 2,789 schools. About 16% of programs are labeled high completion.")
    nat_res = read_results("results_national.csv")
    if nat_res is not None:
        model_chart(nat_res, "National models: F1 and ROC-AUC (5-fold cross-validation grouped by school)")
        cols = [c for c in ["Model", "accuracy", "precision", "recall", "f1", "roc_auc"] if c in nat_res.columns]
        st.dataframe(nat_res[cols].round(3), hide_index=True)
    else:
        st.info("notebooks/results_national.csv not found.")
    st.caption("The Dummy model always predicts the most common answer, so it is the floor every model must beat.")

    st.markdown("### Colorado model: Year 1 earnings beat the high school baseline")
    st.write("343 programs at 22 Colorado schools. About 42% of programs are labeled 1.")
    co_res = read_results("results_colorado.csv")
    if co_res is not None:
        model_chart(co_res, "Colorado models: F1 and ROC-AUC (5-fold cross-validation grouped by school)")
        cols = [c for c in ["Model", "accuracy", "precision", "recall", "f1", "roc_auc"] if c in co_res.columns]
        st.dataframe(co_res[cols].round(3), hide_index=True)
    else:
        st.info("notebooks/results_colorado.csv not found.")
    st.caption("Each fold holds out only 4 or 5 schools, so scores are noisy. Treat models within about 0.05 of each other as tied.")

    folds = pd.DataFrame({
        "Fold": [1, 2, 3, 4, 5],
        "ROC-AUC": [0.77, 0.78, 0.75, 0.85, 0.70],
        "F1": [0.66, 0.72, 0.65, 0.69, 0.55],
    }).melt(id_vars="Fold", var_name="Metric", value_name="Score")
    fig = px.bar(folds, x="Fold", y="Score", color="Metric", barmode="group", range_y=[0, 1],
                 title="Colorado XGBoost: score by cross-validation fold")
    st.plotly_chart(fig)

    st.markdown("### What drives the scores")
    left, right = st.columns(2)
    with left:
        show_image("shap_national_beeswarm.png",
                   "National model (SHAP): school size is by far the strongest driver. Large schools push the score down.")
    with right:
        show_image("shap_colorado_bar.png",
                   "Colorado model (SHAP): program subject is by far the strongest driver.")
    st.caption("SHAP shows how much each feature pushed a prediction up or down. It shows what the model relies on, "
               "not what causes outcomes.")

    a, b = st.columns(2)
    with a:
        simple_bar(["Subject only", "All features", "Everything except subject"], [0.793, 0.768, 0.563],
                   "Colorado: ROC-AUC with different inputs", "ROC-AUC", fmt="%{text:.2f}")
    with b:
        simple_bar(["School size only", "All features", "Everything except size"], [0.882, 0.947, 0.904],
                   "National: ROC-AUC with different inputs", "ROC-AUC", fmt="%{text:.2f}")
    st.caption("In Colorado, subject alone predicts about as well as the full model. Nationally, school size alone gets "
               "most of the way, but the model still scores well without it, because other columns carry similar "
               "information about school type.")

    co_subj = {11: (16, 0.94), 43: (23, 0.87), 15: (21, 0.76), 46: (20, 0.70), 52: (33, 0.64), 48: (21, 0.38),
               51: (110, 0.26), 47: (24, 0.21), 13: (17, 0.06), 12: (19, 0.00)}
    d = pd.DataFrame({
        "Subject": [CIP_FAMILIES[k] + f" ({v[0]} programs)" for k, v in co_subj.items()],
        "Share beating baseline": [v[1] * 100 for v in co_subj.values()],
    }).sort_values("Share beating baseline")
    fig = px.bar(d, x="Share beating baseline", y="Subject", orientation="h",
                 title="Colorado: share of certificate programs whose Year 1 earnings beat the high school baseline",
                 labels={"Share beating baseline": "Percent of programs", "Subject": ""})
    st.plotly_chart(fig)
    st.caption("Subjects with at least 15 Colorado programs.")

# ---------- Fairness and data limits ----------
with tab_fair:
    st.subheader("How the national label differs by school size and type")
    st.write(
        "The national label (school completion at 70% or higher) is met far more often at small schools and at "
        "private schools. A score built on it would tend to favor those schools, which matters for fairness "
        "across institution types."
    )
    a, b = st.columns(2)
    with a:
        simple_bar(["Under 100", "100 to 500", "500 to 2,000", "2,000 to 10,000", "Over 10,000"],
                   [65, 51, 12, 1, 0], "Programs at or above 70% completion, by school size (students)",
                   "Percent of programs", fmt="%{text}%")
    with b:
        simple_bar(["Public", "Private for-profit", "Private nonprofit"], [10, 45, 52],
                   "Programs at or above 70% completion, by school type", "Percent of programs", fmt="%{text}%")
    st.caption(
        "Counts are programs, so large schools with many programs weigh heavily. The public group may include a few "
        "programs with no school-type value. We do not know why these gaps exist. They could reflect who each type "
        "serves and how the completion rate is measured (it only covers first-time, full-time students), not program quality."
    )

    st.subheader("Where data went missing")
    gaps = pd.DataFrame([
        ("Programs with no occupation match", "768", "1.8%", "Program codes with no entry in the program-to-occupation crosswalk"),
        ("Programs with no school match", "2,108", "4.9%", "903 have no school ID, and 1,205 belong to schools missing from the institution file"),
        ("Schools with no county match", "341", "5.4%", "School ZIP codes not found in the HUD ZIP-to-county file"),
        ("Schools outside any metro area", "614", "9.8%", "Rural counties with no metropolitan or micropolitan area"),
        ("Programs with no county economic data", "2,395", "5.5%", "Follows from the county and metro gaps"),
        ("Programs with no local wage match", "18,843", "43.4%", "No occupation match, no metro area, or the metro does not report that occupation"),
    ], columns=["Gap", "Count", "Share", "Likely cause"])
    st.dataframe(gaps, hide_index=True)
    st.caption("Percentages are of the 43,465 certificate programs, except the school-level rows (of 6,273 schools). "
               "No rows were dropped. Missing values were kept and flagged.")

    st.subheader("Other limits")
    st.markdown(
        "- **Completion is school-level.** Every program at a school shares one value, and we found no public program-level rate.\n"
        "- **Placement is mostly missing.** College Scorecard's employment figures are about 84% missing for certificates.\n"
        "- **Program length is rarely available.** It covers only a small share of programs, and where it exists most programs look longer than the 8 to 15 week range.\n"
        "- **Colorado is a small training set.** The outcome model learned from 343 programs at 22 schools."
    )

# ---------- About ----------
with tab_about:
    st.subheader("What this score is")
    st.write(
        "A readiness score estimates how closely a certificate program resembles programs whose graduates "
        "tend to earn well, in places with local demand for the occupation. It helps prioritize where to look. "
        "It does not say whether a program qualifies for Workforce Pell. Eligibility depends on program length, "
        "program-level completion and placement, and state approval, none of which public data gives us."
    )
    st.subheader("How it is built")
    st.markdown(
        "- **Outcome score (0 to 100):** a model trained on 343 Colorado certificate programs at 22 schools "
        "predicts whether a program's graduates' Year 1 median earnings beat the high school baseline in the "
        "school's county. It is applied to programs in every state, which is an extrapolation. Only subjects "
        "with at least 5 Colorado programs are scored.\n"
        "- **Demand score (0 to 100):** the average percentile of local employment and job concentration for the "
        "occupation linked to the program, in the school's metro area. It is blank where no wage data matched.\n"
        "- **Overall score and tier:** the outcome percentile averaged with the demand score, split into thirds "
        "(Low, Medium, High). Tiers are relative to other scored programs. Programs without a demand score "
        "use the outcome part alone, and the 'Based on' column says which recipe applies.\n"
        "- **School completion rate:** the school's real completion rate from College Scorecard, shown "
        "separately. Every program at a school shares the same value."
    )
    st.subheader("Known limits")
    st.markdown(
        "- Subject drives most of the outcome score, so it separates subjects more than schools.\n"
        "- Colorado earnings only count graduates found in Colorado wage records who earned at least a "
        "full-time minimum wage.\n"
        "- We found no public program-level completion or placement data, and program length is available "
        "for only a small share of programs.\n"
        "- The outcome model's cross-validated ROC-AUC on Colorado was about 0.77, with only 22 schools, so "
        "treat scores as rough guides.\n"
        "- Roughly 40% of programs have no local wage match, so their demand score is blank."
    )
    st.subheader("Data sources")
    st.write(
        "College Scorecard, Colorado Department of Higher Education return on investment report, "
        "NCES CIP-to-SOC crosswalk, BLS Occupational Employment and Wage Statistics, "
        "Census American Community Survey, HUD ZIP-to-county crosswalk, Census metro area definitions."
    )
