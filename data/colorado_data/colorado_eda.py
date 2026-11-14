"""
EDA on the Colorado data, Workforce Pell Opportunity Score (Sage Road Solutions, Group 1B)
Author: Gabby Bui
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------------------
# Paths and output
# ----------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent  # data/colorado_data
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
LOG_PATH = HERE / "eda_output.txt"

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 80)


class Tee:
    """Print to the screen and to eda_output.txt at the same time."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.screen = sys.stdout

    def write(self, text):
        self.screen.write(text)
        self.file.write(text)

    def flush(self):
        self.screen.flush()
        self.file.flush()


sys.stdout = Tee(LOG_PATH)


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def sub(title):
    print("\n--- " + title + " ---")


# ----------------------------------------------------------------------------------
# Figure style (one consistent look for every chart)
# ----------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#a3a29c"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.size": 10,
})


def save(fig, name, note=None):
    if note:
        fig.text(0.01, 0.005, note, fontsize=8, color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(FIG_DIR / name, dpi=160)
    plt.close(fig)
    print(f"[saved figure] figures/{name}")


def dollars(x, _=None):
    return f"${x / 1000:.0f}k"


def short_name(name, n=34):
    name = str(name)
    return name if len(name) <= n else name[: n - 1] + "…"


# ----------------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------------
y1 = pd.read_csv(DATA_DIR / "year_1_clean.csv")
mt = pd.read_csv(DATA_DIR / "colorado_model_table.csv", dtype={"COUNTY_FIPS": str})

y1["CIP_FAMILY"] = y1["CIPCODE"] // 100
mt["CIP_FAMILY"] = mt["CIPCODE"] // 100
y1["SHORT"] = np.where(y1["DEGREE"].str.contains("<1"), "Under 1 year", "Over 1 year")
mt["SHORT"] = np.where(mt["DEGREE"].str.contains("<1"), "Under 1 year", "Over 1 year")
LEN_COLORS = {"Under 1 year": BLUE, "Over 1 year": ORANGE}

# Official 2-digit CIP family names (NCES CIP 2020), for the families that show up here
CIP_NAMES = {
    1: "Agriculture", 3: "Natural Resources", 9: "Communication", 10: "Communications Tech",
    11: "Computer & Info Sciences", 12: "Personal & Culinary Services", 13: "Education",
    14: "Engineering", 15: "Engineering Technologies", 19: "Family & Consumer Sciences",
    22: "Legal Professions", 24: "Liberal Arts", 26: "Biological Sciences", 30: "Interdisciplinary",
    31: "Parks & Recreation", 39: "Theology", 41: "Science Technologies", 42: "Psychology",
    43: "Law Enforcement & Firefighting", 44: "Public Administration", 45: "Social Sciences",
    46: "Construction Trades", 47: "Mechanic & Repair Tech", 48: "Precision Production",
    49: "Transportation", 50: "Visual & Performing Arts", 51: "Health Professions", 52: "Business",
}


# ==================================================================================
# PART 1. year_1_clean.csv
# ==================================================================================
section("PART 1. year_1_clean.csv (the cleaned Colorado earnings file)")

# ---- 1.1 Overview ---------------------------------------------------------------
sub("1.1 Overview")
print("Rows (programs):", len(y1), "| Columns:", y1.shape[1])
print("Schools:", y1["ENTITY_ID"].nunique(), "| CIP codes:", y1["CIPCODE"].nunique(),
      "| CIP families (2-digit):", y1["CIP_FAMILY"].nunique())
print("\nPrograms by certificate length:")
print(y1["DEGREE"].value_counts().to_string())
print("\nPrograms by ROI group:")
print(y1["ROI_GROUP"].value_counts().to_string())
print("\nPrograms per school (summary):")
print(y1.groupby("ENTITY_NAME").size().describe().round(1).to_string())

# ---- 1.2 Quality checks ---------------------------------------------------------
sub("1.2 Quality checks (every line should say 0 problems)")
key = ["ENTITY_ID", "CIPCODE", "DEGREE_LEVEL_ID"]
checks = {
    "duplicate school + CIP + certificate length rows": y1.duplicated(key).sum(),
    "EARN_Q1 above EARN_MEDIAN": (y1["EARN_Q1"] > y1["EARN_MEDIAN"]).sum(),
    "EARN_MEDIAN above EARN_Q3": (y1["EARN_MEDIAN"] > y1["EARN_Q3"]).sum(),
    "PCT_WITH_WAGE outside 0 to 1": ((y1["PCT_WITH_WAGE"] < 0) | (y1["PCT_WITH_WAGE"] > 1)).sum(),
    "COMPLETERS_WITH_WAGE above COMPLETERS": (y1["COMPLETERS_WITH_WAGE"] > y1["COMPLETERS"]).sum(),
    "PCT_WITH_WAGE not equal to with-wage / completers (off by > 0.01)":
        ((y1["COMPLETERS_WITH_WAGE"] / y1["COMPLETERS"] - y1["PCT_WITH_WAGE"]).abs() > 0.01).sum(),
    "non-positive earnings": (y1[["EARN_Q1", "EARN_MEDIAN", "EARN_Q3"]] <= 0).any(axis=1).sum(),
    "COMPLETERS_SUPPRESSED flag but COMPLETERS present":
        (y1["COMPLETERS_SUPPRESSED"] & y1["COMPLETERS"].notna()).sum(),
    "EARNINGS_SUPPRESSED flag but EARN_MEDIAN present":
        (y1["EARNINGS_SUPPRESSED"] & y1["EARN_MEDIAN"].notna()).sum(),
    "EARN_MEDIAN missing but not flagged":
        (~y1["EARNINGS_SUPPRESSED"] & y1["EARN_MEDIAN"].isna()).sum(),
}
for k, v in checks.items():
    print(f"{v:>4} problems | {k}")
print("Earnings range: $%s to $%s" % (f"{y1['EARN_MEDIAN'].min():,.0f}", f"{y1['EARN_MEDIAN'].max():,.0f}"))
print("Completers range:", y1["COMPLETERS"].min(), "to", y1["COMPLETERS"].max())

# ---- 1.3 Hidden (suppressed) values ---------------------------------------------
sub("1.3 Hidden values: who gets suppressed?")
y1["STATUS"] = np.select(
    [y1["COMPLETERS_SUPPRESSED"], y1["EARNINGS_SUPPRESSED"]],
    ["Everything hidden", "Earnings hidden"],
    default="Visible",
)
y1["VISIBLE"] = (y1["STATUS"] == "Visible").astype(int)
print(y1["STATUS"].value_counts().to_string())
print("Share with visible earnings:", round(y1["VISIBLE"].mean(), 3))

print("\nShare visible by certificate length:")
print(y1.groupby("DEGREE")["VISIBLE"].agg(["size", "mean"]).round(2).to_string())
print("\nShare visible by ROI group:")
roi_vis = y1.groupby("ROI_GROUP")["VISIBLE"].agg(["size", "mean"]).sort_values("mean")
print(roi_vis.round(2).to_string())

print("\nCompleters, visible vs earnings hidden (completers are known for both):")
print(y1[y1["STATUS"] != "Everything hidden"].groupby("STATUS")["COMPLETERS"]
      .describe().round(1).to_string())

school_vis = (y1.groupby("ENTITY_NAME")
              .agg(programs=("VISIBLE", "size"), visible=("VISIBLE", "sum"), share=("VISIBLE", "mean"))
              .sort_values("share"))
print("\nShare visible by school:")
print(school_vis.round(2).to_string())

# Figure 1. Share visible by ROI group, split by certificate length
grp = (y1.groupby(["ROI_GROUP", "SHORT"])["VISIBLE"].agg(["mean", "size"]).reset_index())
order = roi_vis.index.tolist()
fig, ax = plt.subplots(figsize=(9, 5.2))
ypos = np.arange(len(order))
h = 0.38
for i, length in enumerate(["Under 1 year", "Over 1 year"]):
    g = grp[grp["SHORT"] == length].set_index("ROI_GROUP").reindex(order)
    ax.barh(ypos + (i - 0.5) * h, g["mean"], height=h - 0.04, color=LEN_COLORS[length],
            label=length, edgecolor=SURFACE, linewidth=1)
    for yy, (m, n) in zip(ypos + (i - 0.5) * h, zip(g["mean"], g["size"])):
        if pd.notna(m):
            ax.text(m + 0.01, yy, f"{m:.0%} (n={int(n)})", va="center", fontsize=8, color=INK_2)
ax.set_yticks(ypos)
ax.set_yticklabels([short_name(o, 40) for o in order])
ax.set_xlim(0, 1.12)
ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
ax.set_xlabel("Share of programs with visible earnings")
ax.grid(axis="y", visible=False)
ax.set_title("Fig 1. Share of programs with visible earnings")
ax.legend(loc="lower right", title="Certificate length", title_fontsize=9)
save(fig, "01_visible_share_by_field.png", "Source: year_1_clean.csv (747 programs).")

# Figure 2. Program size: visible vs earnings hidden
fig, ax = plt.subplots(figsize=(8, 4.2))
bins = np.logspace(1, np.log10(y1["COMPLETERS"].max() + 1), 30)
for status, color in [("Visible", BLUE), ("Earnings hidden", ORANGE)]:
    vals = y1.loc[y1["STATUS"] == status, "COMPLETERS"].dropna()
    ax.hist(vals, bins=bins, color=color, alpha=0.25)
    ax.hist(vals, bins=bins, histtype="step", color=color, linewidth=2, label=f"{status} (n={len(vals)})")
ax.set_xscale("log")
ax.set_xlabel("Number of completers (log scale)")
ax.set_ylabel("Programs")
ax.set_title("Fig 2. Programs with hidden earnings are much smaller")
ax.legend()
save(fig, "02_completers_visible_vs_hidden.png",
     "The 270 programs with everything hidden have no completer count, so they are not shown.")

# ---- 1.4 Earnings ----------------------------------------------------------------
sub("1.4 Earnings (programs with visible earnings only)")
vis = y1[y1["VISIBLE"] == 1].copy()
vis["SPREAD"] = vis["EARN_Q3"] - vis["EARN_Q1"]
print("Median earnings, summary:")
print(vis["EARN_MEDIAN"].describe().round(0).to_string())
print("Skew:", round(vis["EARN_MEDIAN"].skew(), 2))
print("\nBy certificate length:")
print(vis.groupby("DEGREE")["EARN_MEDIAN"].describe().round(0).to_string())
print("\nBy ROI group (sorted by median):")
roi_earn = vis.groupby("ROI_GROUP")["EARN_MEDIAN"].agg(["size", "median", "min", "max"])
print(roi_earn.sort_values("median").round(0).to_string())
print("\nQ1 to Q3 spread, summary:")
print(vis["SPREAD"].describe().round(0).to_string())

# Figure 3. Earnings distribution by certificate length
fig, ax = plt.subplots(figsize=(8, 4.2))
bins = np.linspace(vis["EARN_MEDIAN"].min(), vis["EARN_MEDIAN"].max(), 30)
for length in ["Under 1 year", "Over 1 year"]:
    vals = vis.loc[vis["SHORT"] == length, "EARN_MEDIAN"]
    ax.hist(vals, bins=bins, color=LEN_COLORS[length], alpha=0.25)
    ax.hist(vals, bins=bins, histtype="step", color=LEN_COLORS[length], linewidth=2,
            label=f"{length} (n={len(vals)}, median ${vals.median():,.0f})")
ax.xaxis.set_major_formatter(dollars)
ax.set_xlabel("Year 1 median earnings")
ax.set_ylabel("Programs")
ax.set_title("Fig 3. Year 1 median earnings by certificate length")
ax.legend()
save(fig, "03_earnings_by_length.png", "Source: year_1_clean.csv, 347 programs with visible earnings.")

# Figure 4. Earnings by ROI group (dots = programs, black line = group median)
order = roi_earn.sort_values("median").index.tolist()
fig, ax = plt.subplots(figsize=(9, 5.2))
rng = np.random.default_rng(0)
for i, g in enumerate(order):
    sub_df = vis[vis["ROI_GROUP"] == g]
    jitter = rng.uniform(-0.22, 0.22, len(sub_df))
    ax.scatter(sub_df["EARN_MEDIAN"], i + jitter, s=14, color=BLUE, alpha=0.55, linewidths=0)
    med = sub_df["EARN_MEDIAN"].median()
    ax.plot([med, med], [i - 0.32, i + 0.32], color=INK, linewidth=2)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{short_name(g, 36)} (n={int(roi_earn.loc[g, 'size'])})" for g in order])
ax.xaxis.set_major_formatter(dollars)
ax.set_xlabel("Year 1 median earnings")
ax.grid(axis="y", visible=False)
ax.set_title("Fig 4. Year 1 median earnings by field")
save(fig, "04_earnings_by_field.png", "Each dot is a program. The black line is the median for that field.")

# ---- 1.5 Placement ---------------------------------------------------------------
sub("1.5 Placement (PCT_WITH_WAGE = share of completers found in Colorado wage records)")
print(vis["PCT_WITH_WAGE"].describe().round(3).to_string())
print("Programs with placement under 50%:", int((vis["PCT_WITH_WAGE"] < 0.5).sum()),
      "of", len(vis), f"({(vis['PCT_WITH_WAGE'] < 0.5).mean():.0%})")
print("\nBy certificate length:")
print(vis.groupby("DEGREE")["PCT_WITH_WAGE"].describe().round(2).to_string())
print("\nBy ROI group (sorted by median):")
print(vis.groupby("ROI_GROUP")["PCT_WITH_WAGE"].agg(["size", "median"]).sort_values("median")
      .round(2).to_string())
rho = vis["EARN_MEDIAN"].corr(vis["PCT_WITH_WAGE"], method="spearman")
print("\nSpearman correlation, earnings vs placement:", round(rho, 2))

# Figure 5. Earnings vs placement
fig, ax = plt.subplots(figsize=(8, 5))
for length in ["Under 1 year", "Over 1 year"]:
    d = vis[vis["SHORT"] == length]
    ax.scatter(d["PCT_WITH_WAGE"], d["EARN_MEDIAN"], s=22, color=LEN_COLORS[length], alpha=0.7,
               edgecolor=SURFACE, linewidth=0.8, label=length)
ax.axvline(0.5, color=INK_2, linewidth=1, linestyle="--")
ax.text(0.505, ax.get_ylim()[1] * 0.97, "50% placement cutoff used in LABEL", fontsize=8,
        color=INK_2, va="top")
ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
ax.yaxis.set_major_formatter(dollars)
ax.set_xlabel("Placement (share of completers with a Colorado wage record)")
ax.set_ylabel("Year 1 median earnings")
ax.set_title(f"Fig 5. Higher earning programs also tend to place more people (Spearman {rho:.2f})")
ax.legend(title="Certificate length", title_fontsize=9, loc="upper left")
save(fig, "05_earnings_vs_placement.png", "Source: year_1_clean.csv, 347 programs with visible earnings.")


# ==================================================================================
# PART 2. colorado_model_table.csv
# ==================================================================================
section("PART 2. colorado_model_table.csv (the model-ready table)")

# ---- 2.1 Coverage ----------------------------------------------------------------
sub("2.1 Coverage: from 747 programs to the model table")
print("Rows:", len(mt), "| Columns:", mt.shape[1])
print("Schools:", mt["UNITID"].nunique(), "| Counties:", mt["COUNTY_FIPS"].nunique(),
      "| CIP families:", mt["CIP_FAMILY"].nunique())
print("Duplicate UNITID + CIP + certificate length rows:",
      mt.duplicated(["UNITID", "CIPCODE", "DEGREE"]).sum())
print(f"\n747 programs -> {len(vis)} with visible earnings -> {len(mt)} in the model table")

k = ["ENTITY_NAME", "CIPCODE", "DEGREE"]
lost = vis.merge(mt[k].assign(IN_TABLE=1), on=k, how="left")
lost = lost[lost["IN_TABLE"].isna()]
print(f"\nVisible programs that did not make it into the table: {len(lost)}")
print(lost[["ENTITY_NAME", "CIPCODE", "DEGREE", "COMPLETERS", "EARN_MEDIAN"]].to_string(index=False))
print("(colorado_label.ipynb joins on school + 4-digit CIP. These 4 have no school + CIP row in"
      " victoria_datasets/clean_dataset.csv, so they drop out.)")

missing_schools = sorted(set(y1["ENTITY_NAME"]) - set(mt["ENTITY_NAME"]))
print(f"\nSchools in year_1_clean but not in the model table: {len(missing_schools)}")
print(school_vis.loc[missing_schools, ["programs", "visible"]].to_string())

print("\nModel table vs all visible programs (is the table still representative?):")
comp = pd.DataFrame({
    "all visible": [len(vis), vis["EARN_MEDIAN"].median(), vis["PCT_WITH_WAGE"].median(),
                    (vis["SHORT"] == "Under 1 year").mean()],
    "model table": [len(mt), mt["EARN_MEDIAN"].median(), mt["PLACEMENT"].median(),
                    (mt["SHORT"] == "Under 1 year").mean()],
}, index=["programs", "median earnings", "median placement", "share under 1 year"])
print(comp.round(2).to_string())

sub("2.1b Does the label match its own definition?")
rebuilt = ((mt["EARN_MEDIAN"] > mt["BASELINE"]) & (mt["PLACEMENT"] >= 0.5)).astype(int)
print("LABEL rows that disagree with (earnings > baseline) and (placement >= 50%):",
      int((rebuilt != mt["LABEL"]).sum()))
print("LABEL_EARNINGS rows that disagree with (earnings > baseline):",
      int(((mt["EARN_MEDIAN"] > mt["BASELINE"]).astype(int) != mt["LABEL_EARNINGS"]).sum()))
print("BASELINE equal to B20004_003E in every row:",
      bool(np.allclose(mt["BASELINE"], mt["B20004_003E"])))
print("Missing values in EARN_MEDIAN / PLACEMENT / BASELINE:",
      mt[["EARN_MEDIAN", "PLACEMENT", "BASELINE"]].isna().sum().tolist())

sub("2.1c Short and long certificates in the same field share one set of features")
# The join is on school + CIP only, so a school's <1 year and >1 year certificate in the same
# CIP get the exact same feature values. DEGREE is left out of the model, so these rows look
# identical to the model.
pair_key = ["UNITID", "CIPCODE"]
shared = mt[mt.duplicated(pair_key, keep=False)]
n_pairs = shared.groupby(pair_key).ngroups
pairs_diff = int((shared.groupby(pair_key)["LABEL"].nunique() > 1).sum())
same_count = int((shared.groupby(pair_key)["IPEDSCOUNT1"].nunique(dropna=False) == 1).sum())
print(f"Rows that share a school + CIP with another row: {len(shared)} of {len(mt)} "
      f"({len(shared) / len(mt):.0%}), in {n_pairs} pairs")
print(f"Pairs where IPEDSCOUNT1 (program size) is also identical: {same_count} of {n_pairs}")
print(f"Pairs with the SAME features but DIFFERENT labels: {pairs_diff} of {n_pairs}")
print("Inside those pairs, which certificate gets LABEL = 1?")
diff_pairs = shared.groupby(pair_key).filter(lambda g: g["LABEL"].nunique() > 1)
print(diff_pairs.groupby("DEGREE")["LABEL"].sum().rename("rows with LABEL = 1").to_string())

# ---- 2.2 The label -----------------------------------------------------------
sub("2.2 The label")
print("LABEL (composite) balance:")
print(mt["LABEL"].value_counts().to_string(), "| share 1 =", round(mt["LABEL"].mean(), 3))
print("LABEL_EARNINGS balance:")
print(mt["LABEL_EARNINGS"].value_counts().to_string(), "| share 1 =", round(mt["LABEL_EARNINGS"].mean(), 3))

beat = mt["EARN_MEDIAN"] > mt["BASELINE"]
placed = mt["PLACEMENT"] >= 0.5
mt["WHY"] = np.select(
    [beat & placed, beat & ~placed, ~beat & placed],
    ["Passes both", "Fails placement only", "Fails earnings only"],
    default="Fails both",
)
print("\nWhy programs get each label:")
print(mt["WHY"].value_counts().to_string())

mt["MARGIN"] = mt["EARN_MEDIAN"] - mt["BASELINE"]
mt["RATIO"] = mt["EARN_MEDIAN"] / mt["BASELINE"]
print("\nEarnings minus baseline ($):")
print(mt["MARGIN"].describe().round(0).to_string())
print("Baseline (county HS earnings) summary:")
print(mt["BASELINE"].describe().round(0).to_string())
print("Distinct baseline values:", mt["BASELINE"].nunique())

near_earn = mt["MARGIN"].abs() <= 2000
near_place = (mt["PLACEMENT"] - 0.5).abs() <= 0.05
print(f"\nFragile labels: within $2,000 of the baseline = {int(near_earn.sum())}, "
      f"within 5 points of the 50% cutoff = {int(near_place.sum())}, "
      f"either = {int((near_earn | near_place).sum())} of {len(mt)}")

def rate_table(col):
    return (mt.groupby(col)
            .agg(programs=("LABEL", "size"), label_rate=("LABEL", "mean"),
                 earnings_only_rate=("LABEL_EARNINGS", "mean"))
            .sort_values("programs", ascending=False).round(2))

print("\nLabel rate by certificate length:")
print(rate_table("DEGREE").to_string())
print("\nWhy programs get each label, by certificate length (row shares):")
print(pd.crosstab(mt["DEGREE"], mt["WHY"], normalize="index").round(2).to_string())
print("\nLabel rate by CIP family (2-digit CIP, families with 5+ programs):")
cip = rate_table("CIP_FAMILY")
cip = cip[cip["programs"] >= 5]
cip.index = [f"{int(i):02d}" for i in cip.index]
desc = (mt.assign(F=mt["CIP_FAMILY"].map(lambda v: f"{int(v):02d}"))
        .groupby("F")["CIPDESC"].agg(lambda s: s.value_counts().index[0]))
cip.insert(0, "family", [CIP_NAMES.get(int(i), "?") for i in cip.index])
cip["most common program"] = [short_name(desc[i], 45) for i in cip.index]
print(cip.to_string())

# Figure 11. Label rate by CIP family: composite vs earnings only
cf = cip.sort_values("label_rate")
fig, ax = plt.subplots(figsize=(9.5, 6))
ypos = np.arange(len(cf))
ax.hlines(ypos, cf["label_rate"], cf["earnings_only_rate"], color=GRID, linewidth=3)
ax.scatter(cf["earnings_only_rate"], ypos, s=60, color=ORANGE, edgecolor=SURFACE, linewidth=1.5,
           zorder=3, label="Earnings only (LABEL_EARNINGS)")
ax.scatter(cf["label_rate"], ypos, s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
           zorder=4, label="Earnings + placement (LABEL)")
ax.set_yticks(ypos)
ax.set_yticklabels([f"{i} {t} (n={int(n)})"
                    for i, t, n in zip(cf.index, cf["family"], cf["programs"])], fontsize=9)
ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
ax.set_xlim(-0.03, 1.03)
ax.set_xlabel("Share of programs labeled 1")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right")
ax.set_title("Fig 11. Field matters a lot, and the placement rule hits some fields hard")
save(fig, "11_label_rate_by_field.png",
     "CIP families with 5+ programs. The gap between the dots is the placement rule. One dot = the two rates are equal.")

print("\nLabel rate by school:")
school_rate = rate_table("ENTITY_NAME")
print(school_rate.to_string())
print("\nLabel rate by county:")
county_rate = (mt.groupby(["COUNTY_FIPS", "CBSA Title"], dropna=False)
               .agg(programs=("LABEL", "size"), label_rate=("LABEL", "mean"),
                    baseline=("BASELINE", "first"))
               .sort_values("programs", ascending=False).round(2))
print(county_rate.to_string())

# Figure 6. The label as a picture: the two cutoffs
why_colors = {"Passes both": AQUA, "Fails placement only": ORANGE,
              "Fails earnings only": BLUE, "Fails both": GRAY}
fig, ax = plt.subplots(figsize=(8.5, 5.6))
for why, color in why_colors.items():
    d = mt[mt["WHY"] == why]
    ax.scatter(d["PLACEMENT"], d["RATIO"], s=24, color=color, alpha=0.8, edgecolor=SURFACE,
               linewidth=0.8, label=f"{why} (n={len(d)})")
ax.axvline(0.5, color=INK_2, linewidth=1, linestyle="--")
ax.axhline(1.0, color=INK_2, linewidth=1, linestyle="--")
ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
ax.yaxis.set_major_formatter(lambda x, _: f"{x:.1f}x")
ax.set_xlabel("Placement (share with a Colorado wage record)")
ax.set_ylabel("Earnings ÷ county high school baseline")
ax.set_title("Fig 6. LABEL = 1 only in the top-right box (both cutoffs passed)")
ax.legend(loc="upper left", fontsize=8.5)
save(fig, "06_label_two_cutoffs.png", "Source: colorado_model_table.csv (343 programs). Dashed lines are the cutoffs.")

# Figure 7. Label rate by school
sr = school_rate.sort_values("label_rate")
fig, ax = plt.subplots(figsize=(9, 6.8))
ax.barh(range(len(sr)), sr["label_rate"], color=BLUE, height=0.7, edgecolor=SURFACE)
for i, (r, n) in enumerate(zip(sr["label_rate"], sr["programs"])):
    ax.text(r + 0.01, i, f"{r:.0%}  (n={int(n)})", va="center", fontsize=8, color=INK_2, zorder=4,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
ax.axvline(mt["LABEL"].mean(), color=INK, linewidth=1, linestyle="--", zorder=3)
ax.text(mt["LABEL"].mean() + 0.01, len(sr) - 0.4, f"overall {mt['LABEL'].mean():.0%}",
        fontsize=8, color=INK)
ax.set_yticks(range(len(sr)))
ax.set_yticklabels([short_name(s, 38) for s in sr.index])
ax.set_xlim(0, 1.15)
ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
ax.set_xlabel("Share of programs with LABEL = 1")
ax.grid(axis="y", visible=False)
ax.set_title("Fig 7. The label rate swings a lot from school to school")
save(fig, "07_label_rate_by_school.png", "Schools with few programs (small n) are noisy, so read them with care.")

# ---- 2.3 Features vs the label --------------------------------------------------
sub("2.3 Features vs the label")
# Same feature list as notebooks/model_setup_colorado.py, so the EDA matches the model
mt["LOCALE_GROUP"] = mt["LOCALE"] // 10
for c in ["IPEDSCOUNT1", "UGDS", "B01003_001E"]:
    mt[c + "_LOG"] = np.log1p(mt[c])
FEATURES = {
    "IPEDSCOUNT1_LOG": ("program completers (log)", "program"),
    "CIP_FAMILY": ("CIP family (2-digit)", "program"),
    "UGDS_LOG": ("school enrollment (log)", "school"),
    "PCTPELL": ("share of students on Pell", "school"),
    "PCTFLOAN": ("share with federal loans", "school"),
    "UG25ABV": ("share of students 25+", "school"),
    "LOCALE_GROUP": ("locale (city/suburb/town/rural)", "school"),
    "CONTROL_inst": ("public / private", "school"),
    "HBCU": ("HBCU", "school"),
    "HSI": ("Hispanic-serving", "school"),
    "B01003_001E_LOG": ("county population (log)", "county"),
    "B19013_001E": ("county median household income", "county"),
    "S1501_C02_010E": ("county ACS education share (010E)", "county"),
    "S1501_C02_014E": ("county ACS education share (014E)", "county"),
    "S1501_C02_015E": ("county ACS education share (015E)", "county"),
    "S2301_C02_001E": ("county labor force participation", "county"),
    "S2301_C04_001E": ("county unemployment rate", "county"),
    "NO_ACS": ("missing ACS flag", "county"),
}
OTHER = {  # not model features, checked for leakage or for the demand score
    "BASELINE": "county HS baseline (= B20004_003E)",
    "EARN_MDN_1YR_CLEAN": "Scorecard 1-yr earnings",
    "PLACEMENT_RATE": "Scorecard placement rate",
    "DEBT_ALL_STGP_EVAL_MDN": "Scorecard median debt",
    "C150_L4": "school completion rate",
    "A_MEDIAN": "OEWS occupation median wage",
    "LOC_QUOTIENT": "OEWS location quotient",
    "TOT_EMP": "OEWS total employment",
}

rows = []
for col, (label, level) in {**FEATURES, **{k: (v, "other") for k, v in OTHER.items()}}.items():
    s = pd.to_numeric(mt[col], errors="coerce").astype(float)
    ok = s.notna()
    auc = np.nan
    if ok.sum() > 10 and s[ok].nunique() > 1 and mt.loc[ok, "LABEL"].nunique() == 2:
        auc = roc_auc_score(mt.loc[ok, "LABEL"], s[ok])
    miss_rate = mt.loc[~ok, "LABEL"].mean() if (~ok).any() else np.nan
    rows.append({
        "column": col, "meaning": label, "level": level,
        "missing_%": round(100 * (~ok).mean(), 1),
        "distinct values": s.nunique(),
        "label rate if missing": round(miss_rate, 2) if pd.notna(miss_rate) else "",
        "AUC": round(auc, 3) if pd.notna(auc) else np.nan,
        "strength": round(abs(auc - 0.5) * 2, 3) if pd.notna(auc) else np.nan,
        "direction": ("higher -> more 1s" if auc > 0.5 else "higher -> more 0s") if pd.notna(auc) else "",
    })
feat_tab = pd.DataFrame(rows)
print("AUC here = how well one column alone sorts LABEL 1 from 0 (0.5 = no signal).")
print("strength = |AUC - 0.5| x 2, from 0 (none) to 1 (perfect).\n")
print(feat_tab.sort_values(["level", "strength"], ascending=[True, False]).to_string(index=False))
print("\nOverall label rate for comparison:", round(mt["LABEL"].mean(), 2))
print("Features with only one value (carry no information here):",
      feat_tab.loc[feat_tab["distinct values"] <= 1, "column"].tolist())
print("Note: CIP_FAMILY is a code, not a quantity, so its AUC above is not meaningful.")
print("Spread of label rate across CIP families with 5+ programs:",
      f"{cip['label_rate'].min():.0%} to {cip['label_rate'].max():.0%} (see Fig 11)")

sub("2.3b Does our program size feature agree with Colorado's own completer count?")
size_chk = mt.merge(y1[["ENTITY_NAME", "CIPCODE", "DEGREE", "COMPLETERS"]],
                    on=["ENTITY_NAME", "CIPCODE", "DEGREE"], how="left")
print("Spearman, IPEDSCOUNT1 vs Colorado COMPLETERS:",
      round(size_chk["IPEDSCOUNT1"].corr(size_chk["COMPLETERS"], method="spearman"), 2))
print("IPEDSCOUNT1 = 0 even though Colorado reports completers:",
      int(((size_chk["IPEDSCOUNT1"] == 0) & (size_chk["COMPLETERS"] > 0)).sum()),
      "| IPEDSCOUNT1 missing:", int(size_chk["IPEDSCOUNT1"].isna().sum()))

print("\nHow many distinct values each level of feature can take:")
print("  school-level features:", mt["UNITID"].nunique(), "schools, so at most that many values")
print("  county-level features:", mt["COUNTY_FIPS"].nunique(), "counties, so at most that many values")
print("  schools per county:")
print(mt.groupby("COUNTY_FIPS")["UNITID"].nunique().value_counts().sort_index()
      .rename_axis("schools in county").rename("counties").to_string())

# Figure 8. One-feature signal strength
ft = (feat_tab[(feat_tab["level"] != "other") & (feat_tab["column"] != "CIP_FAMILY")]
      .dropna(subset=["AUC"]).sort_values("strength"))
level_colors = {"program": ORANGE, "school": BLUE, "county": AQUA}
fig, ax = plt.subplots(figsize=(9, 6.4))
ax.barh(range(len(ft)), ft["strength"], color=[level_colors[l] for l in ft["level"]],
        height=0.7, edgecolor=SURFACE)
for i, (s, a, d) in enumerate(zip(ft["strength"], ft["AUC"], ft["direction"])):
    arrow = "↑" if "1s" in d else "↓"
    ax.text(s + 0.005, i, f"AUC {a:.2f} {arrow}", va="center", fontsize=8, color=INK_2)
ax.set_yticks(range(len(ft)))
ax.set_yticklabels(ft["meaning"])
ax.set_xlabel("Signal strength on its own, |AUC − 0.5| × 2")
ax.set_xlim(0, max(ft["strength"].max() * 1.35, 0.1))
ax.grid(axis="y", visible=False)
ax.legend(handles=[Line2D([0], [0], color=c, linewidth=8, label=f"{k}-level")
                   for k, c in level_colors.items()], loc="lower right")
ax.set_title("Fig 8. How much each model feature says about LABEL by itself")
save(fig, "08_feature_signal.png",
     "↑ means higher values go with more LABEL = 1, ↓ the opposite. CIP family is a category, so it is in Fig 11 instead.")

# Figure 9. Correlation between model features
num_feats = [c for c in FEATURES if mt[c].nunique() > 2 and c != "CIP_FAMILY"]
corr = mt[num_feats].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
fig, ax = plt.subplots(figsize=(9, 7.6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_feats)))
ax.set_yticks(range(len(num_feats)))
names = [FEATURES[c][0] for c in num_feats]
ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
ax.set_yticklabels(names, fontsize=8)
ax.grid(False)
for i in range(len(num_feats)):
    for j in range(len(num_feats)):
        v = corr.values[i, j]
        if i != j and abs(v) >= 0.7:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.85 else INK)
fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman correlation")
ax.set_title("Fig 9. Which model features move together (|ρ| ≥ 0.7 printed)")
save(fig, "09_feature_correlation.png")

print("\nFeature pairs with |Spearman| >= 0.8:")
pairs = []
for i, a in enumerate(num_feats):
    for b in num_feats[i + 1:]:
        if abs(corr.loc[a, b]) >= 0.8:
            pairs.append((FEATURES[a][0], FEATURES[b][0], round(corr.loc[a, b], 2)))
for p in sorted(pairs, key=lambda t: -abs(t[2])):
    print(f"  {p[2]:+.2f}  {p[0]}  <->  {p[1]}")
if not pairs:
    print("  none")

# Figure 10. Top features by label
top = ft.sort_values("strength", ascending=False).head(6)["column"].tolist()
fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
for ax, col in zip(axes.flat, top):
    data = [pd.to_numeric(mt.loc[mt["LABEL"] == v, col], errors="coerce").dropna() for v in [0, 1]]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "linewidth": 2})
    for patch, color in zip(bp["boxes"], [GRAY, AQUA]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(INK_2)
    for v, d in enumerate(data, start=1):
        ax.scatter(v + rng.uniform(-0.15, 0.15, len(d)), d, s=6, color=INK_2, alpha=0.35, linewidths=0)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"LABEL 0\n(n={len(data[0])})", f"LABEL 1\n(n={len(data[1])})"])
    ax.set_title(FEATURES[col][0], fontsize=10, fontweight="normal")
    ax.grid(axis="x", visible=False)
fig.suptitle("Fig 10. The six strongest single features, split by label", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
save(fig, "10_top_features_by_label.png", "Boxes show the middle 50%, the black line is the median, dots are programs.")

# ---- 2.4 Leakage checks -----------------------------------------------------
sub("2.4 Leakage checks (columns that are basically the label, keep them out of X)")
for a, b in [("EARN_MEDIAN", "EARN_MDN_1YR_CLEAN"), ("PLACEMENT", "PLACEMENT_RATE")]:
    both = mt[[a, b]].dropna()
    print(f"{a} vs {b}: {len(both)} programs have both, Spearman = "
          f"{both[a].corr(both[b], method='spearman'):.2f}")
leak = feat_tab[feat_tab["column"].isin(OTHER)][["column", "meaning", "missing_%", "AUC", "strength"]]
print(leak.sort_values("strength", ascending=False).to_string(index=False))

model_cols = set(FEATURES)
danger = {"BASELINE", "B20004_003E", "EARN_MEDIAN", "PLACEMENT", "LABEL_EARNINGS",
          "EARN_MDN_1YR_CLEAN", "PLACEMENT_RATE", "MARGIN", "RATIO"}
print("\nAny label ingredient inside the model feature list?",
      sorted(model_cols & danger) or "no, the feature list is clean")

section("Done. Figures are in figures/, the full printout is in eda_output.txt")
