
"""Validate model-ready Workforce Pell datasets without modifying them."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve paths from this file so validation works from any directory.
ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "national": ROOT / "data/national_data/clean_dataset.csv",
    "colorado": ROOT / "data/colorado_data/colorado_model_table.csv",
}

MODEL_SCRIPTS = {
    "national": ROOT / "notebooks/model_setup.py",
    "colorado": ROOT / "notebooks/model_setup_colorado.py",
}

# Generated predictors must be checked against their source CSV columns.
DERIVED_FEATURES = {
    "IPEDSCOUNT1_LOG": "IPEDSCOUNT1",
    "UGDS_LOG": "UGDS",
    "B01003_001E_LOG": "B01003_001E",
    "TOT_EMP_LOG": "TOT_EMP",
    "CIP_FAMILY": "CIPCODE",
    "LOCALE_GROUP": "LOCALE",
}

# Exclude target inputs and other outcome variables from model predictors.
EXCLUDED_FEATURES = {
    "national": {
        "C150_L4", "LABEL", "LABEL_EARNINGS",
        "EARN_MEDIAN", "PLACEMENT", "BASELINE",
        "EARN_MDN_1YR_CLEAN", "PLACEMENT_RATE",
        "DEBT_ALL_STGP_EVAL_MDN",
    },
    "colorado": {
        "LABEL", "LABEL_EARNINGS", "EARN_MEDIAN",
        "PLACEMENT", "BASELINE", "B20004_003E",
        "DEGREE", "EARN_MDN_1YR_CLEAN",
        "PLACEMENT_RATE", "DEBT_ALL_STGP_EVAL_MDN",
    },
}

# Scorecard proportions use 0-1; ACS percentages use 0-100.
NUMERIC_RANGES = {
    "PCTPELL": (0, 1),
    "PCTFLOAN": (0, 1),
    "UG25ABV": (0, 1),
    "S1501_C02_010E": (0, 100),
    "S1501_C02_014E": (0, 100),
    "S1501_C02_015E": (0, 100),
    "S2301_C02_001E": (0, 100),
    "S2301_C04_001E": (0, 100),
    "IPEDSCOUNT1": (0, None),
    "UGDS": (0, None),
    "B01003_001E": (0, None),
    "B19013_001E": (0, None),
    "B20004_003E": (0, None),
    "TOT_EMP": (0, None),
    "A_MEDIAN": (0, None),
    "LOC_QUOTIENT": (0, None),
    "C150_L4": (0, 1),
    "PLACEMENT": (0, 1),
}

# Each flag tracks one source column, not the entire dataset.
MISSINGNESS_FLAGS = {
    "NO_ACS": "B19013_001E",
    "NO_OEWS_WAGE": "A_MEDIAN",
}


def get_model_features(path):
    """Extract feature_cols without executing the model setup script."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "feature_cols":
                features = ast.literal_eval(node.value)

                if not isinstance(features, list):
                    raise ValueError("feature_cols must be a list")

                if not all(isinstance(f, str) for f in features):
                    raise ValueError("Feature names must be strings")

                if len(features) != len(set(features)):
                    raise ValueError("Duplicate feature names")

                return features

    raise ValueError(f"feature_cols not found: {path}")


def required_columns(features, dataset):
    """Return the source columns needed by a model."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    required = {
        DERIVED_FEATURES.get(feature, feature)
        for feature in features
    }
    required.add("CONTROL_inst")

    if dataset == "national":
        required.update(["UNITID", "CIPCODE", "C150_L4"])
    else:
        required.update([
            "UNITID", "CIPCODE", "DEGREE",
            "EARN_MEDIAN", "PLACEMENT", "BASELINE", "LABEL",
        ])

    return required


def validate(df, dataset, features):
    """Return validation findings as severity, check, and affected-row count."""
    required = required_columns(features, dataset)
    findings = []

    def report(severity, check, count):
        if int(count):
            findings.append({
                "severity": severity,
                "check": check,
                "count": int(count),
            })

    for feature in sorted(set(features) & EXCLUDED_FEATURES[dataset]):
        report("error", f"Excluded predictor: {feature}", 1)

    for column in sorted(required - set(df.columns)):
        report("error", f"Missing column: {column}", 1)

    numeric_columns = set(NUMERIC_RANGES) | {
        "CIPCODE", "LOCALE", "EARN_MEDIAN", "BASELINE", "LABEL",
    }

    for column in sorted(numeric_columns & set(df.columns)):
        raw = df[column]
        numeric = pd.to_numeric(raw, errors="coerce")

        report(
            "error",
            f"Non-numeric: {column}",
            (raw.notna() & numeric.isna()).sum(),
        )

        report(
            "error",
            f"Infinite values: {column}",
            np.isinf(
                numeric.to_numpy(dtype=float, na_value=np.nan)
            ).sum(),
        )

        if column in NUMERIC_RANGES:
            low, high = NUMERIC_RANGES[column]

            if low is not None:
                report(
                    "error",
                    f"Below minimum: {column}",
                    (numeric < low).sum(),
                )

            if high is not None:
                report(
                    "error",
                    f"Above maximum: {column}",
                    (numeric > high).sum(),
                )

    # CSV imports may represent Boolean flags as text or numbers.
    accepted_flags = {
        "true": True, "false": False,
        "1": True, "0": False,
        "1.0": True, "0.0": False,
    }

    for flag, source in MISSINGNESS_FLAGS.items():
        if not {flag, source}.issubset(df.columns):
            continue

        actual = (
            df[flag].astype("string")
            .str.strip()
            .str.lower()
            .map(accepted_flags)
            .astype("boolean")
        )

        expected = df[source].isna()

        report(
            "error",
            f"Flag mismatch: {flag}",
            (
                actual.isna()
                | actual.ne(expected).fillna(True)
            ).sum(),
        )

    if dataset == "national":
        keys = ["UNITID", "CIPCODE"]

        if set(keys).issubset(df.columns):
            # Missing UNITIDs cannot reliably identify a school/program.
            valid = df.dropna(subset=keys)

            report(
                "error",
                "Duplicate national keys",
                valid.duplicated(keys).sum(),
            )

            report(
                "warning",
                "Missing national identifiers",
                df[keys].isna().any(axis=1).sum(),
            )

    else:
        # CDHE reports separate outcomes for certificates of different lengths.
        keys = ["UNITID", "CIPCODE", "DEGREE"]

        if set(keys).issubset(df.columns):
            report(
                "error",
                "Duplicate Colorado keys",
                df.duplicated(keys).sum(),
            )

            report(
                "error",
                "Missing Colorado identifiers",
                df[keys].isna().any(axis=1).sum(),
            )

        label_columns = {
            "EARN_MEDIAN", "BASELINE", "PLACEMENT", "LABEL"
        }

        if label_columns.issubset(df.columns):
            earnings = pd.to_numeric(
                df["EARN_MEDIAN"], errors="coerce"
            )
            baseline = pd.to_numeric(
                df["BASELINE"], errors="coerce"
            )
            placement = pd.to_numeric(
                df["PLACEMENT"], errors="coerce"
            )
            label = pd.to_numeric(
                df["LABEL"], errors="coerce"
            )
            # A positive label requires both earnings and placement thresholds.
            expected = (
                (earnings > baseline)
                & (placement >= 0.50)
            ).astype(int)

            invalid_inputs = (
                earnings.isna()
                | baseline.isna()
                | placement.isna()
                | ~np.isfinite(earnings)
                | ~np.isfinite(baseline)
                | ~np.isfinite(placement)
            )

            report(
                "error",
                "Missing or invalid label inputs",
                invalid_inputs.sum(),
            )

            report(
                "error",
                "Colorado label mismatch",
                (label.isna() | label.ne(expected)).sum(),
            )

            report(
                "error",
                "Invalid Colorado label",
                (label.notna() & ~label.isin([0, 1])).sum(),
            )

    return findings


def main():
    """Validate both datasets and exit with code 1 if errors are found."""
    errors = 0

    for dataset, path in DATASETS.items():
        features = get_model_features(MODEL_SCRIPTS[dataset])
        df = pd.read_csv(path, low_memory=False)
        findings = validate(df, dataset, features)

        print(f"\n=== {dataset.upper()} ===")
        print(f"Rows: {len(df)}")
        print(f"Explicit features: {len(features)}")

        if not findings:
            print("PASS: No errors or warnings")

        for finding in findings:
            print(
                f"{finding['severity'].upper()}: "
                f"{finding['check']} ({finding['count']})"
            )

            if finding["severity"] == "error":
                errors += 1

    print(f"\nValidation: {errors} error categories")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
