
"""Unit tests for the Workforce Pell data validator."""

import sys
import unittest
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.feature_validation import (
    get_model_features,
    required_columns,
    validate,
)


FEATURES = ["IPEDSCOUNT1_LOG", "PCTPELL", "NO_ACS"]


def national_data():
    """Return a small national dataset with valid program identifiers."""
    return pd.DataFrame({
        "UNITID": [101, 102],
        "CIPCODE": [1101, 1101],
        "C150_L4": [0.8, 0.6],
        "IPEDSCOUNT1": [20, 30],
        "PCTPELL": [0.3, 0.4],
        "B19013_001E": [50000, None],
        "NO_ACS": [False, True],
        "A_MEDIAN": [45000, None],
        "NO_OEWS_WAGE": [False, True],
        "CONTROL_inst": [1, 2],
    })


def colorado_data():
    """Return two certificate lengths at the same institution and CIP."""
    return pd.DataFrame({
        "UNITID": [101, 101],
        "CIPCODE": [1101, 1101],
        "DEGREE": [
            "Certificate(<1 year)",
            "Certificate(>1 year)",
        ],
        "IPEDSCOUNT1": [20, 20],
        "PCTPELL": [0.3, 0.3],
        "B19013_001E": [50000, 50000],
        "NO_ACS": [False, False],
        "A_MEDIAN": [45000, 45000],
        "NO_OEWS_WAGE": [False, False],
        "CONTROL_inst": [1, 1],
        "EARN_MEDIAN": [40000, 55000],
        "BASELINE": [45000, 45000],
        "PLACEMENT": [0.4, 0.7],
        "LABEL": [0, 1],
    })


def names(findings):
    return {finding["check"] for finding in findings}


class FeatureValidationTests(unittest.TestCase):

    def test_feature_counts(self):
        self.assertEqual(
            len(get_model_features(ROOT / "notebooks/model_setup.py")),
            22,
        )
        self.assertEqual(
            len(get_model_features(ROOT / "notebooks/model_setup_colorado.py")),
            17,
        )

    def test_derived_feature_sources(self):
        required = required_columns(FEATURES, "national")
        self.assertIn("IPEDSCOUNT1", required)
        self.assertIn("CONTROL_inst", required)
        self.assertNotIn("IPEDSCOUNT1_LOG", required)

    def test_valid_national(self):
        self.assertEqual(
            validate(national_data(), "national", FEATURES), []
        )

    def test_valid_colorado(self):
        self.assertEqual(
            validate(colorado_data(), "colorado", FEATURES), []
        )

    def test_missing_column(self):
        df = national_data().drop(columns=["PCTPELL"])
        self.assertIn(
            "Missing column: PCTPELL",
            names(validate(df, "national", FEATURES)),
        )

    def test_invalid_numeric(self):
        df = national_data()
        df["PCTPELL"] = ["invalid", 0.4]
        self.assertIn(
            "Non-numeric: PCTPELL",
            names(validate(df, "national", FEATURES)),
        )

    def test_invalid_range(self):
        df = national_data()
        df.loc[0, "PCTPELL"] = 1.5
        self.assertIn(
            "Above maximum: PCTPELL",
            names(validate(df, "national", FEATURES)),
        )

    def test_missingness_mismatch(self):
        df = national_data()
        df.loc[1, "NO_ACS"] = False
        self.assertIn(
            "Flag mismatch: NO_ACS",
            names(validate(df, "national", FEATURES)),
        )

    def test_target_leakage(self):
        self.assertIn(
            "Excluded predictor: C150_L4",
            names(validate(
                national_data(),
                "national",
                FEATURES + ["C150_L4"],
            )),
        )

    def test_duplicate_national_key(self):
        df = national_data()
        df.loc[1, "UNITID"] = 101
        self.assertIn(
            "Duplicate national keys",
            names(validate(df, "national", FEATURES)),
        )

    def test_missing_national_id_warning(self):
        df = national_data()
        df.loc[1, "UNITID"] = None

        findings = validate(df, "national", FEATURES)
        warnings = [
            f for f in findings
            if f["check"] == "Missing national identifiers"
        ]

        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["severity"], "warning")

    def test_distinct_colorado_lengths(self):
        self.assertEqual(
            validate(colorado_data(), "colorado", FEATURES), []
        )

    def test_duplicate_colorado_key(self):
        df = colorado_data()
        df.loc[1, "DEGREE"] = df.loc[0, "DEGREE"]
        self.assertIn(
            "Duplicate Colorado keys",
            names(validate(df, "colorado", FEATURES)),
        )

    def test_colorado_label_mismatch(self):
        df = colorado_data()
        df.loc[1, "LABEL"] = 0
        self.assertIn(
            "Colorado label mismatch",
            names(validate(df, "colorado", FEATURES)),
        )

    def test_missing_label_input(self):
        df = colorado_data()
        df.loc[1, "EARN_MEDIAN"] = None
        self.assertIn(
            "Missing or invalid label inputs",
            names(validate(df, "colorado", FEATURES)),
        )

    def test_invalid_completion(self):
        df = national_data()
        df.loc[0, "C150_L4"] = 1.4
        self.assertIn(
            "Above maximum: C150_L4",
            names(validate(df, "national", FEATURES)),
        )

    def test_invalid_placement(self):
        df = colorado_data()
        df.loc[1, "PLACEMENT"] = 1.2
        self.assertIn(
            "Above maximum: PLACEMENT",
            names(validate(df, "colorado", FEATURES)),
        )

    def test_invalid_binary_label(self):
        df = colorado_data()
        df.loc[1, "LABEL"] = 2
        self.assertIn(
            "Invalid Colorado label",
            names(validate(df, "colorado", FEATURES)),
        )


if __name__ == "__main__":
    unittest.main()
