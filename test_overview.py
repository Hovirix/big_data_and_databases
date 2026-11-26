import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from overview import create_overview


class OverviewTest(unittest.TestCase):
    def test_overview_preserves_source_and_reports_blanks(self):
        source = Path("data/Telco-Customer-Churn.csv")
        original = hashlib.sha256(source.read_bytes()).digest()
        with TemporaryDirectory() as directory:
            output = Path(directory)
            quality = create_overview(source, output)
            self.assertEqual((quality["rows"], quality["columns"]), (7043, 21))
            self.assertEqual(quality["blank_total_charges"], 11)
            self.assertEqual(quality["missing_after_conversion"]["TotalCharges"], 11)
            self.assertEqual(quality["duplicate_customer_ids"], 0)
            self.assertEqual(json.loads((output / "quality.json").read_text()), quality)
            categories = pd.read_csv(output / "categorical_churn.csv")
            contract = categories[categories["column"].eq("Contract")]
            self.assertEqual(contract["customers"].sum(), 7043)
            self.assertEqual(contract["churned"].sum(), 1869)
            self.assertNotIn("customerID", categories["column"].values)
            self.assertIn("SeniorCitizen", categories["column"].values)
            dictionary = pd.read_csv(output / "data_dictionary.csv")
            self.assertEqual(len(dictionary), 21)
            self.assertEqual(
                dictionary.set_index("column").loc["MonthlyCharges", "group"],
                "Subscription",
            )
            report = (output / "overview.md").read_text()
            self.assertIn("5,174 (73.46%)", report)
            self.assertIn("1,869 (26.54%)", report)
            self.assertIn("19 candidate predictors", report)
            for name in dictionary.loc[
                dictionary["group"].isin(["Demographics", "Subscription", "Services"]), "column"
            ]:
                self.assertIn(f"`{name}`", report)
            groups = pd.read_csv(output / "group_churn.csv")
            for _, table in groups.groupby("column"):
                self.assertEqual(table["customers"].sum(), 7043)
                self.assertEqual(table["churned"].sum(), 1869)
                pd.testing.assert_series_equal(
                    table["churn_percentage"],
                    table["churned"] / table["customers"] * 100,
                    check_names=False,
                )
            pd.testing.assert_series_equal(
                groups["difference_pp"],
                groups["churn_percentage"] - 1869 / 7043 * 100,
                check_names=False,
            )
            tenure = groups[groups["column"].eq("Tenure band (months)")]
            self.assertEqual(tenure["customers"].tolist(), [2186, 1024, 1594, 2239])
            settings = json.loads((output / "group_settings.json").read_text())
            charges = settings["monthly_charges"]
            self.assertEqual(charges["edges"], [18.25, 35.5, 70.35, 89.85, 118.75])
            boundaries = pd.cut(
                pd.Series(charges["edges"]), bins=charges["edges"],
                labels=charges["labels"], include_lowest=True,
                right=charges["right_closed"],
            )
            self.assertEqual(boundaries.tolist(), ["Q1", "Q1", "Q2", "Q3", "Q4"])
            tenure_bins = settings["tenure"]
            boundaries = pd.cut(
                pd.Series([0, 12, 13, 24, 25, 48, 49, 72]),
                bins=tenure_bins["edges"], labels=tenure_bins["labels"],
                right=tenure_bins["right_closed"],
            )
            self.assertEqual(
                boundaries.tolist(),
                ["0–12", "0–12", "13–24", "13–24", "25–48", "25–48", "49–72", "49–72"],
            )
            patterns = (output / "churn_patterns.md").read_text()
            self.assertIn("42.71%", patterns)
            self.assertLess(patterns.index("## Contract"), patterns.index("## gender"))
        self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), original)

    def test_unexpected_charges_are_not_silently_treated_as_blanks(self):
        df = pd.read_csv("data/Telco-Customer-Churn.csv").head(3)
        df.loc[0, "TotalCharges"] = "invalid"
        with TemporaryDirectory() as directory:
            source = Path(directory) / "invalid.csv"
            df.to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "non-numeric TotalCharges"):
                create_overview(source, Path(directory) / "output")
