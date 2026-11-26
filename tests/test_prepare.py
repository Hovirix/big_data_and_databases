import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.prepare import CATEGORICAL, NUMERIC, make_preprocessor, prepare


class PreparationTest(unittest.TestCase):
    def test_frozen_evaluation_blocks_upstream_writes(self):
        from src.train import tune

        for marker in ["frozen_settings.json", "test_metrics.csv"]:
            with self.subTest(marker=marker), TemporaryDirectory() as directory:
                output = Path(directory)
                (output / "final").mkdir()
                (output / "final" / marker).write_text("historical result")
                (output / "settings.json").write_text("historical settings")
                before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
                for run in [prepare, tune]:
                    with self.assertRaises(FileExistsError):
                        run(Path("missing.csv"), output)
                    self.assertEqual(before, {
                        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
                    })

    def test_split_and_training_only_baseline(self):
        source = Path("data/Telco-Customer-Churn.csv")
        digest = hashlib.sha256(source.read_bytes()).digest()
        with TemporaryDirectory() as directory:
            output = Path(directory)
            settings, metrics = prepare(source, output)
            split = pd.read_csv(output / "split_ids.csv")
            train = split[split["partition"].eq("train")]
            test = split[split["partition"].eq("test")]
            self.assertEqual((len(train), len(test)), (5634, 1409))
            self.assertEqual(split["customerID"].nunique(), 7043)
            self.assertTrue(set(train["customerID"]).isdisjoint(test["customerID"]))
            self.assertAlmostEqual(
                train["Churn"].eq("Yes").mean(), test["Churn"].eq("Yes").mean(), places=3,
            )
            oof = pd.read_csv(output / "baseline_cv_predictions.csv")
            self.assertEqual(set(oof["customerID"]), set(train["customerID"]))
            self.assertEqual(oof["customerID"].nunique(), len(train))
            self.assertEqual(set(oof["fold"]), {1, 2, 3, 4, 5})
            self.assertTrue(oof["prediction"].eq(0).all())
            self.assertEqual(metrics["roc_auc"], 0.5)
            self.assertEqual(metrics["recall"], 0)
            raw = pd.read_csv(source).set_index("customerID").loc[train["customerID"]]
            expected = pd.to_numeric(raw["TotalCharges"], errors="coerce").median()
            self.assertEqual(settings["training_preprocessing"]["medians"]["TotalCharges"], expected)
            again, _ = prepare(source, output)
            self.assertEqual(settings, again)
            pd.testing.assert_frame_equal(split, pd.read_csv(output / "split_ids.csv"))
        self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), digest)

    def test_preprocessing_learns_only_fit_data_and_handles_unknowns(self):
        train = pd.DataFrame({name: [1.0, 3.0] for name in NUMERIC})
        for name in CATEGORICAL:
            train[name] = [0, 1] if name == "SeniorCitizen" else ["No", "Yes"]
        test = train.iloc[[0]].copy()
        test["TotalCharges"] = float("nan")
        test["Contract"] = "unseen"
        test["customerID"] = "not-a-feature"
        scaled = make_preprocessor().fit(train)
        transformed = scaled.transform(test)
        self.assertEqual(scaled.named_transformers_["numeric"]["impute"].statistics_.tolist(), [2, 2, 2])
        self.assertFalse(pd.isna(transformed).any())
        self.assertFalse(any("customerID" in name for name in scaled.get_feature_names_out()))
        unscaled = make_preprocessor(False).fit(train)
        self.assertNotIn("scale", unscaled.named_transformers_["numeric"].named_steps)
