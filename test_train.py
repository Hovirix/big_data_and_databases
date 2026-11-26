import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from prepare import prepare
from train import load_training, model_specs, tune


class TrainingTest(unittest.TestCase):
    def test_training_only_tuning_and_saved_fold_validation(self):
        source = Path("data/Telco-Customer-Churn.csv")
        with TemporaryDirectory() as directory:
            output = Path(directory)
            prepare(source, output)
            inputs = [source, output / "split_ids.csv", output / "settings.json"]
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs]
            X, y, folds, baseline, _ = load_training(source, output)
            self.assertEqual(X.shape, (5634, 19))
            self.assertNotIn("customerID", X.columns)
            self.assertIn("tenure", X.columns)
            for fold, (fit, validation) in enumerate(folds, start=1):
                self.assertTrue(set(fit).isdisjoint(validation))
                self.assertEqual(len(fit) + len(validation), len(y))
                self.assertTrue(baseline.iloc[validation]["fold"].eq(fold).all())
            specs = model_specs(42)
            self.assertEqual(specs["logistic_regression"][1]["model__C"], [0.1, 1, 10])
            self.assertEqual(specs["random_forest"][0]["model"].n_estimators, 300)
            # Smoke-test real pipeline/CV paths with smaller grids and fewer trees.
            specs["random_forest"][0].set_params(model__n_estimators=3)
            small = {
                name: (pipeline, {key: [values[0]] for key, values in grid.items()})
                for name, (pipeline, grid) in specs.items()
            }
            with patch("train.model_specs", return_value=small):
                comparison = tune(source, output)
            self.assertEqual(len(comparison), 3)
            self.assertTrue(comparison["mean_cv_roc_auc"].between(0, 1).all())
            self.assertEqual(comparison.iloc[0]["mean_cv_roc_auc"], 0.5)
            settings = json.loads((output / "tuning/selected_settings.json").read_text())
            self.assertEqual(settings["classification_threshold"], 0.5)
            self.assertEqual(len(settings["models"]), 2)
            for name in small:
                results = pd.read_csv(output / f"tuning/{name}_cv_results.csv")
                self.assertEqual(len(results), 1)
                self.assertIn("split4_test_roc_auc", results.columns)
            self.assertEqual(hashes, [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs])
            baseline.loc[0, "fold"] = 99
            baseline.to_csv(output / "baseline_cv_predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "folds"):
                load_training(source, output)
