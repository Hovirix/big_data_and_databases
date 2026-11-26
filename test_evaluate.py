import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from evaluate import evaluate, validate_results
from prepare import prepare
from train import model_specs, tune


class EvaluationTest(unittest.TestCase):
    def test_final_outputs_fixed_settings_and_no_overwrite(self):
        source = Path("data/Telco-Customer-Churn.csv")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with TemporaryDirectory() as directory:
            preparation = Path(directory)
            prepare(source, preparation)
            specs = model_specs(42)
            specs["random_forest"][0].set_params(model__n_estimators=3)
            specs = {
                name: (pipeline, {key: [values[0]] for key, values in grid.items()})
                for name, (pipeline, grid) in specs.items()
            }
            with patch("train.model_specs", return_value=specs):
                tune(source, preparation)
            selected_path = preparation / "tuning/selected_settings.json"
            selected_hash = hashlib.sha256(selected_path.read_bytes()).hexdigest()
            scores = evaluate(source, preparation).set_index("model")
            output = preparation / "final"
            predictions = pd.read_csv(output / "test_predictions.csv")
            split = pd.read_csv(preparation / "split_ids.csv")
            test_ids = split.loc[split["partition"].eq("test"), "customerID"].tolist()
            matrices = json.loads((output / "confusion_matrices.json").read_text())["models"]
            for name, rows in predictions.groupby("model"):
                self.assertEqual(rows["customerID"].tolist(), test_ids)
                self.assertEqual(len(rows), 1409)
                self.assertTrue(rows["churn_probability"].between(0, 1).all())
                self.assertEqual(
                    rows["prediction"].tolist(), rows["churn_probability"].gt(0.5).astype(int).tolist(),
                )
                for metric, expected in {
                    "roc_auc": roc_auc_score(rows.actual, rows.churn_probability),
                    "precision": precision_score(rows.actual, rows.prediction, zero_division=0),
                    "recall": recall_score(rows.actual, rows.prediction),
                    "f1": f1_score(rows.actual, rows.prediction),
                }.items():
                    self.assertAlmostEqual(scores.loc[name, metric], expected)
                self.assertEqual(matrices[name], confusion_matrix(rows.actual, rows.prediction).tolist())
            self.assertEqual(scores.loc["majority_baseline", "roc_auc"], 0.5)
            importance = pd.read_csv(output / "random_forest_permutation_importance.csv")
            self.assertEqual(len(importance), 19)
            self.assertNotIn("customerID", importance["predictor"].tolist())
            contrasts = pd.read_csv(output / "logistic_contrasts.csv")
            self.assertTrue(contrasts["odds_ratio"].gt(0).all())
            self.assertIn("versus", " ".join(contrasts["comparison"]))
            for filename in ["roc_auc.svg", "precision.svg", "recall.svg", "f1.svg", "importance.svg"]:
                ElementTree.parse(output / filename)
            frozen = json.loads((output / "frozen_settings.json").read_text())
            self.assertEqual(frozen["selected_settings_sha256"], selected_hash)
            for name, expected in frozen["code_sha256"].items():
                archived = output / frozen["source_archive"] / name
                self.assertEqual(archived.read_bytes(), Path(name).read_bytes())
                self.assertEqual(hashlib.sha256(archived.read_bytes()).hexdigest(), expected)
            self.assertTrue((output / "evaluation.md").exists())
            pd.testing.assert_frame_equal(
                validate_results(source, preparation).set_index("model"), scores,
                check_exact=False,
            )
            with self.assertRaises(FileExistsError):
                evaluate(source, preparation)
            self.assertEqual(hashlib.sha256(selected_path.read_bytes()).hexdigest(), selected_hash)
            archived = output / frozen["source_archive"] / "evaluate.py"
            original = archived.read_bytes()
            archived.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Archived source changed"):
                validate_results(source, preparation)
            archived.write_bytes(original)
            predictions.loc[0, "churn_probability"] = 1.5
            predictions.to_csv(output / "test_predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "probabilities"):
                validate_results(source, preparation)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
