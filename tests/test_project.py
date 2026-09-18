"""Meaningful leakage/schema/inference integration checks; run python -m unittest discover -s tests."""

import unittest, json
import joblib, numpy as np, pandas as pd
from src.data import ROOT, CONFIG, load_data, features
from src.models import score


class ProjectChecks(unittest.TestCase):
    def test_saved_models_and_splits(self):
        for key in CONFIG:
            with self.subTest(dataset=key):
                df, a = load_data(key)
                x = features(df, key)
                p = ROOT / "artifacts" / key
                self.assertFalse(
                    set(x)
                    & {
                        "label",
                        "isFraud",
                        "isFraud - Copy",
                        "Fraud_Label",
                        "IsFraud",
                        "TransactionID",
                        "User_ID",
                        "newbalanceOrig",
                        "newbalanceDest",
                    }
                )
                manifest = pd.read_csv(p / "split_manifest.csv")
                self.assertEqual(manifest.row_id.nunique(), len(df))
                self.assertEqual(set(manifest.split), {"train", "validation", "test"})
                if key == "synthetic":
                    groups = [
                        set(
                            df.iloc[
                                manifest.loc[manifest.split.eq(s), "row_id"]
                            ].User_ID
                        )
                        for s in ["train", "validation", "test"]
                    ]
                    self.assertFalse(
                        groups[0] & groups[1]
                        or groups[0] & groups[2]
                        or groups[1] & groups[2]
                    )
                b = joblib.load(p / "best_model.joblib")
                test = pd.read_csv(p / "test_predictions.csv").head(30)
                actual = score(b["model"], features(test, key), b["kind"])
                np.testing.assert_allclose(
                    actual, test.risk_score, rtol=1e-6, atol=1e-9
                )
                changed = test.copy()
                changed[CONFIG[key]["target"]] = "altered_label"
                np.testing.assert_allclose(
                    actual, score(b["model"], features(changed, key), b["kind"])
                )
                with self.assertRaises(ValueError):
                    features(test.drop(columns=CONFIG[key]["amount"]), key)
                unknown = test.copy()
                unknown[CONFIG[key]["categorical"][0]] = "new_category"
                self.assertTrue(
                    np.isfinite(
                        score(b["model"], features(unknown, key), b["kind"])
                    ).all()
                )
                m = pd.read_csv(p / "model_metrics.csv")
                self.assertEqual(len(m), 13)
                counts = m[["test_tn", "test_fp", "test_fn", "test_tp"]].sum(axis=1)
                self.assertTrue((counts == (manifest.split == "test").sum()).all())


if __name__ == "__main__":
    unittest.main()
