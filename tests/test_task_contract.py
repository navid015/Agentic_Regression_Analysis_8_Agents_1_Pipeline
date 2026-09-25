"""Prediction-time contract, target eligibility and label-free shift checks."""
import unittest

import numpy as np
import pandas as pd

from utils.modeling import filter_zoo_for_data, get_default_model_zoo
from utils.code_generator import generate_python_script
from utils.task_contract import RegressionTask, assess_target, prepare_task_frames, shift_report, apply_feature_contract, subgroup_error_report


class TaskContractTests(unittest.TestCase):
    def test_only_declared_features_and_safe_ratio(self):
        df = pd.DataFrame({"price": [4., 5.], "area": [2., 0.],
                           "future_invoice": [90, 100], "y": [1., 2.]})
        task = RegressionTask(available_features=("price", "area"),
                              ratio_features=(("price", "area"),))
        train, _ = prepare_task_frames(task, df, None, "y", time_column=None, group_column=None)
        self.assertNotIn("future_invoice", train)
        self.assertEqual(train["ratio__price__over__area"].iloc[0], 2.)
        self.assertTrue(np.isnan(train["ratio__price__over__area"].iloc[1]))
        raw_prediction = df[["price", "area", "future_invoice"]]
        inference = apply_feature_contract(task, raw_prediction, ["price", "area"])
        self.assertEqual(list(inference), ["price", "area", "ratio__price__over__area"])
        self.assertEqual(inference.iloc[0, -1], 2.)

    def test_target_type_and_models(self):
        positive = assess_target(np.linspace(.2, 100, 90), RegressionTask(target_kind="positive"))
        self.assertIn("GammaRegressor", positive.candidate_models)
        self.assertNotIn("PoissonRegressor", positive.candidate_models)
        zoo = get_default_model_zoo()
        kept, skipped = filter_zoo_for_data(zoo, np.array([0., 1., 2.]))
        self.assertNotIn("GammaRegressor", kept)
        self.assertIn("GammaRegressor", skipped)
        self.assertIn("PoissonRegressor", kept)

    def test_unseen_categories_report_does_not_require_test_target(self):
        tr = pd.DataFrame({"region": ["LA"] * 30, "y": np.arange(30)})
        te = pd.DataFrame({"region": ["TX"] * 10, "y": [float("nan")] * 10})
        self.assertEqual(shift_report(tr, te, "y")["region"]["kind"], "unseen_categories")

    def test_final_period_diagnostic(self):
        report = subgroup_error_report(np.arange(30), np.arange(30) + 2, time_ordered=True)
        self.assertEqual(report["later_test_period"]["mae"], 2.0)
        self.assertEqual(report["earlier_test_period"]["rows"], 15)

    def test_exported_script_replays_task_contract(self):
        script = generate_python_script(target="y", model_names=["Ridge"],
            task_spec={"available_features": ("x",), "ratio_features": ()},
            reserve_final_test=True)
        compile(script, "generated_regression.py", "exec")
        self.assertIn("df_train, df_test = reserve_test(", script)
        self.assertIn("df_train, df_test = prepare_task_frames(", script)

    def test_invalid_prediction_contract_is_rejected(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with self.assertRaises(ValueError):
            prepare_task_frames(RegressionTask(available_features=("y",)), df, None, "y",
                                time_column=None, group_column=None)
        with self.assertRaises(ValueError):
            assess_target(np.arange(40) - 5, RegressionTask(target_kind="count"))


if __name__ == "__main__":
    unittest.main()
