"""Isolation and proposal permissions for the eight-specialist laboratory."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from crew.laboratory import SPECIALISTS, reserve_test, run_eight_agent_pipeline, valid_proposal


class EightAgentMode(unittest.TestCase):
    def test_all_eight_run_before_locked_test_is_used(self):
        data = pd.DataFrame({"x": np.arange(70), "y": np.arange(70) * 2.0})
        calls, prompts = [], []

        def fake_run(train, val, test, target, config, progress):
            calls.append((len(train), test is not None))
            return SimpleNamespace(
                selection_metric="rmse", selection_basis="cv", best_model="Ridge", run_options=dict(config),
                profile={"n_rows": len(train), "n_cols": 2, "missing_total": 0},
                results={"Ridge": SimpleNamespace(metrics={"CV_RMSE_mean": 1.0}, y_pred_test=np.zeros(len(test) if test is not None else 14))},
                refinement_log=[], run_inputs={}, warnings=[],
                preprocessing=SimpleNamespace(y_test_original=np.zeros(len(test) if test is not None else 14)))

        def propose(agent, prompt):
            prompts.append((agent.name, prompt))
            return '{"actions":[]}'

        with patch("crew.orchestrator._run_once", side_effect=fake_run):
            result = run_eight_agent_pipeline(df_train=data, target="y", proposer=propose,
                test_size=.2, random_state=4, split_strategy="random", cv_strategy="kfold")
        self.assertEqual(len(prompts), 8)
        self.assertEqual([name for name, _ in prompts], [s.name for s in SPECIALISTS])
        self.assertEqual(calls, [(56, False), (56, True)])
        self.assertEqual(len(result.refinement_log), 8)
        for _, prompt in prompts:
            self.assertNotIn("Final test", prompt)

    def test_accepted_proposal_reaches_final_run(self):
        data = pd.DataFrame({"x": np.arange(70), "y": np.arange(70) * 2.0})
        calls = []

        def fake_run(train, val, test, target, config, progress):
            calls.append((test is not None, config.get("add_interactions", False)))
            return SimpleNamespace(selection_metric="rmse", selection_basis="cv",
                best_model="Ridge", run_options=dict(config),
                profile={"n_rows": len(train), "n_cols": 2, "missing_total": 0},
                results={"Ridge": SimpleNamespace(metrics={"CV_RMSE_mean": 1.0}, y_pred_test=np.zeros(len(test) if test is not None else 14))},
                refinement_log=[], run_inputs={}, warnings=[],
                preprocessing=SimpleNamespace(y_test_original=np.zeros(len(test) if test is not None else 14)))

        def propose(agent, prompt):
            if agent.name == "features":
                return '{"actions":[{"action":"add_interactions","value":true,"reason":"nonlinearity"}]}'
            return '{"actions":[]}'

        with patch("crew.orchestrator._run_once", side_effect=fake_run), \
             patch("crew.laboratory._improves", return_value=(True, "paired CV gain")):
            out = run_eight_agent_pipeline(df_train=data, target="y", proposer=propose,
                test_size=.2, random_state=4, split_strategy="random", cv_strategy="kfold")
        self.assertEqual(calls, [(False, False), (False, True), (True, True)])
        self.assertTrue(out.refinement_log[2]["accepted"])

    def test_roles_cannot_select_another_family_or_change_splits(self):
        linear = next(s for s in SPECIALISTS if s.name == "linear")
        self.assertIsNone(valid_proposal(json.dumps({"actions": [
            {"action": "selected_models", "value": ["XGBoost"]}]}), linear, {"x"}))
        self.assertIsNone(valid_proposal(json.dumps({"actions": [
            {"action": "cv_strategy", "value": "kfold"}]}), linear, {"x"}))
        self.assertIsNotNone(valid_proposal(json.dumps({"actions": [
            {"action": "selected_models", "value": ["Ridge"]}]}), linear, {"x"}))

    def test_reservation_respects_time_and_groups(self):
        frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=80),
                              "group": np.repeat(np.arange(20), 4), "y": np.arange(80)})
        dev, test = reserve_test(frame, test_size=.2, seed=42, split="time",
                                 time_column="date", group_column=None)
        self.assertLess(dev.date.max(), test.date.min())
        dev, test = reserve_test(frame, test_size=.2, seed=42, split="random",
                                 time_column=None, group_column="group")
        self.assertFalse(set(dev.group) & set(test.group))


if __name__ == "__main__":
    unittest.main()
