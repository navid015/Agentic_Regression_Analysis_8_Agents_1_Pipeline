"""Focused safeguards for model proposals and temporal feature encoding."""
import unittest

from crew.advisor import _parse_llm_actions, apply_proposals
from utils.preprocessing import _build_column_transformer


class ExperimentSafeguards(unittest.TestCase):
    def test_model_proposal_rejects_unknown_names(self):
        self.assertEqual(_parse_llm_actions(
            '{"actions":[{"action":"selected_models","value":["made_up"]}]}', {"x"}), [])
        actions = _parse_llm_actions(
            '{"actions":[{"action":"selected_models","value":["Ridge","SVR"]}]}', {"x"})
        self.assertEqual(apply_proposals({}, actions)["selected_models"], ["Ridge", "SVR"])

    def test_temporal_high_cardinality_encoding_uses_no_targets(self):
        transformer = _build_column_transformer([], [], ["store"],
            add_missing_indicators=False, interactions=False,
            high_cardinality_encoding="target", time_ordered=True, random_state=42)
        self.assertEqual(transformer.transformers[0][0], "freq")


if __name__ == "__main__":
    unittest.main()
