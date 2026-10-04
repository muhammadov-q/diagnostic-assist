import unittest
from dataclasses import replace

from diagnostic_assist.config import LabelingConfig
from diagnostic_assist.domain import Case, Extraction, LabeledOutcome, LabelStatus
from diagnostic_assist.labeling.canonicalize import CauseClusterer, apply_parts, cause_table
from diagnostic_assist.labeling.embeddings import LexicalEmbedder


def outcome(case_id: str, component: str, parts: set[str], status=LabelStatus.TRUSTED) -> LabeledOutcome:
    case = Case(case_id, "Aerial Platform AP", "AP-120", "2026-01-01", "en", "leak", "notes", tuple(parts), None)
    extraction = Extraction.from_dict(
        {
            "outcome_type": "fixed",
            "root_cause": {
                "evidence_quote": "notes",
                "component": component,
                "location": None,
                "mechanism": "leakage",
                "certainty": "observed",
            },
            "parts": [{"part_number": p, "role": "fixes_cause"} for p in parts],
            "customer_symptoms": [],
            "error_codes": [],
            "customer_report_contradicted": False,
        },
        "test",
    )
    return LabeledOutcome(case, extraction, status, "test", cause_part_families=frozenset(parts))


class PartsRuleTest(unittest.TestCase):
    def test_disjoint_parts_forbid_a_merge_whatever_the_text_says(self):
        self.assertEqual(apply_parts(0.0, frozenset({"A"}), frozenset({"B"}), 0.5), 1.0)

    def test_shared_parts_shrink_distance_and_missing_parts_are_neutral(self):
        self.assertAlmostEqual(apply_parts(0.4, frozenset({"A"}), frozenset({"A"}), 0.5), 0.2)
        self.assertEqual(apply_parts(0.4, frozenset(), frozenset({"A"}), 0.5), 0.4)


class ClusteringTest(unittest.TestCase):
    def setUp(self):
        self.clusterer = CauseClusterer(LexicalEmbedder(), LabelingConfig())

    def test_seal_kit_and_whole_cylinder_stay_separate_even_with_a_loose_threshold(self):
        loose = CauseClusterer(LexicalEmbedder(), replace(LabelingConfig(), merge_threshold=0.95))
        causes = loose.cluster(
            [outcome("1", "boom cylinder rod seal", {"SEALKIT-BC"}), outcome("2", "boom cylinder", {"CYL-BOOM"})]
        )
        self.assertEqual(len(causes), 2)

    def test_same_cause_in_different_words_merges(self):
        causes = self.clusterer.cluster(
            [
                outcome("1", "condenser water pump mechanical seal", {"SEAL-PMP-CH"}),
                outcome("2", "condenser pump seal", {"SEAL-PMP-CH"}),
                outcome("3", "Condenser pump seal.", set()),
            ]
        )
        self.assertEqual([sorted(c.members) for c in causes], [["1", "2", "3"]])

    def test_head_noun_decides_which_cause_a_provisional_case_joins(self):
        registry = self.clusterer.build_registry(
            [outcome("1", "boom cylinder rod seal", {"SEALKIT-BC"}), outcome("2", "boom cylinder", {"CYL-BOOM"})]
        )
        assignment = registry.assign(outcome("3", "boom cylinder seal", set(), LabelStatus.PROVISIONAL))
        self.assertEqual(registry.causes[assignment.cause_id].members, ["1"])

    def test_unrelated_new_case_goes_to_the_pool_instead_of_the_nearest_cause(self):
        registry = self.clusterer.build_registry([outcome("1", "boom cylinder rod seal", {"SEALKIT-BC"})])
        assignment = registry.assign(outcome("2", "platform control joystick", {"JOY-AP"}))
        self.assertIsNone(assignment.cause_id)

    def test_a_part_less_wording_cannot_bypass_different_parts_at_assignment(self):
        registry = self.clusterer.build_registry(
            [outcome("1", "condenser pump seal", {"SEAL-PMP"}), outcome("2", "condenser pump seal", set())]
        )
        self.assertIsNone(registry.assign(outcome("3", "condenser pump seal", {"PUMP-ASSY"})).cause_id)

    def test_part_probability_counts_each_case_once(self):
        o = outcome("1", "connecting rod bearing", {"BEARING-CR"})
        o = replace(o, extraction=replace(o.extraction, parts=o.extraction.parts * 2))
        registry = self.clusterer.build_registry([o])
        self.assertEqual(cause_table(registry, [o])[0]["p_part"], {"BEARING-CR": 1.0})


if __name__ == "__main__":
    unittest.main()
