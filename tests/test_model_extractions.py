"""Real model output, replayed: fixtures/extractions_model.json from scripts/record_extractions.py."""

import json
import unittest
from pathlib import Path

from diagnostic_assist.domain import Case, LabelStatus
from diagnostic_assist.labeling.extractors import RecordedExtractor
from diagnostic_assist.labeling.pipeline import LabelingPipeline
from diagnostic_assist.labeling.quality import bcubed

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class ModelExtractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = [Case.from_dict(d) for d in json.loads((FIXTURES / "sample_cases.json").read_text())]
        cls.gold = json.loads((FIXTURES / "gold_causes.json").read_text())
        cls.result = LabelingPipeline(RecordedExtractor.from_file(FIXTURES / "extractions_model.json")).backfill(cases)
        cls.by_id = {o.case.case_id: o for o in cls.result.outcomes}

    def test_every_case_was_extracted_and_passed_the_checks(self):
        self.assertEqual(len(self.by_id), 22)
        self.assertEqual([o.case.case_id for o in self.result.outcomes if o.status is LabelStatus.REVIEW], [])

    def test_grouping_matches_hand_labels(self):
        predicted = {cid: o.cause_id for cid, o in self.by_id.items() if o.status is LabelStatus.TRUSTED}
        self.assertEqual(bcubed(predicted, self.gold["causes"]), (1.0, 1.0, 1.0))

    def test_temporary_fix_without_a_stated_cause_is_excluded_not_guessed(self):
        # The one disagreement with the hand labels: "ordered seal kit" names the part for the
        # return visit, not what failed. The prompt allows both readings (submission, section 3).
        self.assertIs(self.by_id["C-48801"].status, LabelStatus.EXCLUDED)


if __name__ == "__main__":
    unittest.main()
