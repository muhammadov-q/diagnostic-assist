"""The example case files in examples/ are documentation: keep them honest."""

import json
import unittest
from pathlib import Path

from diagnostic_assist.domain import Case, LabelStatus
from diagnostic_assist.labeling.extractors import FirstAvailableExtractor, RecordedExtractor
from diagnostic_assist.labeling.pipeline import LabelingPipeline

ROOT = Path(__file__).resolve().parents[1]


class ExampleCasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus = [Case.from_dict(d) for d in json.loads((ROOT / "fixtures" / "sample_cases.json").read_text())]
        files = [json.loads(p.read_text()) for p in sorted((ROOT / "examples").glob("*.json"))]
        new_cases = [Case.from_dict(f["case"]) for f in files]
        extractor = FirstAvailableExtractor(
            [
                RecordedExtractor.from_file(ROOT / "fixtures" / "extractions_sample.json"),
                RecordedExtractor({f["case"]["case_id"]: f["extraction"] for f in files}),
            ]
        )
        pipeline = LabelingPipeline(extractor)
        cls.update = pipeline.update(pipeline.backfill(corpus).registry, new_cases)
        cls.by_id = {o.case.case_id: o for o in cls.update.outcomes}

    def test_french_cx300_contactor_joins_the_existing_contactor_cause(self):
        self.assertEqual(self.by_id["C-90001"].cause_id, "air-compressor-cx/main-contactor")

    def test_unseen_failure_becomes_a_proposal_not_a_forced_match(self):
        self.assertIsNone(self.by_id["C-90002"].cause_id)
        self.assertIn(["C-90002"], [p.members for p in self.update.proposals])

    def test_cause_taken_from_the_customer_is_held_for_review(self):
        o = self.by_id["C-90003"]
        self.assertIs(o.status, LabelStatus.REVIEW)
        self.assertIn("evidence_from_customer", [i.code for i in o.issues])


if __name__ == "__main__":
    unittest.main()
