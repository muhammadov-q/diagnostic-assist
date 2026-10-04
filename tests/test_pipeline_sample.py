"""End to end on the 22 sample cases: one assertion per trap in the sample."""

import json
import unittest
from pathlib import Path

from diagnostic_assist.domain import Case, LabelStatus
from diagnostic_assist.labeling.extractors import FirstAvailableExtractor, RecordedExtractor
from diagnostic_assist.labeling.pipeline import LabelingPipeline
from diagnostic_assist.labeling.quality import bcubed, check

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class SampleBackfillTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = [Case.from_dict(d) for d in json.loads((FIXTURES / "sample_cases.json").read_text())]
        cls.gold = json.loads((FIXTURES / "gold_causes.json").read_text())
        cls.result = LabelingPipeline(RecordedExtractor.from_file(FIXTURES / "extractions_sample.json")).backfill(cases)
        cls.by_id = {o.case.case_id: o for o in cls.result.outcomes}

    def cause(self, case_id):
        return self.by_id[case_id].cause_id

    def test_clustering_matches_hand_labels(self):
        predicted = {cid: o.cause_id for cid, o in self.by_id.items() if o.status is LabelStatus.TRUSTED}
        self.assertEqual(bcubed(predicted, self.gold["causes"]), (1.0, 1.0, 1.0))

    def test_same_cause_across_languages(self):
        self.assertEqual(self.cause("C-48211"), self.cause("C-48377"))  # contactor EN / DE
        self.assertEqual(self.cause("C-48899"), self.cause("C-49266"))  # valve plate EN / IT
        self.assertEqual(self.cause("C-49402"), self.cause("C-49480"))  # conrod bearing EN / DE

    def test_same_symptom_different_causes(self):
        dead_panel = {self.cause(c) for c in ("C-48211", "C-48590", "C-49040")}
        self.assertEqual(len(dead_panel), 3)  # contactor, ground strap, battery

    def test_label_comes_from_the_technician_not_the_customer(self):
        self.assertIn("drain", self.cause("C-48712"))  # customer said "oil"
        self.assertIn("alternator", self.cause("C-49002"))  # customer said "battery"

    def test_same_leak_different_repair(self):
        self.assertNotEqual(self.cause("C-48604"), self.cause("C-48755"))  # seal kit vs whole cylinder
        self.assertNotEqual(self.cause("C-48899"), self.cause("C-49437"))  # valve plate vs reed (CX-300)

    def test_cases_without_a_usable_cause(self):
        self.assertIs(self.by_id["C-49118"].status, LabelStatus.EXCLUDED)  # billing dispute
        self.assertIs(self.by_id["C-49203"].status, LabelStatus.EXCLUDED)  # warranty swap
        self.assertIsNone(self.cause("C-49203"))
        self.assertIs(self.by_id["C-49301"].status, LabelStatus.NO_FAULT_FOUND)

    def test_provisional_case_is_attached_but_not_counted(self):
        o = self.by_id["C-48801"]
        self.assertIs(o.status, LabelStatus.PROVISIONAL)
        cause = self.result.registry.causes[o.cause_id]
        self.assertEqual(cause.members, ["C-48604"])
        self.assertEqual(cause.provisional_members, ["C-48801"])

    def test_error_codes_normalised_across_spellings(self):
        for case_id in ("C-48899", "C-49266", "C-49437"):
            self.assertEqual(self.by_id[case_id].error_codes, ("E207",))

    def test_quality_checks_are_quiet_on_clean_data(self):
        report = check(self.result.registry, self.result.outcomes)
        self.assertEqual(report.findings, [])
        self.assertEqual(report.metrics["review_rate"], 0.0)

    def test_unchecked_quotes_are_reported_by_language(self):
        shares = check(self.result.registry, self.result.outcomes).metrics["trusted_component_not_checked_by_language"]
        self.assertEqual(shares["en"], 0.0)
        self.assertEqual({shares[lang] for lang in ("de", "fr", "it")}, {1.0})


class NightlyUpdateTest(unittest.TestCase):
    def test_new_case_in_another_language_joins_existing_cause_and_new_causes_become_proposals(self):
        cases = sorted(
            (Case.from_dict(d) for d in json.loads((FIXTURES / "sample_cases.json").read_text())),
            key=lambda c: c.created_at,
        )
        before = [c for c in cases if c.created_at < "2026-03-08"]
        after = [c for c in cases if c.created_at >= "2026-03-08"]
        pipeline = LabelingPipeline(RecordedExtractor.from_file(FIXTURES / "extractions_sample.json"))
        base = pipeline.backfill(before)
        update = pipeline.update(base.registry, after)

        by_id = {o.case.case_id: o for o in update.outcomes}
        self.assertEqual(by_id["C-49480"].cause_id, "air-compressor-cx/connecting-rod-bearing")
        proposals = {tuple(p.members) for p in update.proposals}
        self.assertIn(("C-49511", "C-49590"), proposals)
        self.assertTrue(all(p.status == "proposed" for p in update.proposals))

    def test_proposal_never_reuses_an_approved_cause_id(self):
        cases = [Case.from_dict(d) for d in json.loads((FIXTURES / "sample_cases.json").read_text())]
        new = Case("C-1", "Air Compressor CX", "CX-450", "2026-04-01", "en", "dead panel", "Main contactor burnt.", ("CONTACTOR-M2",), None)
        extraction = {
            "outcome_type": "fixed",
            "root_cause": {
                "evidence_quote": "Main contactor burnt.",
                "component": "main contactor",
                "location": None,
                "mechanism": "electrical_open",
                "certainty": "observed",
            },
            "parts": [{"part_number": "CONTACTOR-M2", "role": "fixes_cause"}],
            "customer_symptoms": [],
            "error_codes": [],
            "customer_report_contradicted": False,
        }
        recorded = RecordedExtractor.from_file(FIXTURES / "extractions_sample.json")
        pipeline = LabelingPipeline(FirstAvailableExtractor([recorded, RecordedExtractor({"C-1": extraction})]))
        base = pipeline.backfill(cases)
        (proposal,) = pipeline.update(base.registry, [new]).proposals
        self.assertEqual(proposal.members, ["C-1"])
        self.assertNotIn(proposal.cause_id, base.registry.causes)


if __name__ == "__main__":
    unittest.main()
