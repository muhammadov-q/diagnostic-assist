"""Validation against adversarial extractions built from the sample's traps."""

import unittest

from diagnostic_assist.domain import Case, Extraction, LabelStatus
from diagnostic_assist.labeling.labeler import apply_catalogue, build_outcome
from diagnostic_assist.labeling.normalize import PartCatalogue
from diagnostic_assist.labeling.validate import locate_quote, validate

OIL_PUDDLE = Case.from_dict(
    {
        "case_id": "C-48712",
        "equipment_family": "Water Chiller CH",
        "equipment_type": "CH-200",
        "created_at": "2026-02-02T11:30:00Z",
        "language": "en",
        "customer_description": "Customer reported puddle of oil under the machine.",
        "technician_notes": "No oil leak found. Fluid was condensate from the chiller drain pan, drain line was blocked. Cleared line.",
        "parts_replaced": [],
        "resolution_text": "No leak. Blocked condensate drain line, cleared.",
    }
)

WARRANTY_SWAP = Case.from_dict(
    {
        "case_id": "C-49203",
        "equipment_family": "Air Compressor CX",
        "equipment_type": "CX-450",
        "created_at": "2026-02-25T13:47:00Z",
        "language": "en",
        "customer_description": "Rattling noise from compressor head at high load.",
        "technician_notes": "Unit within warranty, swapped out complete unit per policy, did not strip down.",
        "parts_replaced": ["UNIT-CX450"],
        "resolution_text": "Replaced unit under warranty, works now.",
    }
)


def extraction(**overrides) -> Extraction:
    base = {
        "outcome_type": "fixed",
        "root_cause": {
            "evidence_quote": "drain line was blocked",
            "component": "condensate drain line",
            "location": None,
            "mechanism": "blockage",
            "certainty": "observed",
        },
        "parts": [],
        "customer_symptoms": ["puddle of oil under the machine"],
        "error_codes": [],
        "customer_report_contradicted": True,
    }
    base.update(overrides)
    return Extraction.from_dict(base, "test")


def codes(issues):
    return {i.code for i in issues}


class GroundingTest(unittest.TestCase):
    def test_cause_quoted_from_technician_passes(self):
        self.assertEqual(validate(OIL_PUDDLE, extraction()), [])

    def test_cause_taken_from_the_customer_is_rejected(self):
        ex = extraction(
            root_cause={
                "evidence_quote": "puddle of oil under the machine",
                "component": "oil seal",
                "location": None,
                "mechanism": "leakage",
                "certainty": "inferred",
            }
        )
        self.assertIn("evidence_from_customer", codes(validate(OIL_PUDDLE, ex)))

    def test_paraphrased_or_invented_quote_is_rejected(self):
        ex = extraction(
            root_cause={
                "evidence_quote": "the drain was clogged with debris",
                "component": "condensate drain line",
                "location": None,
                "mechanism": "blockage",
                "certainty": "observed",
            }
        )
        self.assertIn("ungrounded_evidence", codes(validate(OIL_PUDDLE, ex)))

    def test_negated_quote_is_not_evidence(self):
        ex = extraction(
            root_cause={
                "evidence_quote": "oil leak",
                "component": "oil seal",
                "location": None,
                "mechanism": "leakage",
                "certainty": "inferred",
            }
        )
        self.assertIn("negated_evidence", codes(validate(OIL_PUDDLE, ex)))

    def test_quote_matching_survives_transliteration(self):
        case = Case.from_dict({**OIL_PUDDLE.__dict__, "technician_notes": "Hauptschuetz defekt.", "parts_replaced": []})
        self.assertEqual(locate_quote("Hauptschütz defekt", case), "technician")


def notes(text: str, language: str = "en") -> Case:
    return Case.from_dict({**OIL_PUDDLE.__dict__, "technician_notes": text, "resolution_text": None, "language": language})


def cause(quote: str, component: str) -> Extraction:
    return extraction(
        root_cause={
            "evidence_quote": quote,
            "component": component,
            "location": None,
            "mechanism": "other",
            "certainty": "observed",
        }
    )


GROUND_STRAP = notes("Ground strap at frame was loose, heavy corrosion. Cleaned and retorqued. Contactor tested OK.")
ALTERNATOR = notes("Alternator output 12.1V at 1500rpm, not charging. Battery itself tested serviceable. Replaced alternator.")


class QuoteStatesFaultTest(unittest.TestCase):
    """A grounded quote is not enough: it has to say the named component was at fault."""

    def test_component_the_technician_cleared_is_not_evidence(self):
        self.assertIn("no_fault_in_quote", codes(validate(GROUND_STRAP, cause("Contactor tested OK.", "main contactor"))))
        self.assertIn("no_fault_in_quote", codes(validate(ALTERNATOR, cause("Battery itself tested serviceable.", "battery"))))
        self.assertEqual(validate(GROUND_STRAP, cause("Ground strap at frame was loose, heavy corrosion.", "frame ground strap")), [])

    def test_negation_inside_the_quote_is_not_evidence(self):
        self.assertIn("no_fault_in_quote", codes(validate(OIL_PUDDLE, cause("No oil leak found.", "compressor oil seal"))))

    def test_a_fault_stated_with_a_negation_still_counts(self):
        quote = "Alternator output 12.1V at 1500rpm, not charging."
        self.assertEqual(validate(ALTERNATOR, cause(quote, "alternator")), [])
        case = notes("No power to panel. Main contactor coil open.")
        self.assertEqual(validate(case, cause("No power to panel. Main contactor coil open.", "main contactor")), [])

    def test_negation_in_the_previous_sentence_does_not_count(self):
        self.assertEqual(validate(notes("Unit not starting. Contactor coil open."), cause("Contactor coil open.", "main contactor")), [])

    def test_quote_about_another_component_is_rejected(self):
        # The customer said "battery"; the technician's quote is about the alternator.
        quote = "Alternator output 12.1V at 1500rpm, not charging."
        self.assertIn("quote_component_mismatch", codes(validate(ALTERNATOR, cause(quote, "battery"))))

    def test_phone_shorthand_names_the_component(self):
        self.assertEqual(validate(notes("batt failed load test, replaced"), cause("batt failed load test", "battery")), [])

    def test_a_longer_word_is_not_the_component(self):
        # "sealant" is not a seal: C-48801's temporary fix says nothing about which seal failed.
        case = notes("temp fix w/ sealant, ordered seal kit")
        self.assertIn("quote_component_mismatch", codes(validate(case, cause("temp fix w/ sealant", "boom cylinder rod seal"))))
        self.assertEqual(validate(notes("Both rod seals weeping."), cause("Both rod seals weeping.", "boom cylinder rod seal")), [])

    def test_german_shorthand_for_ok_and_not_ok(self):
        self.assertIn("no_fault_in_quote", codes(validate(notes("Schütz i.O.", "de"), cause("Schütz i.O.", "main contactor"))))
        self.assertEqual(codes(validate(notes("Schütz n.i.O.", "de"), cause("Schütz n.i.O.", "main contactor"))), {"component_not_checked"})

    def test_non_english_component_is_noted_not_blocked(self):
        case, ex = notes("Hauptschuetz defekt.", "de"), cause("Hauptschütz defekt.", "main contactor")
        issues = validate(case, ex)
        self.assertEqual(codes(issues), {"component_not_checked"})
        self.assertIs(build_outcome(case, ex, issues).status, LabelStatus.TRUSTED)


class PartsAndOutcomeTest(unittest.TestCase):
    def test_invented_part_is_rejected(self):
        ex = extraction(parts=[{"part_number": "DRAIN-HOSE", "role": "fixes_cause"}])
        self.assertIn("invented_part", codes(validate(OIL_PUDDLE, ex)))

    def test_warranty_swap_dressed_up_as_a_diagnosis_goes_to_review(self):
        # The model claims a diagnosed fix and calls the whole unit the fixing part.
        ex = Extraction.from_dict(
            {
                "outcome_type": "fixed",
                "root_cause": {
                    "evidence_quote": "swapped out complete unit per policy",
                    "component": "compressor unit",
                    "location": None,
                    "mechanism": "other",
                    "certainty": "inferred",
                },
                "parts": [{"part_number": "UNIT-CX450", "role": "fixes_cause"}],
                "customer_symptoms": [],
                "error_codes": [],
                "customer_report_contradicted": False,
            },
            "test",
        )
        ex, overrides = apply_catalogue(ex, PartCatalogue())
        issues = overrides + validate(WARRANTY_SWAP, ex)
        self.assertIn("part_role_overridden", codes(issues))
        self.assertIn("whole_unit_swap_marked_fixed", codes(issues))
        self.assertIs(build_outcome(WARRANTY_SWAP, ex, issues).status, LabelStatus.REVIEW)

    def test_no_fault_found_must_not_carry_a_cause(self):
        self.assertIn("unexpected_cause", codes(validate(OIL_PUDDLE, extraction(outcome_type="no_fault_found"))))

    def test_correct_warranty_extraction_is_excluded_not_labelled(self):
        ex = Extraction.from_dict(
            {
                "outcome_type": "replaced_without_diagnosis",
                "root_cause": None,
                "parts": [{"part_number": "UNIT-CX450", "role": "whole_unit"}],
                "customer_symptoms": ["rattling noise"],
                "error_codes": [],
                "customer_report_contradicted": False,
            },
            "test",
        )
        outcome = build_outcome(WARRANTY_SWAP, ex, validate(WARRANTY_SWAP, ex))
        self.assertIs(outcome.status, LabelStatus.EXCLUDED)
        self.assertIsNone(outcome.cause_id)

    def test_missed_error_code_is_flagged_but_not_blocking(self):
        case = Case.from_dict({**OIL_PUDDLE.__dict__, "customer_description": "Puddle under machine, panel shows LP-01."})
        issues = validate(case, extraction())
        self.assertIn("missed_error_code", codes(issues))
        self.assertIs(build_outcome(case, extraction(), issues).status, LabelStatus.TRUSTED)


if __name__ == "__main__":
    unittest.main()
