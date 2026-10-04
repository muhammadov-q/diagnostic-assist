"""Extractor request and response handling, against a fake client."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from diagnostic_assist.config import ExtractorConfig
from diagnostic_assist.domain import Case, Extraction, OutcomeType
from diagnostic_assist.labeling.extractors import (
    ExtractionError,
    FirstAvailableExtractor,
    LLMExtractor,
    MissingExtraction,
    RecordedExtractor,
    _build_request_params,
    _parse_message,
)

CASE = Case("C-1", "Air Compressor CX", "CX-450", "2026-01-01", "en", "dead panel", "Contactor coil open.", ("CONTACTOR-M1",), None)
GOOD = {
    "outcome_type": "fixed",
    "root_cause": {
        "evidence_quote": "Contactor coil open.",
        "component": "main contactor",
        "location": None,
        "mechanism": "electrical_open",
        "certainty": "observed",
    },
    "parts": [{"part_number": "CONTACTOR-M1", "role": "fixes_cause"}],
    "customer_symptoms": ["dead panel"],
    "error_codes": [],
    "customer_report_contradicted": False,
}


def message(text: str | None, stop_reason: str = "end_turn"):
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class RequestTest(unittest.TestCase):
    def test_request_uses_structured_output_and_caches_the_instructions(self):
        params = _build_request_params(CASE, ExtractorConfig(model="test-model"))
        self.assertEqual(params["model"], "test-model")
        self.assertEqual(params["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(params["output_config"]["effort"], "low")
        self.assertEqual(params["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertIn("Contactor coil open.", params["messages"][0]["content"])


class ParseTest(unittest.TestCase):
    def test_skips_thinking_block_and_parses_json(self):
        ex = _parse_message("C-1", message(json.dumps(GOOD)), "v")
        self.assertIs(ex.outcome_type, OutcomeType.FIXED)
        self.assertEqual(ex.root_cause.component, "main contactor")

    def test_refusal_and_truncation_become_extraction_errors(self):
        for stop in ("refusal", "max_tokens"):
            with self.subTest(stop=stop), self.assertRaises(ExtractionError):
                _parse_message("C-1", message(None, stop), "v")


class LLMExtractorTest(unittest.TestCase):
    def test_model_is_read_when_the_extractor_is_built(self):
        with mock.patch.dict(os.environ, {"EXTRACT_MODEL": "m"}):
            self.assertEqual(LLMExtractor(client=object()).version, "extract-v1/m")

    def test_missing_model_fails_before_any_call(self):
        with mock.patch.dict(os.environ, {"EXTRACT_MODEL": ""}), self.assertRaises(ValueError):
            LLMExtractor(client=object())


class RecordedAndChainTest(unittest.TestCase):
    def test_extraction_round_trips_through_its_stored_form(self):
        self.assertEqual(Extraction.from_dict(GOOD, "v").to_dict(), GOOD)

    def test_wrongly_shaped_record_is_an_extraction_error(self):
        with self.assertRaises(ExtractionError):
            RecordedExtractor({"C-1": {**GOOD, "root_cause": "main contactor"}}).extract(CASE)

    def test_chain_falls_through_only_on_missing_records(self):
        broken = RecordedExtractor({"C-1": {"outcome_type": "not-a-type"}})
        good = RecordedExtractor({"C-1": GOOD})
        self.assertEqual(FirstAvailableExtractor([RecordedExtractor({}), good]).extract(CASE).root_cause.component, "main contactor")
        with self.assertRaises(ExtractionError):  # a broken record must not be silently skipped
            FirstAvailableExtractor([broken, good]).extract(CASE)
        with self.assertRaises(MissingExtraction):
            FirstAvailableExtractor([RecordedExtractor({})]).extract(CASE)


if __name__ == "__main__":
    unittest.main()
