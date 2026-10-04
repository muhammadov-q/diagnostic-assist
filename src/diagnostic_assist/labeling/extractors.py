"""LLM extraction: one closed case in, one ``Extraction`` out.

The SDK is imported lazily, so the pipeline and the tests run without it. ``LLMExtractor``
has not been run against the live API (no credentials); the tests use recorded extractions.

Stub, not built: the backfill sends the same request through the provider's batch API
(half price) in chunks of 10k cases, so a failed or expired batch is cheap to resubmit.
Results come back in any order and are keyed by case ID; a failed request becomes an
``ExtractionError`` for that case, not for the batch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from ..config import ExtractorConfig
from ..domain import Case, Extraction
from .prompt import EXTRACTION_SCHEMA, SYSTEM_PROMPT, render_case


class ExtractionError(Exception):
    """The case could not be extracted; it goes to review."""


class MissingExtraction(ExtractionError):
    """No stored extraction for this case (as opposed to a broken one)."""


class Extractor(Protocol):
    version: str

    def extract(self, case: Case) -> Extraction: ...


def _build_request_params(case: Case, config: ExtractorConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "max_tokens": config.max_tokens,
        # Identical for every case, so cached.
        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "output_config": {
            "effort": config.effort,
            "format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA},
        },
        "messages": [{"role": "user", "content": render_case(case)}],
    }


def _parse_message(case_id: str, message: Any, version: str) -> Extraction:
    if message.stop_reason == "refusal":
        raise ExtractionError(f"{case_id}: model declined the request")
    if message.stop_reason == "max_tokens":
        raise ExtractionError(f"{case_id}: output truncated at max_tokens")
    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        raise ExtractionError(f"{case_id}: no text block in response")
    try:
        return Extraction.from_dict(json.loads(text), version)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ExtractionError(f"{case_id}: unparseable extraction: {exc}") from exc


def _require_model(config: ExtractorConfig) -> None:
    if not config.model:
        raise ValueError("no extraction model configured: set EXTRACT_MODEL to the provider's model ID")


class LLMExtractor:
    """One synchronous call per case.

    ``client`` is the SDK client: ``anthropic.AnthropicAWS()`` in production (AWS billing and
    IAM), ``anthropic.Anthropic()`` for a local run.
    """

    def __init__(self, client: Any, config: ExtractorConfig | None = None):
        config = config if config is not None else ExtractorConfig()
        _require_model(config)
        self._client = client
        self._config = config
        self.version = f"{config.version}/{config.model}"

    def extract(self, case: Case) -> Extraction:
        import anthropic

        try:
            # The SDK already retries 429/5xx/connection errors with backoff.
            message = self._client.messages.create(**_build_request_params(case, self._config))
        except anthropic.BadRequestError as exc:
            raise ExtractionError(f"{case.case_id}: request rejected: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise ExtractionError(f"{case.case_id}: API error {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            raise ExtractionError(f"{case.case_id}: connection failed") from exc
        return _parse_message(case.case_id, message, self.version)


class RecordedExtractor:
    def __init__(self, records: dict[str, dict[str, Any]], version: str = "recorded"):
        self._records = dict(records)
        self.version = version

    @classmethod
    def from_file(cls, path: str | Path, version: str = "recorded") -> RecordedExtractor:
        """A JSON object {case_id: extraction}, each in the shape the LLM returns."""
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), version)

    def extract(self, case: Case) -> Extraction:
        raw = self._records.get(case.case_id)
        if raw is None:
            raise MissingExtraction(f"{case.case_id}: no recorded extraction")
        try:
            return Extraction.from_dict(raw, self.version)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ExtractionError(f"{case.case_id}: malformed recorded extraction: {exc}") from exc


class FirstAvailableExtractor:
    """Try extractors in order. Only a missing record falls through; a broken one raises."""

    def __init__(self, extractors: list[Extractor]):
        self._extractors = extractors
        self.version = "+".join(e.version for e in extractors)

    def extract(self, case: Case) -> Extraction:
        for extractor in self._extractors:
            try:
                return extractor.extract(case)
            except MissingExtraction:
                continue
        raise MissingExtraction(f"{case.case_id}: no extractor has this case")
