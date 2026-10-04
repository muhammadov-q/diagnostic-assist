"""Wires the stages together.

``backfill`` runs once over the history and builds the cause list. ``update`` runs nightly:
it assigns new cases and turns the misses into proposed causes. Orchestration and
persistence live outside this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..config import LabelingConfig
from ..domain import Case, Issue, LabeledOutcome, LabelStatus
from .canonicalize import Assignment, Cause, CauseClusterer, CauseRegistry, unique_id
from .embeddings import Embedder, LexicalEmbedder
from .extractors import ExtractionError, Extractor
from .labeler import apply_catalogue, build_outcome
from .normalize import PartCatalogue, equipment_prefixes
from .triage import skip_reason
from .validate import validate


@dataclass
class RunResult:
    outcomes: list[LabeledOutcome]
    registry: CauseRegistry
    unassigned: list[tuple[LabeledOutcome, Assignment]] = field(default_factory=list)
    proposals: list[Cause] = field(default_factory=list)


class LabelingPipeline:
    def __init__(
        self,
        extractor: Extractor,
        embedder: Embedder | None = None,
        catalogue: PartCatalogue | None = None,
        config: LabelingConfig = LabelingConfig(),
    ):
        self._extractor = extractor
        self._embedder = embedder or LexicalEmbedder()
        self._catalogue = catalogue or PartCatalogue()
        self._config = config
        self._clusterer = CauseClusterer(self._embedder, config)
        # Model-name prefixes ("CX" from "CX-450") are never error codes.
        self._equipment_types: set[str] = set()

    def label(self, cases: Iterable[Case]) -> list[LabeledOutcome]:
        cases = list(cases)
        self._equipment_types |= {c.equipment_type for c in cases}
        prefixes = equipment_prefixes(self._equipment_types)
        outcomes: list[LabeledOutcome] = []
        for case in cases:
            if reason := skip_reason(case):
                outcomes.append(LabeledOutcome(case, None, LabelStatus.EXCLUDED, reason))
                continue
            try:
                extraction = self._extractor.extract(case)
            except ExtractionError as exc:
                issue = Issue("extraction_failed", str(exc))
                outcomes.append(LabeledOutcome(case, None, LabelStatus.REVIEW, str(exc), issues=(issue,)))
                continue
            extraction, overrides = apply_catalogue(extraction, self._catalogue)
            issues = overrides + validate(case, extraction, prefixes)
            outcomes.append(build_outcome(case, extraction, issues))
        return outcomes

    def backfill(self, cases: Iterable[Case]) -> RunResult:
        outcomes = self.label(cases)
        registry = self._clusterer.build_registry(o for o in outcomes if o.status is LabelStatus.TRUSTED)
        by_id = {o.case.case_id: o for o in outcomes}
        for cause in registry.causes.values():
            for case_id in cause.members:
                by_id[case_id].cause_id = cause.cause_id
        result = RunResult(outcomes, registry)
        self._attach_rest(result, outcomes, statuses=(LabelStatus.PROVISIONAL, LabelStatus.NO_FAULT_FOUND))
        return result

    def update(self, registry: CauseRegistry, cases: Iterable[Case]) -> RunResult:
        outcomes = self.label(cases)
        result = RunResult(outcomes, registry)
        self._attach_rest(
            result, outcomes, statuses=(LabelStatus.TRUSTED, LabelStatus.PROVISIONAL, LabelStatus.NO_FAULT_FOUND)
        )
        pool = [o for o, _ in result.unassigned if o.status is LabelStatus.TRUSTED]
        result.proposals = self._clusterer.cluster(pool, status="proposed")
        taken = set(registry.causes)
        for proposal in result.proposals:
            proposal.cause_id = unique_id(proposal.cause_id, taken)
            taken.add(proposal.cause_id)
        return result

    def _attach_rest(self, result: RunResult, outcomes: list[LabeledOutcome], statuses: tuple[LabelStatus, ...]) -> None:
        for o in outcomes:
            if o.status not in statuses:
                continue
            if o.status is LabelStatus.NO_FAULT_FOUND:
                result.registry.attach_no_fault_found(o)
                continue
            assignment = result.registry.assign(o)
            if assignment.cause_id is None:
                result.unassigned.append((o, assignment))
            else:
                result.registry.attach(o, assignment.cause_id, provisional=o.status is LabelStatus.PROVISIONAL)
