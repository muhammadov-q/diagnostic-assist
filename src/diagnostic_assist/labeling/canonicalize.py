"""From free-text cause mentions to one cause list per equipment family.

Granularity rule: two outcomes share a cause iff a technician would bring the same parts
and do the same repair. "Main contactor" and "Hauptschütz" are one cause; a boom-cylinder
seal kit and a whole cylinder are two.

Per family, identical (component, parts) mentions are deduplicated, compared by text
distance adjusted by fix parts (disjoint parts cannot link), and grouped by complete-linkage
clustering: k is unknown, HDBSCAN would drop the small groups that are real causes on rare
types, and an over-split is cheap to fix while an over-merge is silent.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..config import LabelingConfig
from ..domain import LabeledOutcome, PartRole
from .embeddings import Embedder
from .normalize import normalise_component

NO_FAULT_FOUND_KIND = "no_fault_found"


@dataclass
class Exemplar:
    """One distinct way a cause has been written down: (component, parts)."""

    component: str
    part_families: frozenset[str]
    case_ids: list[str] = field(default_factory=list)


@dataclass
class Cause:
    cause_id: str
    family: str
    label: str
    kind: str = "cause"  # "cause" | "no_fault_found"
    status: str = "approved"  # "approved" | "proposed" (awaiting expert review)
    exemplars: list[Exemplar] = field(default_factory=list)
    members: list[str] = field(default_factory=list)  # trusted case ids
    provisional_members: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause_id": self.cause_id,
            "family": self.family,
            "label": self.label,
            "kind": self.kind,
            "status": self.status,
            "exemplars": [
                {"component": e.component, "part_families": sorted(e.part_families), "case_ids": e.case_ids}
                for e in self.exemplars
            ],
            "members": self.members,
            "provisional_members": self.provisional_members,
        }


@dataclass(frozen=True)
class Assignment:
    cause_id: str | None
    distance: float | None
    reason: str


def unique_id(base: str, taken: Collection[str]) -> str:
    cause_id, n = base, 2
    while cause_id in taken:
        cause_id, n = f"{base}-{n}", n + 1
    return cause_id


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def apply_parts(text_distance: float, a: frozenset[str], b: frozenset[str], discount: float) -> float:
    if not a or not b:
        return text_distance
    shared = len(a & b)
    if shared == 0:
        return 1.0
    return text_distance * (1.0 - discount * shared / len(a | b))


def distance_matrix(vectors: np.ndarray, part_sets: Sequence[frozenset[str]], discount: float) -> np.ndarray:
    dist = 1.0 - np.clip(vectors @ vectors.T, 0.0, 1.0)
    n = len(part_sets)
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = apply_parts(dist[i, j], part_sets[i], part_sets[j], discount)
    np.fill_diagonal(dist, 0.0)
    return dist


def _mentions(outcomes: Iterable[LabeledOutcome]) -> dict[str, list[Exemplar]]:
    """family -> deduplicated (component, parts) mentions."""
    by_key: dict[tuple[str, str, frozenset[str]], Exemplar] = {}
    for o in outcomes:
        if o.component is None:
            continue
        key = (o.case.equipment_family, normalise_component(o.component), o.cause_part_families)
        ex = by_key.setdefault(key, Exemplar(key[1], key[2]))
        ex.case_ids.append(o.case.case_id)
    families: dict[str, list[Exemplar]] = {}
    for (family, _, _), ex in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1], sorted(kv[0][2]))):
        families.setdefault(family, []).append(ex)
    return families


def _label(exemplars: list[Exemplar]) -> str:
    # Most frequent wording, ties to the shortest; the expert renames on review.
    return min(exemplars, key=lambda e: (-len(e.case_ids), len(e.component), e.component)).component


class CauseClusterer:
    def __init__(self, embedder: Embedder, config: LabelingConfig = LabelingConfig()):
        self._embedder = embedder
        self._config = config

    def cluster(self, outcomes: Iterable[LabeledOutcome], status: str = "approved") -> list[Cause]:
        causes: list[Cause] = []
        for family, mentions in _mentions(outcomes).items():
            for group in self._cluster_family(mentions):
                members = [cid for e in group for cid in e.case_ids]
                label = _label(group)
                causes.append(
                    Cause(
                        cause_id=f"{slug(family)}/{slug(label)}",
                        family=family,
                        label=label,
                        status=status,
                        exemplars=group,
                        members=sorted(members),
                    )
                )
        return causes

    def build_registry(self, trusted: Iterable[LabeledOutcome]) -> CauseRegistry:
        registry = CauseRegistry(self._embedder, self._config)
        for cause in self.cluster(trusted):
            registry.add(cause)
        return registry

    def _cluster_family(self, mentions: list[Exemplar]) -> list[list[Exemplar]]:
        if len(mentions) == 1:
            return [mentions]
        vectors = self._embedder.embed([m.component for m in mentions])
        part_sets = [m.part_families if self._config.use_parts else frozenset() for m in mentions]
        dist = distance_matrix(vectors, part_sets, self._config.parts_agree_discount)
        tree = linkage(squareform(dist, checks=False), method="complete")
        labels = fcluster(tree, t=self._config.merge_threshold, criterion="distance")
        groups: dict[int, list[Exemplar]] = {}
        for mention, label in zip(mentions, labels):
            groups.setdefault(int(label), []).append(mention)
        return list(groups.values())


class CauseRegistry:
    """Approved causes and their members (the ``cause`` and ``cause_exemplar`` tables).

    IDs are immutable once approved. Stub: a full re-cluster is mapped onto existing IDs
    by member overlap and reviewed, never auto-applied.
    """

    def __init__(self, embedder: Embedder, config: LabelingConfig = LabelingConfig()):
        self._embedder = embedder
        self._config = config
        self.causes: dict[str, Cause] = {}
        self._vector_cache: dict[str, np.ndarray] = {}

    def add(self, cause: Cause) -> Cause:
        cause.cause_id = unique_id(cause.cause_id, self.causes)
        self.causes[cause.cause_id] = cause
        return cause

    def family_causes(self, family: str, kind: str = "cause") -> list[Cause]:
        return [c for c in self.causes.values() if c.family == family and c.kind == kind]

    def assign(self, outcome: LabeledOutcome) -> Assignment:
        """Nearest-exemplar assignment, only within the threshold and when unambiguous.

        Anything else goes to the pool, which becomes proposed causes for review.
        """
        if outcome.component is None:
            return Assignment(None, None, "no component to assign")
        candidates = self.family_causes(outcome.case.equipment_family)
        if not candidates:
            return Assignment(None, None, "family has no causes yet")
        component = normalise_component(outcome.component)
        v = self._vector(component)
        parts = outcome.cause_part_families if self._config.use_parts else frozenset()
        scored = sorted((self._distance_to_cause(v, parts, c), c.cause_id) for c in candidates)
        best_d, best_id = scored[0]
        threshold, margin = self._config.merge_threshold, self._config.assignment_margin
        if best_d > threshold:
            text_d, text_id = min((self._distance_to_cause(v, frozenset(), c), c.cause_id) for c in candidates)
            if text_d <= threshold:
                # Worth telling the reviewer: close in wording, different repair.
                return Assignment(None, best_d, f"text matches {text_id} ({text_d:.2f}) but replaced parts differ")
            return Assignment(None, best_d, f"nearest cause {best_id} at {best_d:.2f} > {threshold}")
        if len(scored) > 1 and scored[1][0] <= threshold and scored[1][0] - best_d < margin:
            return Assignment(None, best_d, f"ambiguous between {best_id} and {scored[1][1]}")
        return Assignment(best_id, best_d, "assigned")

    def attach(self, outcome: LabeledOutcome, cause_id: str, provisional: bool = False) -> None:
        cause = self.causes[cause_id]
        case_id = outcome.case.case_id
        (cause.provisional_members if provisional else cause.members).append(case_id)
        if not provisional and outcome.component is not None:
            component = normalise_component(outcome.component)
            match = next(
                (e for e in cause.exemplars if e.component == component and e.part_families == outcome.cause_part_families),
                None,
            )
            if match is None:
                cause.exemplars.append(Exemplar(component, outcome.cause_part_families, [case_id]))
            else:
                match.case_ids.append(case_id)
        outcome.cause_id = cause_id

    def attach_no_fault_found(self, outcome: LabeledOutcome) -> None:
        family = outcome.case.equipment_family
        cause_id = f"{slug(family)}/no-fault-found"
        if cause_id not in self.causes:
            self.add(Cause(cause_id, family, "no fault found", kind=NO_FAULT_FOUND_KIND))
        self.attach(outcome, cause_id)

    def _vector(self, component: str) -> np.ndarray:
        if component not in self._vector_cache:
            self._vector_cache[component] = self._embedder.embed([component])[0]
        return self._vector_cache[component]

    def _distance_to_cause(self, v: np.ndarray, parts: frozenset[str], cause: Cause) -> float:
        if self._config.use_parts and parts:
            # Checked against the whole cause, so a wording recorded without parts cannot bypass it.
            cause_parts = frozenset().union(*(e.part_families for e in cause.exemplars))
            if cause_parts and not parts & cause_parts:
                return 1.0
        best = 1.0
        for e in cause.exemplars:
            text_d = 1.0 - float(np.clip(v @ self._vector(e.component), 0.0, 1.0))
            exemplar_parts = e.part_families if self._config.use_parts else frozenset()
            best = min(best, apply_parts(text_d, parts, exemplar_parts, self._config.parts_agree_discount))
        return best

    def to_dict(self) -> dict[str, Any]:
        return {"embedder": self._embedder.version, "causes": [c.to_dict() for c in self.causes.values()]}


def cause_table(registry: CauseRegistry, outcomes: Iterable[LabeledOutcome]) -> list[dict[str, Any]]:
    """Per-cause counts the online scorer reads: by type, P(part | cause) and evidence."""
    by_id = {o.case.case_id: o for o in outcomes}
    rows = []
    for cause in registry.causes.values():
        members = [by_id[m] for m in cause.members if m in by_id]
        n = len(members)
        parts = Counter(part for o in members for part in _fix_parts(o))
        rows.append(
            {
                "cause_id": cause.cause_id,
                "label": cause.label,
                "kind": cause.kind,
                "status": cause.status,
                "n_trusted": n,
                "n_provisional": len(cause.provisional_members),
                "by_type": dict(Counter(o.case.equipment_type for o in members)),
                "p_part": {p: round(k / n, 2) for p, k in parts.most_common()} if n else {},
                "p_no_part": round(sum(1 for o in members if not o.cause_part_families) / n, 2) if n else None,
                "mechanisms": dict(
                    Counter(o.extraction.root_cause.mechanism.value for o in members if o.extraction and o.extraction.root_cause)
                ),
                "languages": dict(Counter(o.case.language for o in members)),
                "error_codes": dict(Counter(c for o in members for c in o.error_codes)),
                "members": cause.members,
            }
        )
    return sorted(rows, key=lambda r: (r["cause_id"].split("/")[0], -r["n_trusted"], r["cause_id"]))


def _fix_parts(outcome: LabeledOutcome) -> set[str]:
    parts = outcome.extraction.parts if outcome.extraction else ()
    return {p.part_number for p in parts if p.role is PartRole.FIXES_CAUSE}
