"""Nightly checks that need no ground truth, plus B-cubed against the gold set.

Parts purity catches merged causes; the language test and same-fix-parts pairs catch a cause
split in two; the singleton share and status mix catch threshold or extractor drift.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..config import LabelingConfig
from ..domain import LabeledOutcome, LabelStatus
from .canonicalize import Cause, CauseRegistry


@dataclass(frozen=True)
class Finding:
    code: str
    subject: str  # cause id or family
    detail: str


@dataclass
class QualityReport:
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)


def check(registry: CauseRegistry, outcomes: Iterable[LabeledOutcome], config: LabelingConfig = LabelingConfig()) -> QualityReport:
    outcomes = list(outcomes)
    by_id = {o.case.case_id: o for o in outcomes}
    report = QualityReport()

    n = len(outcomes)
    status_counts = Counter(o.status.value for o in outcomes)
    report.metrics["cases"] = n
    report.metrics["status"] = dict(status_counts)
    report.metrics["review_rate"] = round(status_counts.get(LabelStatus.REVIEW.value, 0) / n, 3) if n else 0.0
    report.metrics["issues"] = dict(Counter(i.code for o in outcomes for i in o.issues))
    report.metrics["excluded_reasons"] = dict(Counter(o.reason for o in outcomes if o.status is LabelStatus.EXCLUDED))

    trusted = [o for o in outcomes if o.status is LabelStatus.TRUSTED]
    if trusted:
        report.metrics["trusted_observed_share"] = round(
            sum(1 for o in trusted if o.extraction.root_cause.certainty.value == "observed") / len(trusted), 3
        )
        report.metrics["trusted_customer_contradicted_share"] = round(
            sum(1 for o in trusted if o.extraction.customer_report_contradicted) / len(trusted), 3
        )
        # Non-English quotes are not yet checked against the component (validate.py): how much
        # of each language's labels rests on the quote alone.
        report.metrics["trusted_component_not_checked_by_language"] = _issue_share_by_language(
            trusted, "component_not_checked"
        )

    causes = [c for c in registry.causes.values() if c.kind == "cause"]
    if causes:
        report.metrics["causes"] = len(causes)
        report.metrics["singleton_cause_share"] = round(sum(1 for c in causes if len(c.members) == 1) / len(causes), 3)

    lang_share = _family_language_share(trusted)
    for cause in causes:
        purity = parts_purity(cause, by_id)
        if purity is not None and purity < config.min_parts_purity:
            report.findings.append(
                Finding("low_parts_purity", cause.cause_id, f"{purity:.0%} of members share the modal fix part")
            )
        p = language_split_probability(cause, by_id, lang_share, config.min_cluster_size_for_language_check)
        if p is not None and p < config.language_split_alpha:
            report.findings.append(
                Finding(
                    "single_language_cause",
                    cause.cause_id,
                    f"all members share one language; p={p:.3f} under the family's language mix",
                )
            )
    for family, parts, cause_ids in same_fix_parts(causes, by_id):
        report.findings.append(
            Finding(
                "same_fix_parts",
                family,
                f"{', '.join(cause_ids)} are fixed with the same parts {sorted(parts)}: merge candidates",
            )
        )
    return report


def same_fix_parts(
    causes: list[Cause], by_id: dict[str, LabeledOutcome]
) -> list[tuple[str, frozenset[str], list[str]]]:
    """Causes of one family whose modal fix parts are identical.

    By the granularity rule (same parts, same repair) these are probably one cause split in
    two. A split is cheaper than a merge, not free: it halves the counts and takes two slots
    in the dispatcher's list. So it is surfaced for the reviewer, never merged automatically.
    """
    groups: dict[tuple[str, frozenset[str]], list[str]] = {}
    for cause in causes:
        sets = [by_id[m].cause_part_families for m in cause.members if m in by_id and by_id[m].cause_part_families]
        if sets:
            modal = Counter(sets).most_common(1)[0][0]
            groups.setdefault((cause.family, modal), []).append(cause.cause_id)
    return [(family, parts, ids) for (family, parts), ids in groups.items() if len(ids) > 1]


def parts_purity(cause: Cause, by_id: dict[str, LabeledOutcome]) -> float | None:
    """Share of members (with any fix part) whose fix-part set is the modal one."""
    sets = [by_id[m].cause_part_families for m in cause.members if m in by_id and by_id[m].cause_part_families]
    if len(sets) < 2:
        return None
    return Counter(sets).most_common(1)[0][1] / len(sets)


def language_split_probability(
    cause: Cause, by_id: dict[str, LabeledOutcome], family_lang_share: dict[str, dict[str, float]], min_size: int
) -> float | None:
    """share_L ** n: the chance all n members share language L if language were unrelated to cause."""
    members = [by_id[m] for m in cause.members if m in by_id]
    if len(members) < min_size:
        return None
    langs = {o.case.language for o in members}
    if len(langs) != 1:
        return None
    share = family_lang_share.get(cause.family, {}).get(langs.pop(), 1.0)
    return share ** len(members)


def _issue_share_by_language(outcomes: list[LabeledOutcome], code: str) -> dict[str, float]:
    totals: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for o in outcomes:
        totals[o.case.language] += 1
        hits[o.case.language] += any(i.code == code for i in o.issues)
    return {lang: round(hits[lang] / n, 3) for lang, n in sorted(totals.items())}


def _family_language_share(trusted: list[LabeledOutcome]) -> dict[str, dict[str, float]]:
    counts: dict[str, Counter[str]] = {}
    for o in trusted:
        counts.setdefault(o.case.equipment_family, Counter())[o.case.language] += 1
    return {f: {lang: k / sum(c.values()) for lang, k in c.items()} for f, c in counts.items()}


def bcubed(predicted: dict[str, str], gold: dict[str, str]) -> tuple[float, float, float]:
    """B-cubed precision, recall and F1 over the items in both mappings.

    Defined per case, so it can be sliced by family, language or type.
    """
    items = sorted(set(predicted) & set(gold))
    if not items:
        return 0.0, 0.0, 0.0
    pred_groups: dict[str, set[str]] = {}
    gold_groups: dict[str, set[str]] = {}
    for i in items:
        pred_groups.setdefault(predicted[i], set()).add(i)
        gold_groups.setdefault(gold[i], set()).add(i)
    p = r = 0.0
    for i in items:
        overlap = len(pred_groups[predicted[i]] & gold_groups[gold[i]])
        p += overlap / len(pred_groups[predicted[i]])
        r += overlap / len(gold_groups[gold[i]])
    p, r = p / len(items), r / len(items)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)
