"""(case, extraction, issues) -> LabeledOutcome, whose status decides what the case may influence."""

from __future__ import annotations

from dataclasses import replace

from ..domain import Case, Extraction, Issue, LabeledOutcome, LabelStatus, OutcomeType, PartRole, PartUse
from .normalize import PartCatalogue, normalise_error_code, part_family


def decide_status(extraction: Extraction, issues: list[Issue]) -> tuple[LabelStatus, str]:
    if blocking := [i.code for i in issues if i.blocking]:
        return LabelStatus.REVIEW, f"failed checks: {', '.join(blocking)}"
    match extraction.outcome_type:
        case OutcomeType.NOT_TECHNICAL:
            return LabelStatus.EXCLUDED, "not a technical case"
        case OutcomeType.REPLACED_WITHOUT_DIAGNOSIS:
            return LabelStatus.EXCLUDED, "replaced without diagnosis: cause unknown"
        case OutcomeType.NO_FAULT_FOUND:
            return LabelStatus.NO_FAULT_FOUND, "technician found no fault"
        case OutcomeType.NOT_FIXED if extraction.root_cause is not None:
            return LabelStatus.PROVISIONAL, "not fixed on this visit; provisional diagnosis"
        case OutcomeType.NOT_FIXED:
            return LabelStatus.EXCLUDED, "not fixed and no diagnosis stated"
        case OutcomeType.FIXED:
            return LabelStatus.TRUSTED, f"fixed; cause {extraction.root_cause.certainty.value}"
    raise AssertionError(f"unhandled outcome {extraction.outcome_type}")


def apply_catalogue(extraction: Extraction, catalogue: PartCatalogue) -> tuple[Extraction, list[Issue]]:
    """Master data beats the model on part roles; overrides are recorded so drift shows."""
    parts: list[PartUse] = []
    issues: list[Issue] = []
    for p in extraction.parts:
        hint = catalogue.role_hint(p.part_number)
        if hint is not None and hint is not p.role:
            issues.append(
                Issue("part_role_overridden", f"{p.part_number}: {p.role.value} -> {hint.value}", blocking=False)
            )
            p = replace(p, role=hint)
        parts.append(p)
    return replace(extraction, parts=tuple(parts)), issues


def build_outcome(case: Case, extraction: Extraction, issues: list[Issue]) -> LabeledOutcome:
    status, reason = decide_status(extraction, issues)
    # The regex only cross-checks (validate.py): it also matches bolt sizes and part numbers.
    codes = {c for raw in extraction.error_codes if (c := normalise_error_code(raw))}
    fix_parts = frozenset(
        part_family(p.part_number, case.equipment_type) for p in extraction.parts if p.role is PartRole.FIXES_CAUSE
    )
    return LabeledOutcome(
        case=case,
        extraction=extraction,
        status=status,
        reason=reason,
        issues=tuple(issues),
        error_codes=tuple(sorted(codes)),
        cause_part_families=fix_parts,
    )
