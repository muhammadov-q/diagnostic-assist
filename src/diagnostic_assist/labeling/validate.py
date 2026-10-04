"""Deterministic checks of an extraction against its case.

Blocking issues send the case to review: it never silently becomes a label, nor disappears.
"""

from __future__ import annotations

import re
from collections import Counter

from ..domain import Case, Extraction, Issue, OutcomeType, PartRole, RootCause
from .normalize import extract_error_codes, fold, head_noun, normalise_error_code

_NEGATIONS = {
    "no", "not", "without", "kein", "keine", "keinen", "nicht", "ohne",
    "pas", "sans", "aucun", "aucune", "non", "nessun", "nessuna", "senza",
}
# "Checked, and fine". Short on purpose: an entry that is wrong in some context sends good
# labels to review. Folded forms: "i.O." is "i o", "bon état" is "bon etat".
_CLEARED = {
    ("ok",), ("okay",), ("serviceable",), ("in", "spec"), ("within", "spec"),
    ("i", "o"), ("in", "ordnung"), ("einwandfrei",),
    ("ras",), ("conforme",), ("bon", "etat"),
    ("nella", "norma"), ("funzionante",),
}
# "n.i.O." (nicht in Ordnung), "not OK", "non conforme" all mean defective.
_NEGATES_CLEARED = _NEGATIONS | {"n"}
# A negation before one of these reports an absence ("no leak found"). A negation alone
# does not: "not charging" and "kein Strom" describe the fault.
_FINDING_VERBS = {
    "found", "detected", "observed",
    "gefunden", "festgestellt", "feststellbar",
    "trouve", "trouvee", "constate", "constatee", "detecte", "detectee",
    "trovato", "trovata", "riscontrato", "riscontrata", "rilevato", "rilevata",
}
# Short words that would pass for abbreviations of a component ("the" -> "thermostat").
_FUNCTION_WORDS = {"the", "and", "for", "was", "with", "not", "off", "out", "all", "has", "had", "are", "per", "but"}
# Not the dot in "12.1V" or "n.i.O.".
_CLAUSE_END = re.compile(r"[.,;:!?](?:\s+|$)")


def locate_quote(quote: str, case: Case) -> str | None:
    """Where the evidence quote appears: "technician", "customer" or None."""
    needle = f" {fold(quote)} "
    if needle.strip() == "":
        return None
    if needle in f" {fold(case.technician_text)} ":
        return "technician"
    if needle in f" {fold(case.customer_description)} ":
        return "customer"
    return None


def validate(case: Case, extraction: Extraction, exclude_code_prefixes: set[str] = frozenset()) -> list[Issue]:
    """Run after ``apply_catalogue``, so role checks see the corrected roles."""
    issues: list[Issue] = []
    issues += _check_parts(case, extraction)
    issues += _check_outcome_consistency(extraction)
    issues += _check_grounding(case, extraction)
    issues += _check_error_codes(case, extraction, exclude_code_prefixes)
    return issues


def _check_parts(case: Case, ex: Extraction) -> list[Issue]:
    issues: list[Issue] = []
    listed = Counter(p.part_number for p in ex.parts)
    actual = Counter(case.parts_replaced)
    if invented := listed - actual:
        issues.append(Issue("invented_part", f"not in parts_replaced: {sorted(invented)}"))
    if missing := actual - listed:
        issues.append(Issue("missing_part_role", f"no role given for: {sorted(missing)}"))
    return issues


def _check_outcome_consistency(ex: Extraction) -> list[Issue]:
    issues: list[Issue] = []
    has_cause = ex.root_cause is not None
    if ex.outcome_type is OutcomeType.FIXED and not has_cause:
        issues.append(Issue("fixed_without_cause", "outcome 'fixed' but no root cause"))
    no_cause = (OutcomeType.REPLACED_WITHOUT_DIAGNOSIS, OutcomeType.NOT_TECHNICAL, OutcomeType.NO_FAULT_FOUND)
    if ex.outcome_type in no_cause and has_cause:
        issues.append(Issue("unexpected_cause", f"outcome '{ex.outcome_type.value}' must not carry a cause"))
    if ex.outcome_type is OutcomeType.NO_FAULT_FOUND and any(p.role is PartRole.FIXES_CAUSE for p in ex.parts):
        issues.append(Issue("nff_with_fix_part", "no fault found, yet a part 'fixes the cause'"))
    if ex.outcome_type is OutcomeType.FIXED and any(p.role is PartRole.WHOLE_UNIT for p in ex.parts):
        # How a warranty swap turns into a fake root cause.
        issues.append(Issue("whole_unit_swap_marked_fixed", "whole unit replaced but outcome is 'fixed'"))
    return issues


def _check_grounding(case: Case, ex: Extraction) -> list[Issue]:
    if ex.root_cause is None:
        return []
    quote = ex.root_cause.evidence_quote
    where = locate_quote(quote, case)
    if where == "technician":
        if _negated(quote, case.technician_text):
            return [Issue("negated_evidence", f"quote is negated in the technician text: {quote!r}")]
        return _check_quote_states_fault(case, ex.root_cause)
    if where == "customer":
        return [Issue("evidence_from_customer", f"cause quoted from the customer: {ex.root_cause.evidence_quote!r}")]
    return [Issue("ungrounded_evidence", f"quote not found in technician text: {ex.root_cause.evidence_quote!r}")]


def _negated(quote: str, text: str) -> bool:
    """A negation just before the quote, in the same clause: "No oil leak found" is no
    evidence of an oil leak, but "Unit not starting. Contactor coil open." is of the contactor.
    """
    first = next((f for c in _CLAUSE_END.split(quote) if (f := fold(c))), "")
    for clause in _CLAUSE_END.split(text):
        folded = f" {fold(clause)} "
        if first and f" {first} " in folded:
            before = folded.split(f" {first} ", 1)[0].split()
            return any(word in _NEGATIONS for word in before[-2:])
    return False


def _check_quote_states_fault(case: Case, rc: RootCause) -> list[Issue]:
    """The quote must say the named component was at fault, not just exist.

    "Contactor tested OK." is a real technician sentence (C-48590) and no evidence of a bad
    contactor. Checked on the clauses that name the component, else on the whole quote.
    """
    clauses = [words for c in _CLAUSE_END.split(rc.evidence_quote) if (words := fold(c).split())]
    head = head_noun(fold(rc.component))
    naming = [c for c in clauses if head and _names(c, head)]
    checked = naming or clauses
    if checked and all(_states_no_fault(c) for c in checked):
        return [Issue("no_fault_in_quote", f"quote reports no fault: {rc.evidence_quote!r}")]
    if naming:
        return []
    if case.language == "en":
        return [Issue("quote_component_mismatch", f"quote does not name the {head}: {rc.evidence_quote!r}")]
    # Stub: "Hauptschütz" vs "main contactor" needs the multilingual embedder (quote vs
    # component similarity) or a second LLM call that sees only the quote and the component,
    # never the customer's text. Until then the share of unchecked cases is reported.
    return [Issue("component_not_checked", f"quote not in English; {head!r} not checked", blocking=False)]


def _names(words: list[str], head: str) -> bool:
    """The clause names the component, also in phone shorthand ("cyl", "batt") or plural.

    Only plural endings extend the head: "seals" names a seal, "sealant" does not.
    """
    forms = {head, f"{head}s", f"{head}es"}
    if head.endswith("y"):
        forms.add(f"{head[:-1]}ies")
    return any(w in forms or (len(w) >= 3 and w not in _FUNCTION_WORDS and head.startswith(w)) for w in words)


def _states_no_fault(words: list[str]) -> bool:
    """The clause reports the part as fine, or a fault as not found."""
    for i in range(len(words)):
        for phrase in _CLEARED:
            if tuple(words[i : i + len(phrase)]) == phrase and (i == 0 or words[i - 1] not in _NEGATES_CLEARED):
                return True
    first_negation = next((i for i, w in enumerate(words) if w in _NEGATIONS), None)
    return first_negation is not None and any(w in _FINDING_VERBS for w in words[first_negation + 1 :])


def _check_error_codes(case: Case, ex: Extraction, exclude_prefixes: set[str]) -> list[Issue]:
    llm_codes = {c for raw in ex.error_codes if (c := normalise_error_code(raw))}
    regex_codes = extract_error_codes(f"{case.customer_description} {case.technician_text}", exclude_prefixes)
    if missed := regex_codes - llm_codes:
        return [Issue("missed_error_code", f"found by regex only: {sorted(missed)}", blocking=False)]
    return []
