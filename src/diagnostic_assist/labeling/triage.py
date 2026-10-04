"""Cheap gate in front of the LLM: skip cases that cannot carry a label."""

from __future__ import annotations

from ..domain import Case


def skip_reason(case: Case) -> str | None:
    """Why the case is not worth an LLM call, or None.

    No keyword filters for billing and the like: they are language-dependent, and the
    LLM's ``not_technical`` outcome costs a fraction of a cent.
    """
    if not case.technician_text.strip():
        return "no technician notes and no resolution text"
    return None
