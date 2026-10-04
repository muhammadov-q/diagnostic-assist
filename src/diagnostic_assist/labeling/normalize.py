"""Deterministic normalisation of error codes, part numbers and text.

Independent of the LLM, so it can double-check it.
"""

from __future__ import annotations

import re
import unicodedata

from ..domain import PartRole

# "E-207", "LP-01". Case-sensitive so "cyl 1" never matches; \b keeps "R410A" out.
_PREFIXED_CODE = re.compile(r"\b([A-Z]{1,3})-?(\d{2,4})\b")
# "err 207", "Fehlercode 207", "erreur 207", "errore 207".
# Only the keyword ignores case, so "error at 300 hours" never yields "AT300".
_WORDED_CODE = re.compile(r"\b(?i:err(?:or|eur|ore)?|fehler(?:code)?)\.?\s*[:#-]?\s*([A-Z]{0,3})-?\s?(\d{2,4})\b")


def equipment_prefixes(equipment_types: set[str]) -> set[str]:
    """Letter prefixes of model names ("CX-450" -> "CX"): never error codes."""
    return {m.group(1) for t in equipment_types if (m := re.match(r"([A-Za-z]+)", t))}


def normalise_error_code(raw: str) -> str | None:
    """Canonical form, e.g. "E-207" -> "E207".

    A bare number after "err" is read as an E-code: true for this sample, a
    per-manufacturer fault-code table in production.
    """
    raw = raw.strip()
    if m := _WORDED_CODE.search(raw):
        return f"{(m.group(1) or 'E').upper()}{m.group(2)}"
    if m := _PREFIXED_CODE.search(raw.upper()):
        return f"{m.group(1)}{m.group(2)}"
    return None


def extract_error_codes(text: str, exclude_prefixes: set[str] = frozenset()) -> set[str]:
    codes: set[str] = set()
    for m in _WORDED_CODE.finditer(text):
        codes.add(f"{(m.group(1) or 'E').upper()}{m.group(2)}")
    for m in _PREFIXED_CODE.finditer(text):
        if m.group(1) not in exclude_prefixes:
            codes.add(f"{m.group(1)}{m.group(2)}")
    return codes


def part_family(part_number: str, equipment_type: str) -> str:
    """Strip the equipment model from a part number: SEALKIT-BC-120 on an AP-120 -> SEALKIT-BC.

    Stub: production uses the part class from the ERP parts master.
    """
    model = re.sub(r"[^A-Z0-9]", "", equipment_type.upper())
    digits = re.sub(r"\D", "", equipment_type)
    segments = [s for s in part_number.upper().split("-") if s not in {model, digits}]
    return "-".join(segments) or part_number.upper()


class PartCatalogue:
    """Stub of the ERP parts master (production: a nightly copy in Postgres).

    Overrides the LLM on whole-unit and consumable roles, which decide whether a
    case has a usable cause at all.
    """

    _CONSUMABLE_PREFIXES = ("REFRIG-", "OIL-", "GREASE-", "SEALANT-", "COOLANT-")
    _WHOLE_UNIT_PREFIXES = ("UNIT-",)

    def role_hint(self, part_number: str) -> PartRole | None:
        p = part_number.upper()
        if p.startswith(self._WHOLE_UNIT_PREFIXES):
            return PartRole.WHOLE_UNIT
        if p.startswith(self._CONSUMABLE_PREFIXES):
            return PartRole.CONSUMABLE
        return None


_TRANSLITERATE = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"})


def fold(text: str) -> str:
    """Comparison form for quote matching: umlauts transliterated, accents, case and punctuation dropped."""
    text = unicodedata.normalize("NFC", text).translate(_TRANSLITERATE).casefold()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", " ", text).strip()


# Generic heads carry no identity: "valve plate assembly" is a valve plate.
_GENERIC_HEADS = {"assembly", "unit", "kit", "set", "system", "component", "part", "module"}


def head_noun(component: str) -> str | None:
    """The word that names the component: the last non-generic one (the prompt puts it last)."""
    words = re.findall(r"[a-z0-9]+", component.lower())
    if not words:
        return None
    return next((w for w in reversed(words) if w not in _GENERIC_HEADS), words[-1])


def normalise_component(component: str) -> str:
    text = re.sub(r"\s+", " ", component.strip().lower()).strip(" .,;:")
    return re.sub(r"^(?:the|a|an) ", "", text)
