"""Domain records shared by the labelling pipeline and the online scorer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutcomeType(str, Enum):
    """What happened on the visit, as opposed to what was wrong."""

    FIXED = "fixed"
    NO_FAULT_FOUND = "no_fault_found"
    REPLACED_WITHOUT_DIAGNOSIS = "replaced_without_diagnosis"  # warranty/policy swap: no cause established
    NOT_FIXED = "not_fixed"  # temporary fix or return visit
    NOT_TECHNICAL = "not_technical"


class Mechanism(str, Enum):
    """How a component failed, from a closed list after ISO 14224. It never splits a cause on its own."""

    WEAR = "wear"
    BREAKAGE = "breakage"
    LEAKAGE = "leakage"
    LOOSENESS = "looseness"
    CORROSION = "corrosion"
    BLOCKAGE = "blockage"
    CONTAMINATION = "contamination"
    ELECTRICAL_OPEN = "electrical_open"
    ELECTRICAL_SHORT = "electrical_short"
    ELECTRICAL_OTHER = "electrical_other"
    LOW_LEVEL = "low_level"
    MISADJUSTMENT = "misadjustment"
    CONTROL_SOFTWARE = "control_software"
    OTHER = "other"


class Certainty(str, Enum):
    OBSERVED = "observed"  # tested, measured or visibly damaged
    INFERRED = "inferred"  # declared defective, or the fault went away after a swap


class PartRole(str, Enum):
    FIXES_CAUSE = "fixes_cause"
    CONSUMABLE = "consumable"
    WHOLE_UNIT = "whole_unit"
    OTHER = "other"


class LabelStatus(str, Enum):
    TRUSTED = "trusted"  # counted in cause statistics
    PROVISIONAL = "provisional"  # attached to a cause, not counted
    NO_FAULT_FOUND = "no_fault_found"  # per-family pseudo-cause
    EXCLUDED = "excluded"
    REVIEW = "review"  # failed a blocking check


@dataclass(frozen=True)
class Case:
    case_id: str
    equipment_family: str
    equipment_type: str
    created_at: str
    language: str
    customer_description: str
    technician_notes: str | None
    parts_replaced: tuple[str, ...]
    resolution_text: str | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Case:
        return cls(
            case_id=d["case_id"],
            equipment_family=d["equipment_family"],
            equipment_type=d["equipment_type"],
            created_at=d["created_at"],
            language=d.get("language") or "unknown",
            customer_description=d.get("customer_description") or "",
            technician_notes=d.get("technician_notes"),
            parts_replaced=tuple(d.get("parts_replaced") or ()),
            resolution_text=d.get("resolution_text"),
        )

    @property
    def technician_text(self) -> str:
        """Everything written after someone looked at the machine."""
        return " ".join(t for t in (self.technician_notes, self.resolution_text) if t)


@dataclass(frozen=True)
class RootCause:
    component: str
    location: str | None
    mechanism: Mechanism
    certainty: Certainty
    evidence_quote: str


@dataclass(frozen=True)
class PartUse:
    part_number: str
    role: PartRole


@dataclass(frozen=True)
class Extraction:
    """What the LLM read out of one closed case. Untrusted until validated."""

    outcome_type: OutcomeType
    root_cause: RootCause | None
    parts: tuple[PartUse, ...]
    customer_symptoms: tuple[str, ...]
    error_codes: tuple[str, ...]
    customer_report_contradicted: bool
    extractor_version: str

    @classmethod
    def from_dict(cls, d: dict[str, Any], extractor_version: str) -> Extraction:
        rc = d.get("root_cause")
        return cls(
            outcome_type=OutcomeType(d["outcome_type"]),
            root_cause=None
            if rc is None
            else RootCause(
                component=rc["component"],
                location=rc.get("location"),
                mechanism=Mechanism(rc["mechanism"]),
                certainty=Certainty(rc["certainty"]),
                evidence_quote=rc["evidence_quote"],
            ),
            parts=tuple(PartUse(p["part_number"], PartRole(p["role"])) for p in d.get("parts", ())),
            customer_symptoms=tuple(d.get("customer_symptoms", ())),
            error_codes=tuple(d.get("error_codes", ())),
            customer_report_contradicted=bool(d.get("customer_report_contradicted", False)),
            extractor_version=extractor_version,
        )

    def to_dict(self) -> dict[str, Any]:
        rc = self.root_cause
        return {
            "outcome_type": self.outcome_type.value,
            "root_cause": None
            if rc is None
            else {
                "evidence_quote": rc.evidence_quote,
                "component": rc.component,
                "location": rc.location,
                "mechanism": rc.mechanism.value,
                "certainty": rc.certainty.value,
            },
            "parts": [{"part_number": p.part_number, "role": p.role.value} for p in self.parts],
            "customer_symptoms": list(self.customer_symptoms),
            "error_codes": list(self.error_codes),
            "customer_report_contradicted": self.customer_report_contradicted,
        }


@dataclass(frozen=True)
class Issue:
    code: str
    detail: str
    blocking: bool = True  # sends the case to review


@dataclass
class LabeledOutcome:
    """One case after extraction, validation and status. ``to_record`` is a ``case_record`` row."""

    case: Case
    extraction: Extraction | None
    status: LabelStatus
    reason: str
    issues: tuple[Issue, ...] = ()
    error_codes: tuple[str, ...] = ()  # normalised
    cause_part_families: frozenset[str] = field(default_factory=frozenset)
    cause_id: str | None = None

    @property
    def component(self) -> str | None:
        rc = self.extraction.root_cause if self.extraction else None
        return rc.component if rc else None

    def to_record(self) -> dict[str, Any]:
        ex = self.extraction
        rc = ex.root_cause if ex else None
        return {
            "case_id": self.case.case_id,
            "equipment_family": self.case.equipment_family,
            "equipment_type": self.case.equipment_type,
            "created_at": self.case.created_at,
            "language": self.case.language,
            "symptom_text": self.case.customer_description,
            "customer_symptoms_en": list(ex.customer_symptoms) if ex else [],
            "error_codes": list(self.error_codes),
            "outcome_type": ex.outcome_type.value if ex else None,
            "label_status": self.status.value,
            "label_reason": self.reason,
            "cause_id": self.cause_id,
            "component": rc.component if rc else None,
            "location": rc.location if rc else None,
            "mechanism": rc.mechanism.value if rc else None,
            "certainty": rc.certainty.value if rc else None,
            "evidence_quote": rc.evidence_quote if rc else None,
            "parts": [{"part_number": p.part_number, "role": p.role.value} for p in ex.parts] if ex else [],
            "cause_part_families": sorted(self.cause_part_families),
            "customer_report_contradicted": ex.customer_report_contradicted if ex else None,
            "issues": [i.code for i in self.issues],
            "extractor_version": ex.extractor_version if ex else None,
        }
