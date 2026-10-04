"""Label new closed cases against the cause list built from the sample.

    python scripts/label_case.py examples/*.json

A case file is {"case": {...}, "extraction": {...}}, in the shapes of
fixtures/sample_cases.json and fixtures/extractions_sample.json. The extraction is what
the LLM would return: you write it and play the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagnostic_assist.domain import Case, LabeledOutcome, LabelStatus  # noqa: E402
from diagnostic_assist.labeling.canonicalize import cause_table  # noqa: E402
from diagnostic_assist.labeling.extractors import FirstAvailableExtractor, RecordedExtractor  # noqa: E402
from diagnostic_assist.labeling.pipeline import LabelingPipeline, RunResult  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_files", nargs="+", type=Path)
    parser.add_argument(
        "--corpus-extractions",
        type=Path,
        default=ROOT / "fixtures" / "extractions_model.json",
        help="recorded model extractions of the 22 sample cases (the existing corpus)",
    )
    args = parser.parse_args()

    corpus = [Case.from_dict(d) for d in json.loads((ROOT / "fixtures" / "sample_cases.json").read_text())]
    new_cases, file_extractions = load_case_files(args.case_files, {c.case_id for c in corpus})

    missing = [c.case_id for c in new_cases if c.case_id not in file_extractions]
    if missing:
        sys.exit(f"no extraction in the file for {missing}: add one in the shape of fixtures/extractions_sample.json")
    corpus_extractor = RecordedExtractor.from_file(args.corpus_extractions)
    file_extractor = RecordedExtractor(file_extractions, version="from case file")
    pipeline = LabelingPipeline(FirstAvailableExtractor([corpus_extractor, file_extractor]))
    base = pipeline.backfill(corpus)
    n_causes = sum(1 for c in base.registry.causes.values() if c.kind == "cause")
    print(f"Existing corpus: {len(corpus)} sample cases -> {n_causes} causes\n")

    update = pipeline.update(base.registry, new_cases)
    table = {row["cause_id"]: row for row in cause_table(update.registry, base.outcomes + update.outcomes)}
    for outcome in update.outcomes:
        report(outcome, update, table)


def load_case_files(paths: list[Path], corpus_ids: set[str]) -> tuple[list[Case], dict[str, dict]]:
    cases: list[Case] = []
    extractions: dict[str, dict] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        case = Case.from_dict(data["case"])
        if case.case_id in corpus_ids or case.case_id in extractions or any(c.case_id == case.case_id for c in cases):
            sys.exit(f"{path}: case_id {case.case_id} is already used; pick a new one")
        cases.append(case)
        if data.get("extraction") is not None:
            extractions[case.case_id] = data["extraction"]
    return cases, extractions


def report(o: LabeledOutcome, update: RunResult, table: dict[str, dict]) -> None:
    c = o.case
    print("=" * 100)
    print(f"{c.case_id}  {c.equipment_family} / {c.equipment_type}  ({c.language})")
    print(f"  customer:   {c.customer_description}")
    print(f"  technician: {c.technician_notes or '-'}")
    print(f"  resolution: {c.resolution_text or '-'}")
    print(f"  parts:      {', '.join(c.parts_replaced) or '-'}")

    ex = o.extraction
    if ex is not None:
        print(f"\n  Extraction ({ex.extractor_version}):")
        print(f"    outcome:   {ex.outcome_type.value}")
        if ex.root_cause:
            rc = ex.root_cause
            where = f" @ {rc.location}" if rc.location else ""
            print(f"    cause:     {rc.component}{where} ({rc.mechanism.value}, {rc.certainty.value})")
            print(f'    evidence:  "{rc.evidence_quote}"')
        print(f"    parts:     {', '.join(f'{p.part_number}={p.role.value}' for p in ex.parts) or '-'}")
        print(f"    codes:     {', '.join(o.error_codes) or '-'}")

    print("\n  Checks:")
    if not o.issues:
        print("    all passed")
    for issue in o.issues:
        print(f"    {'FAIL' if issue.blocking else 'note'}  {issue.code}: {issue.detail}")

    print(f"\n  Status: {o.status.value} ({o.reason})")
    print(f"  Result: {result_line(o, update, table)}\n")


def result_line(o: LabeledOutcome, update: RunResult, table: dict[str, dict]) -> str:
    if o.cause_id is not None:
        row = table[o.cause_id]
        verb = "attached (provisional, not counted) to" if o.status is LabelStatus.PROVISIONAL else "JOINS existing cause"
        parts = ", ".join(f"{p} {v:.0%}" for p, v in row["p_part"].items()) or "no part"
        return (
            f"{verb} {o.cause_id}\n"
            f"          that cause now: {row['n_trusted']} cases {row['by_type']}, parts: {parts}, "
            f"languages: {'/'.join(sorted(row['languages']))}"
        )
    pooled = next((a for x, a in update.unassigned if x is o), None)
    if pooled is not None:
        proposal = next((p for p in update.proposals if o.case.case_id in p.members), None)
        why = f"no existing cause fits ({pooled.reason})"
        if proposal is None:
            return f"{why}; held in the unassigned pool"
        return (
            f"NEW CAUSE PROPOSED: {proposal.cause_id}\n"
            f"          {why}.\n"
            f"          Goes to expert review; not counted until approved."
        )
    if o.status is LabelStatus.REVIEW:
        return "held for human review; it influences nothing until someone fixes or confirms it"
    return "no cause label (does not count towards any cause)"


if __name__ == "__main__":
    main()
