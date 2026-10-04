"""Run the label factory on the sample cases.

    python scripts/run_labeling.py
    python scripts/run_labeling.py --extractions fixtures/extractions_sample.json

Part 1 backfills all 22 cases. Part 2 builds the registry from cases before 2026-03-08
and feeds the later ones in as new. Writes output/case_records.jsonl and
output/cause_registry.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagnostic_assist.domain import Case, LabelStatus  # noqa: E402
from diagnostic_assist.labeling.canonicalize import cause_table  # noqa: E402
from diagnostic_assist.labeling.extractors import RecordedExtractor  # noqa: E402
from diagnostic_assist.labeling.pipeline import LabelingPipeline  # noqa: E402
from diagnostic_assist.labeling.quality import check  # noqa: E402

SPLIT_DATE = "2026-03-08"


def load_cases() -> list[Case]:
    raw = json.loads((ROOT / "fixtures" / "sample_cases.json").read_text(encoding="utf-8"))
    return sorted((Case.from_dict(d) for d in raw), key=lambda c: c.created_at)


def print_outcomes(outcomes) -> None:
    print(f"{'case':9} {'status':15} {'cause / reason':58} issues")
    for o in outcomes:
        what = o.cause_id or o.reason
        issues = ",".join(i.code for i in o.issues) or "-"
        print(f"{o.case.case_id:9} {o.status.value:15} {what[:58]:58} {issues}")


def print_cause_table(rows) -> None:
    family = None
    for r in rows:
        fam = r["cause_id"].split("/")[0]
        if fam != family:
            family = fam
            print(f"\n  [{fam}]")
        parts = ", ".join(f"{p} {v:.0%}" for p, v in r["p_part"].items()) or "no part"
        extra = f" +{r['n_provisional']} provisional" if r["n_provisional"] else ""
        langs = "/".join(sorted(r["languages"])) or "-"
        codes = ",".join(r["error_codes"]) or "-"
        print(
            f"  {r['label'][:40]:40} n={r['n_trusted']}{extra:15} parts: {parts:30} "
            f"langs: {langs:8} codes: {codes:6} {r['members']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extractions",
        type=Path,
        default=ROOT / "fixtures" / "extractions_model.json",
        help="recorded extractions to replay (default: real model output; extractions_sample.json is hand-written)",
    )
    args = parser.parse_args()

    extractor = RecordedExtractor.from_file(args.extractions)
    print(f"extractions: {args.extractions.name}")
    cases = load_cases()

    print("=" * 100)
    print("1. BACKFILL over all sample cases")
    print("=" * 100)
    pipeline = LabelingPipeline(extractor)
    result = pipeline.backfill(cases)
    print_outcomes(result.outcomes)
    print("\nCause list (what the expert reviews, and what the online scorer counts over):")
    print_cause_table(cause_table(result.registry, result.outcomes))
    for o, a in result.unassigned:
        print(f"\n  unassigned: {o.case.case_id} ({o.status.value}): {a.reason}")
    report = check(result.registry, result.outcomes)
    print("\nQuality metrics:", json.dumps(report.metrics, indent=2))
    print("Quality findings:", [f"{f.code}@{f.subject}: {f.detail}" for f in report.findings] or "none")

    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    with (out / "case_records.jsonl").open("w", encoding="utf-8") as fh:
        for o in result.outcomes:
            fh.write(json.dumps(o.to_record(), ensure_ascii=False) + "\n")
    (out / "cause_registry.json").write_text(
        json.dumps(result.registry.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 100)
    print(f"2. NIGHTLY PATH: registry from cases before {SPLIT_DATE}, later cases arrive as new")
    print("=" * 100)
    before = [c for c in cases if c.created_at < SPLIT_DATE]
    after = [c for c in cases if c.created_at >= SPLIT_DATE]
    pipeline = LabelingPipeline(extractor)
    base = pipeline.backfill(before)
    print(f"registry built from {len(before)} cases: {len(base.registry.causes)} causes")
    update = pipeline.update(base.registry, after)
    for o in update.outcomes:
        if o.cause_id:
            print(f"  {o.case.case_id} -> assigned to {o.cause_id}")
        elif o.status in (LabelStatus.TRUSTED, LabelStatus.PROVISIONAL):
            reason = next(a.reason for x, a in update.unassigned if x is o)
            print(f"  {o.case.case_id} -> pool ({reason})")
        else:
            print(f"  {o.case.case_id} -> {o.status.value}: {o.reason}")
    print("\nProposed new causes from the pool (go to expert review, not straight into the counts):")
    for p in update.proposals:
        print(f"  {p.cause_id:55} members={p.members}")


if __name__ == "__main__":
    main()
