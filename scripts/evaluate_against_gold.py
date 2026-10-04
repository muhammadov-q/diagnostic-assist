"""B-cubed against the hand labels, swept over the merge threshold, with and without parts.

    python scripts/evaluate_against_gold.py [--extractions other_extractions.json]

On 22 cases this shows where the threshold is safe, not that it is right.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagnostic_assist.config import LabelingConfig  # noqa: E402
from diagnostic_assist.domain import Case  # noqa: E402
from diagnostic_assist.labeling.extractors import RecordedExtractor  # noqa: E402
from diagnostic_assist.labeling.pipeline import LabelingPipeline  # noqa: E402
from diagnostic_assist.labeling.quality import bcubed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractions", type=Path, default=ROOT / "fixtures" / "extractions_model.json")
    args = parser.parse_args()

    cases = [Case.from_dict(d) for d in json.loads((ROOT / "fixtures" / "sample_cases.json").read_text())]
    gold = json.loads((ROOT / "fixtures" / "gold_causes.json").read_text())
    extractor = RecordedExtractor.from_file(args.extractions)
    print(f"extractions: {args.extractions.name}")

    for use_parts in (True, False):
        print(f"\nparts constraint {'ON' if use_parts else 'OFF (ablation: text similarity only)'}")
        sweep(cases, gold, extractor, use_parts)


def sweep(cases, gold, extractor, use_parts: bool) -> None:
    # B-cubed scores trusted cases only; "covered" shows how many gold cases got there.
    print(f"{'threshold':>9} {'P':>6} {'R':>6} {'F1':>6} {'covered':>8} {'causes':>7}  status/provisional ok")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95]:
        config = replace(LabelingConfig(), merge_threshold=t, use_parts=use_parts)
        result = LabelingPipeline(extractor, config=config).backfill(cases)
        predicted = {o.case.case_id: o.cause_id for o in result.outcomes if o.status.value == "trusted"}
        p, r, f = bcubed(predicted, gold["causes"])
        covered = len(set(predicted) & set(gold["causes"])) / len(gold["causes"])
        status_ok = all(
            next(o for o in result.outcomes if o.case.case_id == cid).status.value == s
            for cid, s in gold["status"].items()
        )
        # Provisional case must sit with the gold cause's members.
        prov_ok = True
        for cid, gold_cause in gold["provisional"].items():
            o = next(o for o in result.outcomes if o.case.case_id == cid)
            peers = {k for k, v in gold["causes"].items() if v == gold_cause}
            prov_ok &= o.cause_id is not None and peers <= set(result.registry.causes[o.cause_id].members)
        n_causes = sum(1 for c in result.registry.causes.values() if c.kind == "cause")
        print(f"{t:>9.2f} {p:>6.2f} {r:>6.2f} {f:>6.2f} {covered:>8.0%} {n_causes:>7}  {status_ok}/{prov_ok}")


if __name__ == "__main__":
    main()
