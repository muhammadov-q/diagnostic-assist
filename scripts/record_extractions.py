"""Record real model extractions of the sample cases.

    python scripts/record_extractions.py [--out fixtures/extractions_model.json] [--model opus]

Stand-in for ``LLMExtractor``: I have no API key, so this sends the same system prompt,
JSON schema and case rendering (``labeling/prompt.py``) through the Claude Code CLI
(``claude -p``). One call per case, as in production, so the model never sees two cases
together and cannot harmonise their wording. Tools are off, no MCP servers, and it runs
from an empty directory, so no project context reaches the model.

Differences from production: the CLI wraps the call (it adds a short preamble of its own),
and the batch API is not used. The output has the shape ``RecordedExtractor`` replays.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagnostic_assist.config import ExtractorConfig  # noqa: E402
from diagnostic_assist.domain import Case  # noqa: E402
from diagnostic_assist.labeling.prompt import EXTRACTION_SCHEMA, SYSTEM_PROMPT, render_case  # noqa: E402

# Set by a parent Claude Code session; the child must start as a fresh, independent one.
_PARENT_SESSION_VARS = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_PID",
)


def extract(case: Case, model: str, effort: str, workdir: str, attempts: int = 2) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """(extraction or None, call metadata). A failed case is left out: replay sends it to review."""
    cmd = [
        "claude", "-p", render_case(case),
        "--system-prompt", SYSTEM_PROMPT,
        "--json-schema", json.dumps(EXTRACTION_SCHEMA),
        "--output-format", "json",
        "--model", model,
        "--effort", effort,
        "--tools", "",
        "--strict-mcp-config",
        "--no-session-persistence",
    ]
    env = {k: v for k, v in os.environ.items() if k not in _PARENT_SESSION_VARS}
    error = ""
    for _ in range(attempts):
        proc = subprocess.run(cmd, cwd=workdir, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            error = (proc.stderr or proc.stdout)[:200]
            continue
        if out.get("is_error") or out.get("structured_output") is None:
            error = str(out.get("result") or out.get("subtype"))[:200]
            continue
        usage = out.get("modelUsage", {})
        # The CLI may also call a small model for housekeeping; the extraction is the one that wrote most.
        model_id = max(usage, key=lambda m: usage[m].get("outputTokens", 0)) if usage else model
        return out["structured_output"], {"model": model_id, "cost_usd": out.get("total_cost_usd", 0.0)}
    return None, {"model": model, "cost_usd": 0.0, "error": error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "fixtures" / "extractions_model.json")
    parser.add_argument("--model", default="opus")
    parser.add_argument("--effort", default=ExtractorConfig().effort)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    cases = [Case.from_dict(d) for d in json.loads((ROOT / "fixtures" / "sample_cases.json").read_text(encoding="utf-8"))]
    with tempfile.TemporaryDirectory() as workdir, ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda c: extract(c, args.model, args.effort, workdir), cases))

    records = {c.case_id: ex for c, (ex, _) in zip(cases, results) if ex is not None}
    args.out.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    models = sorted({meta["model"] for _, meta in results})
    cost = sum(meta["cost_usd"] for _, meta in results)
    print(f"recorded {len(records)}/{len(cases)} cases on {date.today()} -> {args.out}")
    print(f"model: {', '.join(models)}  effort: {args.effort}  cost: ${cost:.2f}")
    for c, (ex, meta) in zip(cases, results):
        if ex is None:
            print(f"  FAILED {c.case_id}: {meta['error']}")


if __name__ == "__main__":
    main()
