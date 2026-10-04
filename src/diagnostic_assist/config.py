"""Tunable parameters of the labelling pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExtractorConfig:
    # Strongest model for the one-off backfill: extraction errors propagate everywhere.
    model: str = field(default_factory=lambda: os.environ.get("EXTRACT_MODEL", ""))
    effort: str = "low"
    max_tokens: int = 16000
    version: str = "extract-v1"


@dataclass(frozen=True)
class LabelingConfig:
    # Complete-linkage cut. On the sample, 0.1-0.8 is safe with the parts rule and 0.2-0.5
    # on text alone (scripts/evaluate_against_gold.py); 0.4 also holds for no-part fixes.
    merge_threshold: float = 0.4
    parts_agree_discount: float = 0.5  # distance *= 1 - discount * jaccard(fix parts)
    use_parts: bool = True  # off only for the ablation
    assignment_margin: float = 0.05  # runner-up closer than this: ambiguous, not assigned
    min_parts_purity: float = 0.8
    language_split_alpha: float = 0.05
    min_cluster_size_for_language_check: int = 3
