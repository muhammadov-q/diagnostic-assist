"""Text -> vector for cause mentions.

``LexicalEmbedder`` runs offline and works because extraction already reduced each cause
to a short English noun phrase. ``BedrockEmbedder`` is the production stub.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .normalize import head_noun


class Embedder(Protocol):
    version: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) float32 matrix of L2-normalised rows."""
        ...


class LexicalEmbedder:
    """Hashed bag of features: words (1.0), the head noun (2.0) and character 4-grams (0.5).

    The head noun separates "boom cylinder seal" (a seal) from "boom cylinder" (a cylinder);
    the prompt puts the failed component last. crc32 because ``hash()`` is salted per process.
    """

    version = "lexical-v1"

    def __init__(self, dim: int = 2**14):
        self._dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text):
                out[row, zlib.crc32(feature.encode()) % self._dim] += weight
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)

    @staticmethod
    def _features(text: str) -> list[tuple[str, float]]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        if not words:
            return []
        feats: list[tuple[str, float]] = [(f"w:{w}", 1.0) for w in words]
        feats.append((f"h:{head_noun(text)}", 2.0))
        for w in words:
            padded = f"<{w}>"
            feats += [(f"c:{padded[i:i + 4]}", 0.5) for i in range(max(1, len(padded) - 3))]
        return feats


class BedrockEmbedder:
    """Stub: Cohere Embed Multilingual v3 via boto3 ``bedrock-runtime`` (1024 dims, batches of 96).

    Embeds cause mentions and customer descriptions, DE/EN/FR/IT in one space. Vectors carry
    ``version``; a model change re-embeds everything rather than mixing versions.
    """

    version = "cohere-embed-multilingual-v3"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("Bedrock embedding call: see class docstring")
