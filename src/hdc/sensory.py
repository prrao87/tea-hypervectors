"""Canonicalize aroma/taste phrases and embed them with Ollama's Nomic model."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import ollama

from hdc.manifest import EncoderManifest

WHITESPACE = re.compile(r"\s+")


def canonicalize(phrases: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize, deduplicate, and sort a sensory phrase list.

    Sorting makes the field order-independent: the same descriptors written in
    a different order must produce the same hypervector. An empty result means
    the field is genuinely missing, which is not the same as an empty string.
    """
    if phrases is None:
        return ()
    cleaned = {
        WHITESPACE.sub(" ", unicodedata.normalize("NFKC", phrase)).strip().lower()
        for phrase in phrases
    }
    return tuple(sorted(phrase for phrase in cleaned if phrase))


class PhraseEmbedder:
    """Phrase-level Nomic embeddings, cached on disk.

    Each phrase is embedded on its own and the field embedding is their mean.
    This keeps the result independent of phrase order and list length.
    """

    def __init__(self, manifest: EncoderManifest, cache_path: Path) -> None:
        self.manifest = manifest
        self.cache_path = cache_path
        self._cache = self._load_cache()
        self._dirty = False

    def _load_cache(self) -> dict[str, list[float]]:
        if not self.cache_path.exists():
            return {}
        payload = json.loads(self.cache_path.read_text())
        # The cache is keyed by phrase only, so it must be discarded whenever
        # the model or the task prefix changes.
        if payload.get("model") != self.manifest.embedding_model:
            return {}
        if payload.get("task_prefix") != self.manifest.task_prefix:
            return {}
        return payload["phrases"]

    def save(self) -> None:
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {
                    "model": self.manifest.embedding_model,
                    "task_prefix": self.manifest.task_prefix,
                    "phrases": self._cache,
                }
            )
        )
        self._dirty = False

    def warm(self, phrases: Sequence[str], batch_size: int = 64) -> int:
        """Embed and cache every phrase not already known. Returns the new count."""
        missing = sorted({phrase for phrase in phrases if phrase not in self._cache})
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            response = ollama.embed(
                model=self.manifest.embedding_model,
                input=[self.manifest.task_prefix + phrase for phrase in batch],
            )
            for phrase, embedding in zip(batch, response.embeddings, strict=True):
                self._cache[phrase] = list(embedding)
        self._dirty = self._dirty or bool(missing)
        return len(missing)

    def phrase_embedding(self, phrase: str) -> np.ndarray:
        if phrase not in self._cache:
            self.warm([phrase])
        return np.asarray(self._cache[phrase], dtype=np.float32)

    def field_embedding(self, phrases: Sequence[str]) -> np.ndarray | None:
        """Mean of the phrase vectors. None if empty."""
        if not phrases:
            return None
        stacked = np.stack([self.phrase_embedding(phrase) for phrase in phrases])
        return stacked.mean(axis=0)
