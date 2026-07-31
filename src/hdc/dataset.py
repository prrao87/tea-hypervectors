"""Load the typed tea columns that feed the encoder."""

from __future__ import annotations

from pathlib import Path

import lancedb
import polars as pl

from hdc.manifest import EncoderManifest
from hdc.sensory import PhraseEmbedder, canonicalize

HF_URI = "hf://datasets/prrao87/tea-hypervectors/data"
HF_TABLE = "train"

# The binary `image` column is deliberately absent: images are application
# metadata, not encoder input. Reading them here would move megabytes the
# encoder never looks at.
ENCODER_COLUMNS = [
    "id",
    "title",
    "aroma",
    "taste",
    "class",
    "oxidation",
    "roast",
    "elevation_meters",
    "elevation_confidence",
]

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "phrase_embeddings.json"


def load_teas(uri: str = HF_URI, table_name: str = HF_TABLE, limit: int = 1_000) -> pl.DataFrame:
    table = lancedb.connect(uri).open_table(table_name)
    return table.search().select(ENCODER_COLUMNS).limit(limit).to_polars()


def warm_embedder(
    manifest: EncoderManifest,
    teas: pl.DataFrame,
    cache_path: Path = DEFAULT_CACHE,
) -> PhraseEmbedder:
    """Embed every canonicalized phrase in the dataset once, then cache it.

    Batching here keeps the encoder itself free of network concerns: by the
    time it runs, every phrase lookup is a local dictionary hit.
    """
    embedder = PhraseEmbedder(manifest, cache_path)
    phrases: set[str] = set()
    for row in teas.iter_rows(named=True):
        phrases.update(canonicalize(row["aroma"]))
        phrases.update(canonicalize(row["taste"]))
    embedder.warm(sorted(phrases))
    embedder.save()
    return embedder
