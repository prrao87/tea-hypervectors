#!/usr/bin/env python3
"""Find similar teas and explain the nearest neighbour's score.

    uv run src/search.py
    uv run src/search.py --tea-id 1314971975789

The stored float16 bundles are searched with LanceDB cosine. The contribution
breakdown then re-encodes the neighbours locally so we can attribute each score
to the fields that produced it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lancedb
import polars as pl

from hdc.dataset import DEFAULT_CACHE, HF_TABLE, HF_URI, load_teas, warm_embedder
from hdc.encoder import COMPONENTS, EncodedTea, TeaEncoder
from hdc.manifest import EncoderManifest
from hdc.similarity import component_contributions
from hdc.storage import VECTOR_COLUMN

DEFAULT_DATABASE = Path(__file__).resolve().parents[1] / "data" / "tea-hv-db"
DEFAULT_TABLE = "hypervectors"


def encode_all(
    manifest: EncoderManifest, teas: pl.DataFrame, cache: Path
) -> tuple[TeaEncoder, dict[int, EncodedTea]]:
    encoder = TeaEncoder(manifest, warm_embedder(manifest, teas, cache))
    return encoder, {record["id"]: encoder.encode(record) for record in teas.to_dicts()}


def lancedb_neighbours(database: Path, table_name: str, query, limit: int) -> list[dict]:
    """Cosine search against the persisted float16 column.

    The float32 query vector is fine here: LanceDB widens it to compare against
    the stored halves. Nothing downstream does arithmetic in float16.
    """
    table = lancedb.connect(database).open_table(table_name)
    query_values = query.detach().cpu().numpy()
    return (
        table.search(query_values, vector_column_name=VECTOR_COLUMN)
        .metric("cosine")
        .select(["id", "title", "class", "_distance"])
        .limit(limit)
        .to_list()
    )


def show_neighbours(args: argparse.Namespace, teas: pl.DataFrame) -> None:
    manifest = EncoderManifest()
    _, encodings = encode_all(manifest, teas, args.cache)
    titles = dict(zip(teas["id"], teas["title"], strict=True))

    seed_id = args.tea_id or int(teas["id"][0])
    seed = encodings[seed_id]
    print(f"\nSeed: {titles[seed_id]}  (id={seed_id})")
    print(f"Present components: {', '.join(seed.present)}\n")

    hits = lancedb_neighbours(args.database, args.table_out, seed.bundle, args.limit + 1)
    header = f"{'similarity':>10}  {'class':<8} title"
    print(header)
    print("-" * (len(header) + 20))
    for hit in hits:
        if hit["id"] == seed_id:
            continue  # a tea is always its own nearest neighbour
        print(f"{1 - hit['_distance']:>10.4f}  {hit['class']:<8} {hit['title']}")

    # LanceDB returns a score; the breakdown explains where the score came from.
    best = next(hit for hit in hits if hit["id"] != seed_id)
    contributions = component_contributions(seed.bundle, encodings[best["id"]])
    print(f"\nWhy '{best['title']}' scored {1 - best['_distance']:.4f}:")
    for name in COMPONENTS:
        if name in contributions:
            share = contributions[name]
            bar = "#" * max(0, round(share * 60))
            print(f"  {name:<10} {share:+.4f}  {bar}")
    print(f"  {'total':<10} {sum(contributions.values()):+.4f}  (equals the cosine score)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tea-id", type=int, help="Seed tea; defaults to the first row")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--source", default=HF_URI)
    parser.add_argument("--table-in", default=HF_TABLE)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--table-out", default=DEFAULT_TABLE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    teas = load_teas(args.source, args.table_in)
    show_neighbours(args, teas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
