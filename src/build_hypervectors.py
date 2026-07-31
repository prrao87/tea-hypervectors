#!/usr/bin/env python3
"""Encode every tea into a typed hypervector and persist it to LanceDB.

    uv run src/build_hypervectors.py

Reads the typed columns from the public Lance dataset, embeds the sensory
phrases with Ollama, builds the weighted MAP bundle, and writes it as a
fixed-size float16 vector column alongside the readable manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lancedb
import pyarrow as pa

from hdc.dataset import DEFAULT_CACHE, HF_TABLE, HF_URI, load_teas, warm_embedder
from hdc.encoder import TeaEncoder, assert_encoder_invariants
from hdc.manifest import EncoderManifest
from hdc.storage import arrow_schema, to_row

DEFAULT_DATABASE = Path(__file__).resolve().parents[1] / "data" / "tea-hv-db"
DEFAULT_TABLE = "hypervectors"


def build(args: argparse.Namespace) -> dict:
    manifest = EncoderManifest()
    teas = load_teas(args.source, args.table_in, args.limit)
    embedder = warm_embedder(manifest, teas, args.cache)
    encoder = TeaEncoder(manifest, embedder)

    rows = []
    for record in teas.to_dicts():
        encoded = encoder.encode(record)
        assert_encoder_invariants(encoded, manifest)
        rows.append(to_row(encoded, record))

    args.database.mkdir(parents=True, exist_ok=True)
    database = lancedb.connect(args.database)
    schema = arrow_schema(manifest)
    table = database.create_table(
        args.table_out,
        data=pa.Table.from_pylist(rows, schema=schema),
        mode="overwrite",
    )
    table.optimize()

    manifest_path = args.database / f"{args.table_out}-manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

    return {
        "rows": table.count_rows(),
        "vector_column_type": str(table.schema.field("vector_raw").type),
        "weight_budget": round(manifest.weights.total, 6),
        "manifest_written_to": str(manifest_path),
        "database": str(args.database),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=HF_URI, help="LanceDB URI holding the typed columns")
    parser.add_argument("--table-in", default=HF_TABLE)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--table-out", default=DEFAULT_TABLE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=1_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(build(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
