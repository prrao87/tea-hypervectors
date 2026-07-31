#!/usr/bin/env python3
"""Run a bounded full-text search against the local tea LanceDB table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lancedb

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE = DATA_DIR / "tea-db"
DEFAULT_TABLE = "tea-db"
DEFAULT_QUERY = "roasted oolong high mountain"
FTS_COLUMN = "description"
FTS_INDEX_NAME = "description_fts"
TEA_CLASSES = ("oolong", "green", "white", "black", "yellow")

SETUP_INSTRUCTIONS = """Run the data preparation and indexing steps first:
  uv run data/download_images.py
  uv run data/ingest.py
  uv run data/index.py
Then run the query:
  uv run data/query.py"""


def open_table(database_path: Path, table_name: str):
    if not database_path.exists():
        raise RuntimeError(
            f"LanceDB database does not exist at {database_path}.\n{SETUP_INSTRUCTIONS}"
        )

    try:
        database = lancedb.connect(database_path)
        return database.open_table(table_name)
    except Exception as error:
        raise RuntimeError(
            f"Could not open LanceDB table {table_name!r} at {database_path}.\n{SETUP_INSTRUCTIONS}"
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--class",
        dest="tea_class",
        choices=TEA_CLASSES,
        help="Optionally restrict results to one requested tea class.",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    try:
        table = open_table(args.database, args.table)
        index_names = {index.name for index in table.list_indices()}
        if FTS_INDEX_NAME not in index_names:
            raise RuntimeError(
                f"FTS index {FTS_INDEX_NAME!r} does not exist on table {args.table!r}."
            )
        search = table.search(
            args.query,
            query_type="fts",
            fts_columns=FTS_COLUMN,
        ).select(
            [
                "id",
                "title",
                "class",
                "description",
                "country",
                "region",
                "elevation_meters",
                "source_url",
                "_score",
            ]
        )
        if args.tea_class:
            search = search.where(f"class = '{args.tea_class}'")
        results = search.limit(args.limit).to_list()
    except Exception as error:
        message = str(error)
        if SETUP_INSTRUCTIONS not in message:
            message = (
                "Could not run the FTS query. The table or FTS index may be missing.\n"
                f"{SETUP_INSTRUCTIONS}\n"
                f"Cause: {error}"
            )
        raise SystemExit(message) from error

    print(
        json.dumps(
            {
                "query": args.query,
                "class_filter": args.tea_class,
                "count": len(results),
                "results": results,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
