#!/usr/bin/env python3
"""Create search indexes for the local tea LanceDB table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lancedb
from lancedb.index import FTS

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE = DATA_DIR / "tea-db"
DEFAULT_TABLE = "tea-db"
FTS_COLUMN = "description"
FTS_INDEX_NAME = "description_fts"

SETUP_INSTRUCTIONS = """Run the data preparation steps first:
  uv run data/download_images.py
  uv run data/ingest.py
Then create the index:
  uv run data/index.py"""


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
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument(
        "--replace",
        action="store_true",
        help=f"Replace the existing {FTS_INDEX_NAME!r} index.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        table = open_table(args.database, args.table)
        if FTS_COLUMN not in table.schema.names:
            raise RuntimeError(
                f"Table {args.table!r} has no {FTS_COLUMN!r} column.\n{SETUP_INSTRUCTIONS}"
            )

        table.create_index(
            FTS_COLUMN,
            config=FTS(
                language="English",
                lower_case=True,
                stem=True,
                remove_stop_words=True,
                ascii_folding=True,
            ),
            name=FTS_INDEX_NAME,
            replace=args.replace,
        )
    except Exception as error:
        message = str(error)
        if SETUP_INSTRUCTIONS not in message:
            message = (
                f"Could not create the {FTS_INDEX_NAME!r} FTS index. "
                "If it already exists, rerun with --replace.\n"
                f"{SETUP_INSTRUCTIONS}\n"
                f"Cause: {error}"
            )
        raise SystemExit(message) from error

    print(
        json.dumps(
            {
                "database": str(args.database.resolve()),
                "table": args.table,
                "index": FTS_INDEX_NAME,
                "column": FTS_COLUMN,
                "type": "FTS",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
