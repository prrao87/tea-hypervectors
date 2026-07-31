"""Ingest vetted tea metadata and normalized local images into LanceDB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lancedb
import polars as pl
from normalize_images import verify_jpeg
from tea_data import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_INPUT,
    INGESTION_COLUMNS,
    normalized_image_path,
    read_ingestion_metadata,
)

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE = DATA_DIR / "tea-db"
DEFAULT_TABLE = "tea-db"
NULLABLE_COLUMNS = {"country"}


def load_products(path: Path, image_dir: Path, limit: int = 0) -> pl.DataFrame:
    """Build the typed Polars frame passed directly to LanceDB."""
    metadata = read_ingestion_metadata(path, limit)
    if metadata.is_empty():
        raise ValueError("No products selected for ingestion")

    duplicate_ids = metadata.filter(pl.col("id").is_duplicated()).get_column("id")
    if not duplicate_ids.is_empty():
        raise ValueError(f"Duplicate product IDs: {duplicate_ids.to_list()}")

    invalid_nulls = {
        column: metadata.get_column(column).null_count()
        for column in metadata.columns
        if column not in NULLABLE_COLUMNS and metadata.get_column(column).null_count()
    }
    if invalid_nulls:
        raise ValueError(f"Required source fields contain nulls: {invalid_nulls}")

    image_paths = [
        normalized_image_path(image_dir, int(product_id))
        for product_id in metadata.get_column("id")
    ]
    missing_images = [str(image_path) for image_path in image_paths if not image_path.exists()]
    if missing_images:
        preview = "\n".join(missing_images[:10])
        remainder = len(missing_images) - min(len(missing_images), 10)
        suffix = f"\n... and {remainder} more" if remainder else ""
        raise FileNotFoundError(
            "Normalized images are missing. Run `uv run data/download_images.py` "
            f"before ingestion:\n{preview}{suffix}"
        )

    images: list[bytes] = []
    for image_path in image_paths:
        verify_jpeg(image_path)
        images.append(image_path.read_bytes())

    return (
        metadata.with_columns(pl.Series("image", images, dtype=pl.Binary))
        .drop("image_url")
        .select(INGESTION_COLUMNS)
    )


def write_table(
    products: pl.DataFrame,
    database_path: Path,
    table_name: str,
    overwrite: bool,
) -> tuple[Any, int]:
    """Write one materialized Polars frame and optimize the local table."""
    database_path.mkdir(parents=True, exist_ok=True)
    database = lancedb.connect(database_path)
    mode = "overwrite" if overwrite else "create"
    table = database.create_table(table_name, data=products, mode=mode)
    table.optimize()
    return table, table.count_rows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Ingest only the first N products; zero means all products.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing local table.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or positive")
    return args


def main() -> int:
    args = parse_args()
    products = load_products(args.input, args.image_dir, args.limit)
    table, row_count = write_table(
        products,
        args.database,
        args.table,
        args.overwrite,
    )
    sample = (
        table.search()
        .select(
            [
                "id",
                "class",
                "title",
                "region",
                "elevation_meters",
                "elevation_confidence",
            ]
        )
        .limit(min(3, row_count))
        .to_list()
    )
    print(
        json.dumps(
            {
                "database": str(args.database.resolve()),
                "table": args.table,
                "rows": row_count,
                "columns": table.schema.names,
                "schema": str(table.schema),
                "sample": sample,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
