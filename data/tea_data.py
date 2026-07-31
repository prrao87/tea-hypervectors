"""Shared Polars loading helpers for the final tea dataset."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import polars as pl

DATA_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DATA_DIR.parent
DEFAULT_INPUT = DATA_DIR / "extracted" / "tea_data_final.json"
DEFAULT_IMAGE_DIR = PROJECT_DIR / "img"

INGESTION_COLUMNS = (
    "id",
    "source_url",
    "image",
    "class",
    "title",
    "description",
    "country",
    "region",
    "oxidation",
    "roast",
    "aroma",
    "taste",
    "elevation_meters",
    "elevation_confidence",
)


def read_products(path: Path) -> pl.DataFrame:
    """Read and flatten the top-level products array with Polars."""
    document = pl.read_json(path)
    if "products" not in document.columns:
        raise ValueError(f"Expected a top-level products array in {path}")
    return document.select("products").explode("products").unnest("products")


def read_ingestion_metadata(path: Path, limit: int = 0) -> pl.DataFrame:
    """Project and type the source fields needed for LanceDB ingestion."""
    products = read_products(path).select(
        pl.col("id").cast(pl.Int64),
        pl.col("source_url").cast(pl.String),
        pl.col("image_url").cast(pl.String),
        pl.col("category").cast(pl.String).alias("class"),
        pl.col("title").cast(pl.String),
        pl.col("description").cast(pl.String),
        pl.col("country").cast(pl.String),
        pl.col("region").cast(pl.String),
        pl.col("oxidation").cast(pl.String),
        pl.col("roast").cast(pl.String),
        pl.col("aroma").cast(pl.List(pl.String)),
        pl.col("taste").cast(pl.List(pl.String)),
        pl.col("elevation_meters").cast(pl.Int64),
        pl.col("elevation_confidence").cast(pl.Float64),
    )
    if limit:
        products = products.head(limit)
    return products


def image_extension(url: str) -> str:
    extension = Path(urlsplit(url).path).suffix.lower()
    if extension in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}:
        return extension
    return ".img"


def downloaded_image_path(image_dir: Path, product_id: int, image_url: str) -> Path:
    return image_dir / f"{product_id}{image_extension(image_url)}"


def normalized_image_path(image_dir: Path, product_id: int) -> Path:
    return image_dir / f"{product_id}.jpg"
