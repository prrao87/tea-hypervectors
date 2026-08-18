"""Load the small metadata subset exposed by the interactive visual."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recommender.catalog import TeaCatalog

DEFAULT_METADATA = (
    Path(__file__).resolve().parents[3] / "data" / "extracted" / "tea_data_final.json"
)


@dataclass(frozen=True, slots=True)
class TeaMetadata:
    tea_id: int
    title: str
    tea_class: str
    country: str | None
    region: str | None
    oxidation: str | None
    roast: str | None
    aroma: tuple[str, ...]
    taste: tuple[str, ...]

    def as_artifact_record(self) -> dict[str, str | list[str] | None]:
        return {
            "id": str(self.tea_id),
            "title": self.title,
            "class": self.tea_class,
            "country": self.country,
            "region": self.region,
            "oxidation": self.oxidation,
            "roast": self.roast,
            "aroma": list(self.aroma),
            "taste": list(self.taste),
        }


def _optional_text(product: dict[str, Any], field: str) -> str | None:
    value = product.get(field)
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _phrases(product: dict[str, Any], field: str) -> tuple[str, ...]:
    value = product.get(field)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"metadata field {field!r} must be a list of strings")
    return tuple(item.strip() for item in value if item.strip())


def load_metadata(
    catalog: TeaCatalog,
    path: Path = DEFAULT_METADATA,
) -> tuple[TeaMetadata, ...]:
    """Return display metadata in exactly the same order as the catalogue."""
    document = json.loads(path.read_text(encoding="utf-8"))
    products = document.get("products")
    if not isinstance(products, list):
        raise TypeError("metadata document must contain a products list")

    by_id = {int(product["id"]): product for product in products}
    missing = [tea_id for tea_id in catalog.ids if tea_id not in by_id]
    if missing:
        raise ValueError(f"metadata is missing {len(missing)} catalogue tea IDs")

    records: list[TeaMetadata] = []
    for index, tea_id in enumerate(catalog.ids):
        product = by_id[tea_id]
        title = str(product["title"])
        tea_class = str(product["category"])
        if title != catalog.titles[index] or tea_class != catalog.classes[index]:
            raise ValueError(f"metadata does not match encoded catalogue tea {tea_id}")
        records.append(
            TeaMetadata(
                tea_id=tea_id,
                title=title,
                tea_class=tea_class,
                country=_optional_text(product, "country"),
                region=_optional_text(product, "region"),
                oxidation=_optional_text(product, "oxidation"),
                roast=_optional_text(product, "roast"),
                aroma=_phrases(product, "aroma"),
                taste=_phrases(product, "taste"),
            )
        )
    return tuple(records)
