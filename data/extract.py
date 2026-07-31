#!/usr/bin/env python3
"""Create a flat tea catalog export from Cha Yi's localized product pages.

Collection JSON is used only to discover product handles. Each persisted record
comes from the public product page requested with ``locale=en``. The extractor
keeps the rendered English title and description, the page's localized product
metadata, and a small flat schema suitable for later ontology enrichment.

Every request to shop.chayi.ca starts at least one second after the preceding
request. This script uses public storefront endpoints only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, build_opener

BASE_URL = "https://shop.chayi.ca"
REQUEST_DELAY_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 30
USER_AGENT = (
    "ChaYiTeaResearchDataset/2.0 (non-commercial research; respectful 1-second request interval)"
)


@dataclass(frozen=True)
class CollectionSpec:
    key: str
    handle: str
    name: str


COLLECTIONS = (
    CollectionSpec("oolong", "thes-oolongs", "Oolong teas"),
    CollectionSpec("green", "thes-verts", "Green teas"),
    CollectionSpec("white", "thes-blancs", "White teas"),
    CollectionSpec("black", "thes-noirs", "Black teas"),
    CollectionSpec("yellow", "thes-jaunes", "Yellow teas"),
)
COLLECTIONS_BY_KEY = {collection.key: collection for collection in COLLECTIONS}

# Localized product-page tags generally contain the English country name. The
# vendor fallback covers products whose tags omit it, without rejecting or
# otherwise changing French source text.
COUNTRY_ALIASES = {
    "china": "China",
    "chine": "China",
    "india": "India",
    "inde": "India",
    "japan": "Japan",
    "japon": "Japan",
    "kenya": "Kenya",
    "korea": "South Korea",
    "corée": "South Korea",
    "nepal": "Nepal",
    "népal": "Nepal",
    "rwanda": "Rwanda",
    "sri lanka": "Sri Lanka",
    "taiwan": "Taiwan",
    "vietnam": "Vietnam",
}


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", unescape(value)).strip()


def localized_url(url: str, **params: str | int) -> str:
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({key: str(value) for key, value in params.items()})
    query["locale"] = "en"
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )


def absolute_storefront_url(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("//"):
        return f"https:{value}"
    if value.startswith("/"):
        return f"{BASE_URL}{value}"
    return value


class ProductPageParser(HTMLParser):
    """Extract rendered text and the localized product JSON from a product page."""

    _VOID_TAGS: ClassVar[frozenset[str]] = frozenset(
        {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.in_h1 = False
        self.h1_parts: list[str] = []
        self.description_root_depth: int | None = None
        self.description_parts: list[str] = []
        self.in_product_json = False
        self.product_json_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag not in self._VOID_TAGS:
            self.stack.append(tag)

        if tag == "h1" and not self.h1_parts:
            self.in_h1 = True

        class_name = attrs_dict.get("class") or ""
        if self.description_root_depth is None and "product-single__description" in class_name:
            self.description_root_depth = len(self.stack)

        if tag == "script" and attrs_dict.get("id") == "ProductJson-product-template":
            self.in_product_json = True

        if self.description_root_depth is not None and tag in {
            "br",
            "p",
            "div",
            "li",
        }:
            self.description_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1":
            self.in_h1 = False
        if tag == "script" and self.in_product_json:
            self.in_product_json = False
        if (
            self.description_root_depth is not None
            and len(self.stack) == self.description_root_depth
        ):
            self.description_root_depth = None
        if self.stack:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        if self.in_h1:
            self.h1_parts.append(data)
        if self.description_root_depth is not None:
            self.description_parts.append(data)
        if self.in_product_json:
            self.product_json_parts.append(data)

    def title(self) -> str:
        return clean_text("".join(self.h1_parts))

    def description(self) -> str:
        return clean_text("".join(self.description_parts))

    def product_json(self) -> dict[str, Any]:
        raw_value = "".join(self.product_json_parts).strip()
        if not raw_value:
            raise ValueError("Product page did not contain ProductJson-product-template")
        value = json.loads(raw_value)
        if not isinstance(value, dict):
            raise TypeError("Embedded product JSON was not an object")
        return value


class RespectfulHttpClient:
    """A small HTTP client that enforces a global minimum interval per request."""

    def __init__(self, delay_seconds: float = REQUEST_DELAY_SECONDS) -> None:
        self.delay_seconds = delay_seconds
        self.last_request_started_at: float | None = None
        self.opener = build_opener()
        self.request_count = 0

    def _wait_for_turn(self) -> None:
        if self.last_request_started_at is None:
            return
        remaining = self.delay_seconds - (time.monotonic() - self.last_request_started_at)
        if remaining > 0:
            time.sleep(remaining)

    def get_bytes(self, url: str, *, retries: int = 3) -> bytes:
        for attempt in range(retries):
            self._wait_for_turn()
            self.last_request_started_at = time.monotonic()
            self.request_count += 1
            request = Request(
                url,
                headers={
                    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-CA,en;q=0.9",
                    "User-Agent": USER_AGENT,
                },
            )
            try:
                with self.opener.open(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
                    return response.read()
            except HTTPError as error:
                retryable = error.code in {429, 500, 502, 503, 504}
                if not retryable or attempt == retries - 1:
                    raise RuntimeError(f"HTTP {error.code} while fetching {url}") from error
                retry_after = error.headers.get("Retry-After")
                wait_seconds = max(
                    15.0 * (attempt + 1),
                    float(retry_after) if retry_after and retry_after.isdigit() else 0.0,
                )
                print(
                    f"Received HTTP {error.code}; waiting {wait_seconds:.0f}s "
                    "before one gentle retry.",
                    file=sys.stderr,
                )
                time.sleep(wait_seconds)
            except URLError as error:
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"Network error while fetching {url}: {error.reason}"
                    ) from error
                time.sleep(15.0 * (attempt + 1))
        raise AssertionError("Unreachable")

    def get_text(self, url: str) -> str:
        return self.get_bytes(url).decode("utf-8", errors="replace")

    def get_json(self, url: str) -> dict[str, Any]:
        try:
            value = json.loads(self.get_text(url))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Expected JSON from {url}") from error
        if not isinstance(value, dict):
            raise TypeError(f"Expected a JSON object from {url}")
        return value


def country_from_product(product: dict[str, Any]) -> str | None:
    tags = product.get("tags")
    if isinstance(tags, list):
        for raw_tag in tags:
            country = COUNTRY_ALIASES.get(clean_text(str(raw_tag)).casefold())
            if country:
                return country

    vendor = clean_text(str(product.get("vendor") or ""))
    if " - " in vendor:
        country = COUNTRY_ALIASES.get(vendor.rsplit(" - ", maxsplit=1)[-1].casefold())
        if country:
            return country
    return None


def product_image_url(product: dict[str, Any]) -> str | None:
    featured_image = product.get("featured_image")
    if isinstance(featured_image, str):
        return absolute_storefront_url(featured_image)

    images = product.get("images")
    if isinstance(images, list) and images and isinstance(images[0], str):
        return absolute_storefront_url(images[0])
    return None


def normalize_product(
    page_html: str,
    *,
    source_url: str,
    category: str,
) -> dict[str, str | int | None]:
    parser = ProductPageParser()
    parser.feed(page_html)
    product = parser.product_json()

    title = parser.title() or clean_text(str(product.get("title") or ""))
    description = parser.description()
    if not description:
        description = clean_text(str(product.get("description") or product.get("content") or ""))

    product_id = product.get("id")
    if product_id is None or not title or not description:
        raise ValueError(f"Product page {source_url} is missing id, title, or description")

    return {
        "id": int(product_id),
        "source_url": source_url,
        "category": category,
        "title": title,
        "description": description,
        "country": country_from_product(product),
        "image_url": product_image_url(product),
    }


def discover_handles(
    client: RespectfulHttpClient,
    collections: list[CollectionSpec],
    limit_per_collection: int,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    selected: list[tuple[str, str]] = []
    collection_counts: dict[str, int] = {}
    seen_handles: set[str] = set()

    for collection in collections:
        endpoint = localized_url(
            f"{BASE_URL}/collections/{collection.handle}/products.json",
            limit=250,
        )
        products = client.get_json(endpoint).get("products")
        if not isinstance(products, list):
            raise TypeError(f"Expected a products array from {endpoint}")

        collection_counts[collection.key] = len(products)
        selected_products = products[:limit_per_collection] if limit_per_collection else products
        for product in selected_products:
            if not isinstance(product, dict) or not product.get("handle"):
                continue
            handle = str(product["handle"])
            if handle in seen_handles:
                raise ValueError(f"Product {handle!r} belongs to more than one selected collection")
            seen_handles.add(handle)
            selected.append((handle, collection.key))

        print(
            f"Discovered {len(products):3d} {collection.name}; "
            f"selected {len(selected_products):3d}.",
            file=sys.stderr,
        )

    return selected, collection_counts


def build_dataset(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    client = RespectfulHttpClient(args.delay)
    started_at = datetime.now(UTC)
    collections = [COLLECTIONS_BY_KEY[key] for key in args.collections]
    selections, collection_counts = discover_handles(
        client,
        collections,
        args.limit_per_collection,
    )

    total_limit = 2 if args.limit is None and not args.limit_per_collection else args.limit
    if total_limit:
        selections = selections[:total_limit]

    products: list[dict[str, str | int | None]] = []
    for index, (handle, category) in enumerate(selections, start=1):
        source_url = f"{BASE_URL}/products/{handle}"
        page_html = client.get_text(localized_url(source_url))
        product = normalize_product(
            page_html,
            source_url=source_url,
            category=category,
        )
        products.append(product)
        print(
            f"Fetched {index:3d}/{len(selections)}: {handle}",
            file=sys.stderr,
        )

    finished_at = datetime.now(UTC)
    null_counts = {
        field: sum(product[field] is None for product in products)
        for field in ("country", "image_url")
    }
    dataset = {
        "dataset": "Canadian Loose-Leaf Tea Profiles",
        "retrieved_at": finished_at.isoformat(),
        "scope": {
            "collections": [collection.key for collection in collections],
            "collection_counts": collection_counts,
            "products_exported": len(products),
            "limited_run": len(products) < sum(collection_counts.values()),
        },
        "products": products,
    }
    manifest = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 2),
        "request_count": client.request_count,
        "collection_counts": collection_counts,
        "products_exported": len(products),
        "minimum_seconds_between_all_requests": args.delay,
        "null_counts": null_counts,
    }
    return dataset, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collections",
        nargs="+",
        choices=tuple(COLLECTIONS_BY_KEY),
        default=[collection.key for collection in COLLECTIONS],
        help="Collections to export, in the requested order.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Total products to export. Defaults to 2; zero means all.",
    )
    parser.add_argument(
        "--limit-per-collection",
        type=int,
        default=0,
        help="Export at most this many products from each selected collection.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY_SECONDS,
        help="Minimum seconds between requests; must be at least 1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Dataset JSON destination.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest JSON destination.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be zero or positive")
    if args.limit_per_collection < 0:
        parser.error("--limit-per-collection must be zero or positive")
    if args.limit is not None and args.limit and args.limit_per_collection:
        parser.error("use either --limit or --limit-per-collection, not both")
    if args.delay < 1:
        parser.error("--delay must be at least 1 second")

    extracted_dir = Path(__file__).resolve().parent / "extracted"
    if args.output is None:
        args.output = extracted_dir / "tea_data.json"
    if args.manifest is None:
        args.manifest = args.output.with_name("summary.json")
    return args


def main() -> int:
    args = parse_args()
    dataset, manifest = build_dataset(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}", file=sys.stderr)
    print(f"Wrote {args.manifest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
