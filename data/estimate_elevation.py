#!/usr/bin/env python3
"""Extract tea-growing regions and estimate their elevations with DSPy."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from tavily import AsyncTavilyClient

EXTRACTED_DIR = Path(__file__).with_name("extracted")
DEFAULT_INPUT = EXTRACTED_DIR / "tea_data_enriched.json"
DEFAULT_OUTPUT = EXTRACTED_DIR / "tea_data_final.json"
DEFAULT_SEARCH_CACHE = EXTRACTED_DIR / "elevation_search_cache.json"
DEFAULT_MODEL = "openrouter/google/gemini-3.5-flash-lite"
SEARCH_RESULTS_PER_LOCATION = 5
METERS_PER_FOOT = 0.3048

_ELEVATION_RANGE_PATTERN = re.compile(
    r"(?P<low>\d[\d,]*)\s*(?:to|[-–—])\s*(?P<high>\d[\d,]*)\s*"
    r"(?P<unit>feet|foot|ft|meters?|metres?|m)\b",
    re.IGNORECASE,
)
_ELEVATION_VALUE_PATTERN = re.compile(
    r"(?P<value>\d[\d,]*)\s*(?P<unit>feet|foot|ft|meters?|metres?|m)\b",
    re.IGNORECASE,
)
_ELEVATION_CONTEXT_PATTERN = re.compile(
    r"altitude|elevation|above sea level|high[- ]altitude|high[- ]mountain|"
    r"\bgrown\b|\bcultivated\b|\bproduced\b|\bplantation\b",
    re.IGNORECASE,
)
_APPROXIMATE_PATTERN = re.compile(
    r"\balmost\b|\babout\b|\bapproximately\b|\baround\b|\bnearly\b|\broughly\b",
    re.IGNORECASE,
)
_BOUND_PATTERN = re.compile(
    r"\bup to\b|\bas high as\b|\bat least\b|\bmore than\b|\bover\b|\bunder\b|"
    r"\bbelow\b",
    re.IGNORECASE,
)


class ElevationEstimate(BaseModel):
    """The validated geographic fields appended to each tea."""

    region: str
    elevation_basis: Literal["description", "tavily"]
    elevation_raw_meters: int
    elevation_meters: int
    elevation_confidence: float = Field(ge=0.0, le=1.0)
    elevation_description_evidence: str | None = None
    elevation_tool_call: dict[str, object] | None = None
    elevation_search_query: str | None = None
    elevation_search_results: list[dict[str, str | float]] = Field(default_factory=list)

    @field_validator("elevation_meters")
    @classmethod
    def round_elevation_down(cls, value: int) -> int:
        # Round down to the nearest multiple of 50 metres for consistency and
        # to avoid breaking the pipeline (approx elevations are sufficient).
        return value // 50 * 50


class ExtractGrowingRegion(dspy.Signature):
    """Extract the tea's most specific stated growing region.

    Follow these rules:

    - A region may be a province, prefecture, state, county, mountain, island,
      tea estate, town, or village.
    - Use the title when it supplies a geographic name omitted from the
      description.
    - Do not mistake a processing style, cultivar, tasting note, seller
      location, or comparison to another tea for the growing region.
    - For a blend, select the primary tea base's origin.
    - If no sub-country origin is stated, return the country as the broad
      region.
    """

    title: str = dspy.InputField(desc="The product's English title.")
    country: str = dspy.InputField(
        desc="The extracted country, or an empty string when unavailable."
    )
    description: str = dspy.InputField(desc="The complete English product description.")
    region: str = dspy.OutputField(
        desc="The most specific stated growing region, as a concise place name."
    )


class RequestElevationEvidence(dspy.Signature):
    """Request web evidence for a tea-growing location.

    Follow these rules:

    - Call search_elevation exactly once.
    - Pass the supplied region and country verbatim.
    - Do not answer the elevation question yourself.
    - Do not call any other tool.
    """

    tools: list[dspy.Tool] = dspy.InputField(desc="The only available tool, search_elevation.")
    region: str = dspy.InputField(desc="The extracted tea-growing region.")
    country: str = dspy.InputField(
        desc="The extracted country, or an empty string when unavailable."
    )
    tool_calls: dspy.ToolCalls = dspy.OutputField(
        desc="Exactly one search_elevation call for the supplied location."
    )


class EstimateElevationFromEvidence(dspy.Signature):
    """Estimate representative elevation from the supplied web evidence.

    Follow these rules:

    - A numeric elevation stated in the product description always takes
      precedence; do not invoke this fallback or Tavily for that product.
    - Preserve an approximate stated value such as "almost 2700 meters" as
      2700 metres.
    - For a stated elevation range, use its midpoint before rounding.
    - Do not infer an elevation from qualitative description phrases such as
      "high mountain"; use this evidence-backed fallback instead.
    - Estimate the location's typical altitude above sea level, not its highest
      summit or extreme point.
    - For a broad region such as Yunnan, estimate a representative regional
      average.
    - Prefer explicit elevations and ranges in relevant sources.
    - Reconcile multiple plausible sources rather than copying an outlier.
    - Return metres as an integer rounded to the nearest 50 or 100 metres.
    - Calibrate confidence from 0.0 to 1.0 based on source relevance,
      agreement, geographic specificity, and whether elevation was explicit
      rather than inferred.
    """

    region: str = dspy.InputField(desc="The extracted tea-growing region.")
    country: str = dspy.InputField(
        desc="The extracted country, or an empty string when unavailable."
    )
    search_results: list[dict[str, str | float]] = dspy.InputField(
        desc="Tavily result titles, URLs, snippets, and relevance scores."
    )
    elevation_meters: int = dspy.OutputField(
        desc=(
            "Representative elevation in metres; non-multiples are rounded down "
            "to the nearest multiple of 50 during validation."
        )
    )
    confidence: float = dspy.OutputField(
        desc="Confidence from 0.0 (unsupported) to 1.0 (strong explicit agreement)."
    )


class TavilySearchCache:
    """Execute and persist one-credit Tavily searches by normalized location."""

    def __init__(self, api_key: str, path: Path) -> None:
        self.client = AsyncTavilyClient(api_key=api_key)
        self.path = path
        self.entries = self._load_entries()
        self.lock = asyncio.Lock()

    def _load_entries(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(payload.get("entries"), dict):
            raise ValueError(f"Unsupported Tavily search cache format: {self.path}")
        return payload["entries"]

    async def search(self, region: str, country: str) -> list[dict[str, str | float]]:
        key = _location_key(region, country)
        async with self.lock:
            cached = self.entries.get(key)
        if cached is not None:
            print(f"Tavily cache hit: {_location_label(region, country)}", file=sys.stderr)
            return cached["results"]

        query = _elevation_search_query(region, country)
        response = await self.client.search(
            query=query,
            search_depth="basic",
            topic="general",
            max_results=SEARCH_RESULTS_PER_LOCATION,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
            auto_parameters=False,
        )
        results = [
            {
                "title": str(result.get("title", "")),
                "url": str(result.get("url", "")),
                "content": str(result.get("content", "")),
                "score": float(result.get("score", 0.0)),
            }
            for result in response.get("results", [])
        ]
        if not results:
            raise RuntimeError(f"Tavily returned no results for {_location_label(region, country)}")

        async with self.lock:
            self.entries[key] = {
                "region": region,
                "country": country,
                "query": query,
                "results": results,
            }
            self._write()
        print(f"Tavily search: {_location_label(region, country)}", file=sys.stderr)
        return results

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(
                {"schema_version": 1, "entries": self.entries},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.path)


class TeaElevationEstimator(dspy.Module):
    """Extract regions, search each unique location, and estimate elevation."""

    def __init__(self, search_cache: TavilySearchCache) -> None:
        super().__init__()
        self.extract_region = dspy.Predict(ExtractGrowingRegion)
        self.request_evidence = dspy.Predict(RequestElevationEvidence)
        self.estimate_elevation = dspy.Predict(EstimateElevationFromEvidence)

        async def search_elevation(region: str, country: str) -> list[dict[str, str | float]]:
            """Search for typical elevation evidence for a geographic region."""
            return await search_cache.search(region=region, country=country)

        self.search_tool = dspy.Tool(
            search_elevation,
            name="search_elevation",
            desc=(
                "Search the web for typical or average elevation evidence for a "
                "tea-growing region and country."
            ),
            arg_desc={
                "region": "The extracted tea-growing region.",
                "country": "The extracted country, or an empty string.",
            },
        )

    async def extract_product_region(self, product: dict) -> str:
        prediction = await self.extract_region.acall(
            title=product["title"],
            country=product.get("country") or "",
            description=product["description"],
        )
        region = prediction.region.strip()
        if not region:
            raise ValueError(f"Empty region extracted for {product['title']!r}")
        return region

    async def estimate_location(self, region: str, country: str) -> ElevationEstimate:
        requested = await self.request_evidence.acall(
            tools=[self.search_tool],
            region=region,
            country=country,
        )
        tool_call = _validated_tool_call(requested.tool_calls, region, country)
        search_results = await self.search_tool.acall(**tool_call.args)
        prediction = await self.estimate_elevation.acall(
            region=region,
            country=country,
            search_results=search_results,
        )
        return ElevationEstimate(
            region=region,
            elevation_basis="tavily",
            elevation_raw_meters=prediction.elevation_meters,
            elevation_meters=prediction.elevation_meters,
            elevation_confidence=prediction.confidence,
            elevation_tool_call={
                "name": tool_call.name,
                "args": tool_call.args,
            },
            elevation_search_query=_elevation_search_query(region, country),
            elevation_search_results=search_results,
        )


def _validated_tool_call(
    tool_calls: dspy.ToolCalls, region: str, country: str
) -> dspy.ToolCalls.ToolCall:
    calls = tool_calls.tool_calls
    if len(calls) != 1 or calls[0].name != "search_elevation":
        raise ValueError("Expected exactly one search_elevation tool call")
    expected = {"region": region, "country": country}
    if calls[0].args != expected:
        raise ValueError(
            f"Tool must use the extracted location verbatim: "
            f"expected {expected!r}, got {calls[0].args!r}"
        )
    return calls[0]


def _location_key(region: str, country: str) -> str:
    return "\0".join((region.strip().casefold(), country.strip().casefold()))


def _location_label(region: str, country: str) -> str:
    if country and country.casefold() != region.casefold():
        return f"{region}, {country}"
    return region


def _elevation_search_query(region: str, country: str) -> str:
    return (
        f"{_location_label(region, country)} typical average elevation "
        "altitude above sea level metres"
    )


def _description_elevation(description: str, region: str) -> ElevationEstimate | None:
    """Extract stronger product-specific elevation evidence before web search."""
    range_match = _ELEVATION_RANGE_PATTERN.search(description)
    if range_match and _has_elevation_context(description, range_match):
        low = _distance_in_meters(range_match.group("low"), range_match.group("unit"))
        high = _distance_in_meters(range_match.group("high"), range_match.group("unit"))
        raw_meters = int((low + high) / 2)
        return ElevationEstimate(
            region=region,
            elevation_basis="description",
            elevation_raw_meters=raw_meters,
            elevation_meters=raw_meters,
            elevation_confidence=0.90,
            elevation_description_evidence=range_match.group(0),
        )

    for value_match in _ELEVATION_VALUE_PATTERN.finditer(description):
        if not _has_elevation_context(description, value_match):
            continue
        context_before = description[max(0, value_match.start() - 40) : value_match.start()]
        confidence = 0.99
        evidence_start = value_match.start()
        approximate_match = _APPROXIMATE_PATTERN.search(context_before)
        bound_match = _BOUND_PATTERN.search(context_before)
        if approximate_match:
            confidence = 0.95
            evidence_start = value_match.start() - len(context_before) + approximate_match.start()
        elif bound_match:
            confidence = 0.80
            evidence_start = value_match.start() - len(context_before) + bound_match.start()
        raw_meters = int(_distance_in_meters(value_match.group("value"), value_match.group("unit")))
        return ElevationEstimate(
            region=region,
            elevation_basis="description",
            elevation_raw_meters=raw_meters,
            elevation_meters=raw_meters,
            elevation_confidence=confidence,
            elevation_description_evidence=description[evidence_start : value_match.end()],
        )
    return None


def _has_elevation_context(description: str, match: re.Match[str]) -> bool:
    context = description[max(0, match.start() - 100) : min(len(description), match.end() + 50)]
    return _ELEVATION_CONTEXT_PATTERN.search(context) is not None


def _distance_in_meters(value: str, unit: str) -> float:
    distance = float(value.replace(",", ""))
    if unit.casefold() in {"feet", "foot", "ft"}:
        return distance * METERS_PER_FOOT
    return distance


def _load_products(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    products = payload.get("products")
    if not isinstance(products, list):
        raise TypeError(f"{path} must contain a top-level 'products' list")
    return payload, products


def _select_products(
    products: list[dict], titles: list[str] | None, estimate_all: bool
) -> list[dict]:
    if estimate_all:
        return products
    if not titles:
        raise ValueError("Use --titles for a sample or --all for the complete dataset")

    products_by_title = {product["title"].casefold(): product for product in products}
    missing = [title for title in titles if title.casefold() not in products_by_title]
    if missing:
        raise ValueError(f"Product titles not found: {', '.join(missing)}")
    return [products_by_title[title.casefold()] for title in titles]


async def _extract_regions(
    estimator: TeaElevationEstimator,
    products: list[dict],
    semaphore: asyncio.Semaphore,
) -> list[tuple[dict, str, str, ElevationEstimate | None]]:
    async def extract(
        product: dict,
    ) -> tuple[dict, str, str, ElevationEstimate | None]:
        async with semaphore:
            region = await estimator.extract_product_region(product)
        country = product.get("country") or ""
        stated_elevation = _description_elevation(product["description"], region)
        message = f"Region: {product['title']} -> {region}"
        if stated_elevation is not None:
            message += (
                f"; description elevation -> {stated_elevation.elevation_meters} m "
                f"({stated_elevation.elevation_confidence:.2f})"
            )
        print(message, file=sys.stderr, flush=True)
        return product, region, country, stated_elevation

    return await asyncio.gather(*(extract(product) for product in products))


async def _estimate_locations(
    estimator: TeaElevationEstimator,
    product_locations: list[tuple[dict, str, str, ElevationEstimate | None]],
    semaphore: asyncio.Semaphore,
) -> dict[str, ElevationEstimate]:
    unique_locations: dict[str, tuple[str, str]] = {}
    for _, region, country, stated_elevation in product_locations:
        if stated_elevation is not None:
            continue
        unique_locations.setdefault(_location_key(region, country), (region, country))

    async def estimate(key: str, region: str, country: str) -> tuple[str, ElevationEstimate]:
        async with semaphore:
            result = await estimator.estimate_location(region, country)
        print(
            f"Elevation: {_location_label(region, country)} -> "
            f"{result.elevation_meters} m ({result.elevation_confidence:.2f})",
            file=sys.stderr,
            flush=True,
        )
        return key, result

    estimated = await asyncio.gather(
        *(estimate(key, region, country) for key, (region, country) in unique_locations.items())
    )
    return dict(estimated)


async def _run(args: argparse.Namespace) -> tuple[dict, list[dict], int, int]:
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is missing from .env or the environment")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is missing from .env or the environment")

    input_payload, products = _load_products(args.input)
    selected = _select_products(products, args.titles, args.estimate_all)
    lm = dspy.LM(args.model, cache=not args.no_cache)
    dspy.configure(lm=lm)

    search_cache = TavilySearchCache(tavily_api_key, args.search_cache)
    estimator = TeaElevationEstimator(search_cache)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    product_locations = await _extract_regions(estimator, selected, semaphore)
    estimates = await _estimate_locations(estimator, product_locations, semaphore)

    results = [
        product | (stated_elevation or estimates[_location_key(region, country)]).model_dump()
        for product, region, country, stated_elevation in product_locations
    ]
    unique_location_count = len(
        {_location_key(region, country) for _, region, country, _ in product_locations}
    )
    return input_payload, results, unique_location_count, len(estimates)


def _signature_summary() -> str:
    signatures = (
        ExtractGrowingRegion,
        RequestElevationEvidence,
        EstimateElevationFromEvidence,
    )
    sections = []
    for signature in signatures:
        fields = []
        for name, field in signature.model_fields.items():
            direction = "input" if name in signature.input_fields else "output"
            fields.append(
                f"- {direction} `{name}`: {field.annotation}\n  {field.description or ''}"
            )
        sections.append(
            f"{signature.__name__}\n"
            f"{'-' * len(signature.__name__)}\n"
            f"{signature.__doc__.strip()}\n\n" + "\n".join(fields)
        )
    sections.append(
        "Final appended fields\n"
        "---------------------\n"
        f"{json.dumps(ElevationEstimate.model_json_schema(), indent=2)}"
    )
    return "\n\n".join(sections)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Extract growing regions and estimate elevations with DSPy and Tavily.")
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--search-cache", type=Path, default=DEFAULT_SEARCH_CACHE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--titles",
        nargs="+",
        help="Estimate exact product titles instead of the complete dataset.",
    )
    parser.add_argument(
        "--all",
        dest="estimate_all",
        action="store_true",
        help="Estimate every product in the enriched input dataset.",
    )
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--no-cache", action="store_true", help="Disable DSPy LM cache.")
    parser.add_argument(
        "--show-signatures",
        action="store_true",
        help="Print exact instructions and schemas without using an LM or Tavily.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.show_signatures:
        print(_signature_summary())
        return
    if args.estimate_all and args.titles:
        raise ValueError("Use either --all or --titles, not both")
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be at least 1")

    input_payload, results, unique_location_count, searched_location_count = asyncio.run(_run(args))
    payload = {
        "dataset": "Cha Yi tea metadata final",
        "model": args.model,
        "source_dataset": input_payload.get("dataset"),
        "selection": ({"all": len(results)} if args.estimate_all else {"titles": args.titles}),
        "unique_locations": unique_location_count,
        "tavily_searched_locations": searched_location_count,
        "products": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(results)} teas across {unique_location_count} unique locations "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
