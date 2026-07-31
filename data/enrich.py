#!/usr/bin/env python3
"""Extract a compact tea ontology from Cha Yi product descriptions with DSPy."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Literal

import dspy
from dotenv import load_dotenv
from pydantic import BaseModel

Oxidation = Literal["low", "medium", "high"]
Roast = Literal["none", "light", "heavy"]
RoastBasis = Literal["description", "tea_knowledge"]

EXTRACTED_DIR = Path(__file__).with_name("extracted")
DEFAULT_INPUT = EXTRACTED_DIR / "tea_data.json"
DEFAULT_OUTPUT = EXTRACTED_DIR / "tea_data_enriched.json"
DEFAULT_MODEL = "openrouter/google/gemini-3.5-flash"
SAMPLE_CATEGORIES = ("green", "black")


class TeaOntology(BaseModel):
    """The structured result assembled from the focused extractors."""

    oxidation: Oxidation
    roast: Roast
    roast_basis: RoastBasis
    aroma: list[str]
    taste: list[str]


class InferOxidation(dspy.Signature):
    """Infer oxidation from the description's processing and tea-family evidence.

    Use only the description. When it omits an oxidation percentage, infer the
    closest level from any tea family or processing method it names.
    """

    description: str = dspy.InputField(desc="The complete English product description.")
    oxidation: Oxidation = dspy.OutputField(desc="Oxidation inferred from the description.")


class ExtractDescriptionRoast(dspy.Signature):
    """Extract processing roast when the description explicitly states it.

    Treat an unqualified roast as light and charcoal, deep, dark, or repeated
    roasting as heavy. Count only intentional roasting of the tea leaves.
    Oxidation, drying, firing, smoking, sensory comparisons, and roasted
    ingredients in a blend are not tea-roast evidence. If processing roast is
    not stated, return null rather than assuming none.
    """

    description: str = dspy.InputField(desc="The complete English product description.")
    roast: Roast | None = dspy.OutputField(
        desc="Explicit processing roast, or null when the description does not state it."
    )


class InferRoastFromTeaKnowledge(dspy.Signature):
    """Infer likely processing roast for a tea whose description does not state it.

    Use the product title, category, named style, and tea knowledge
    conservatively. Return none unless the named tea style is specifically
    defined by intentional roasting of its leaves. Do not equate black-tea
    oxidation, drying, firing, or smoking with roasting; Wuyi or Fujian origin
    and hong cha, gongfu, xiao zhong, or lapsang names alone do not imply roast.
    """

    title: str = dspy.InputField(desc="The product's English title.")
    category: str = dspy.InputField(desc="The source collection's tea category.")
    roast: Roast = dspy.OutputField(desc="Likely processing roast inferred from tea knowledge.")


class ExtractAroma(dspy.Signature):
    """Extract verbatim aroma, flavour, and named tasting-note spans.

    Each item must be the smallest useful contiguous substring of the
    description, preserving its exact wording, spelling, and punctuation;
    never paraphrase, normalize, or translate it. Keep flavour identities such
    as floral, fruit, vegetal, mineral, wood, smoke, spice, and nut notes here.
    Exclude tactile structure: body, texture, astringency, tannin, smoothness,
    roundness, thickness, weight, and finish.
    """

    description: str = dspy.InputField(desc="The complete English product description.")
    aroma: list[str] = dspy.OutputField(
        desc="Exact contiguous description spans for aroma, flavour, and tasting notes."
    )


class ExtractTaste(dspy.Signature):
    """Extract verbatim palate, mouthfeel, and finish spans.

    Each item must be the smallest useful contiguous substring of the
    description, preserving its exact wording, spelling, and punctuation;
    never paraphrase, normalize, or translate it. Include only tactile or
    structural evidence such as body, texture, astringency, tannin, smoothness,
    roundness, thickness, weight, or an explicitly named finish or aftertaste.
    Exclude pure aroma or flavour-note lists. Generic adjectives such as
    delicate, robust, generous, balanced, or tasty qualify only when the text
    ties them directly to one of those palate properties.
    """

    description: str = dspy.InputField(desc="The complete English product description.")
    taste: list[str] = dspy.OutputField(
        desc="Exact contiguous description spans for palate, mouthfeel, texture, and finish."
    )


class TeaOntologyExtractor(dspy.Module):
    """Run four focused DSPy predictors and combine them into one result."""

    def __init__(self) -> None:
        super().__init__()
        self.infer_oxidation = dspy.Predict(InferOxidation)
        self.extract_description_roast = dspy.Predict(ExtractDescriptionRoast)
        self.infer_roast_from_tea_knowledge = dspy.Predict(InferRoastFromTeaKnowledge)
        self.extract_aroma = dspy.Predict(ExtractAroma)
        self.extract_taste = dspy.Predict(ExtractTaste)

    async def aforward(self, title: str, category: str, description: str) -> dspy.Prediction:
        oxidation, roast, aroma, taste = await asyncio.gather(
            self.infer_oxidation.acall(description=description),
            self.extract_description_roast.acall(description=description),
            self.extract_aroma.acall(description=description),
            self.extract_taste.acall(description=description),
        )

        roast_basis: RoastBasis = "description"
        roast_value = roast.roast
        if roast_value is None:
            roast_basis = "tea_knowledge"
            inferred_roast = await self.infer_roast_from_tea_knowledge.acall(
                title=title,
                category=category,
            )
            roast_value = inferred_roast.roast

        ontology = TeaOntology(
            oxidation=oxidation.oxidation,
            roast=roast_value,
            roast_basis=roast_basis,
            aroma=_validated_verbatim_spans(description, aroma.aroma, "aroma"),
            taste=_validated_verbatim_spans(description, taste.taste, "taste"),
        )
        return dspy.Prediction(ontology=ontology)


def _validated_verbatim_spans(description: str, spans: list[str], field_name: str) -> list[str]:
    """Reject invented/paraphrased spans and remove exact duplicates."""
    validated: list[str] = []
    for span in spans:
        if not span or span not in description:
            raise ValueError(f"{field_name} value is not an exact description span: {span!r}")
        if span not in validated:
            validated.append(span)
    return validated


def _load_products(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    products = payload.get("products")
    if not isinstance(products, list):
        raise TypeError(f"{path} must contain a top-level 'products' list")
    return products


def _select_products(
    products: list[dict],
    limit_per_category: int,
    titles: list[str] | None,
    enrich_all: bool,
) -> list[dict]:
    if enrich_all:
        return products

    if titles:
        products_by_title = {product["title"].casefold(): product for product in products}
        missing = [title for title in titles if title.casefold() not in products_by_title]
        if missing:
            raise ValueError(f"Product titles not found: {', '.join(missing)}")
        return [products_by_title[title.casefold()] for title in titles]

    selected: list[dict] = []
    for category in SAMPLE_CATEGORIES:
        matches = [product for product in products if product.get("category") == category]
        if len(matches) < limit_per_category:
            raise ValueError(f"Need {limit_per_category} {category} teas, found {len(matches)}")
        selected.extend(matches[:limit_per_category])
    return selected


async def _enrich_product(
    extractor: TeaOntologyExtractor,
    product: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        prediction = await extractor.acall(
            title=product["title"],
            category=product["category"],
            description=product["description"],
        )
    print(f"Enriched: {product['title']}", file=sys.stderr, flush=True)
    return product | prediction.ontology.model_dump()


async def _run(args: argparse.Namespace) -> list[dict]:
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is missing from .env or the environment")

    products = _load_products(args.input)
    selected = _select_products(
        products,
        args.limit_per_category,
        args.titles,
        args.enrich_all,
    )

    lm = dspy.LM(args.model, cache=not args.no_cache)
    dspy.configure(lm=lm)
    extractor = TeaOntologyExtractor()
    semaphore = asyncio.Semaphore(args.max_concurrency)
    return await asyncio.gather(
        *(_enrich_product(extractor, product, semaphore) for product in selected)
    )


def _signature_summary() -> str:
    signatures = (
        InferOxidation,
        ExtractDescriptionRoast,
        InferRoastFromTeaKnowledge,
        ExtractAroma,
        ExtractTaste,
    )
    sections = []
    for signature in signatures:
        sections.append(
            f"{signature.__name__}\n"
            f"{'-' * len(signature.__name__)}\n"
            f"{signature.__doc__.strip()}\n"
            f"Output annotation: "
            f"{signature.model_fields[next(iter(signature.output_fields))].annotation}"
        )
    sections.append(
        f"Compound output\n---------------\n{json.dumps(TeaOntology.model_json_schema(), indent=2)}"
    )
    return "\n\n".join(sections)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract tea ontology fields with focused async DSPy signatures."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit-per-category", type=int, default=3)
    parser.add_argument(
        "--titles",
        nargs="+",
        help="Enrich exact product titles instead of the default green/black sample.",
    )
    parser.add_argument(
        "--all",
        dest="enrich_all",
        action="store_true",
        help="Enrich every product in the input dataset.",
    )
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--show-signatures",
        action="store_true",
        help="Print exact instructions and schema without invoking a model.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.show_signatures:
        print(_signature_summary())
        return
    if args.enrich_all and args.titles:
        raise ValueError("Use either --all or --titles, not both")
    if args.limit_per_category < 1:
        raise ValueError("--limit-per-category must be at least 1")
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be at least 1")

    results = asyncio.run(_run(args))
    payload = {
        "dataset": "Cha Yi tea metadata enriched",
        "model": args.model,
        "selection": (
            {"all": len(results)}
            if args.enrich_all
            else (
                {"titles": args.titles}
                if args.titles
                else {
                    "green": args.limit_per_category,
                    "black": args.limit_per_category,
                }
            )
        ),
        "products": results,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(results)} enriched teas to {args.output}")


if __name__ == "__main__":
    main()
