#!/usr/bin/env python3
"""Check that the encoder actuallty has the geometry we designed, one field at a time.

    uv run src/verify_geometry.py

This is the evidence that the weights encode domain knowledge/judgment.
- It does not claim the encoder beats a text embedding-only baseline on absolute terms.
- Rather, it confirms that the encoder behaves the way the manifest says it should.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from hdc.core import is_bipolar
from hdc.dataset import DEFAULT_CACHE
from hdc.encoder import TeaEncoder
from hdc.manifest import EncoderManifest
from hdc.report import Report
from hdc.sensory import PhraseEmbedder, canonicalize
from hdc.similarity import cosine

# One synthetic tea, varied one field at a time. Synthetic rather than real so
# that every comparison isolates exactly one change.
BASE = {
    "id": 0,
    "title": "Synthetic base tea",
    "aroma": ["honey", "orchid"],
    "taste": ["creamy texture", "stone fruit"],
    "class": "oolong",
    "oxidation": "medium",
    "roast": "none",
    "elevation_meters": 1200,
    "elevation_confidence": 1.0,
}

VARIANTS: list[tuple[str, dict]] = [
    ("roast none -> light", {"roast": "light"}),
    ("roast none -> heavy", {"roast": "heavy"}),
    ("oxidation medium -> high", {"oxidation": "high"}),
    ("class oolong -> green", {"class": "green"}),
    ("elevation +50 m", {"elevation_meters": 1250}),
    ("elevation +500 m", {"elevation_meters": 1700}),
    ("elevation +1500 m", {"elevation_meters": 2700}),
    ("elevation confidence 1.0 -> 0.5", {"elevation_confidence": 0.5}),
    ("elevation confidence 1.0 -> 0.0", {"elevation_confidence": 0.0}),
    ("taste removed", {"taste": []}),
    ("aroma honey -> nectar (synonym)", {"aroma": ["nectar", "orchid"]}),
    ("aroma honey -> diesel fuel (unrelated)", {"aroma": ["diesel fuel", "orchid"]}),
]


def variant(**changes) -> dict:
    record = copy.deepcopy(BASE)
    record.update(changes)
    return record


def invariants(report: Report, encoder: TeaEncoder) -> None:
    report.section("Determinism and invariants")

    first = encoder.encode(BASE)
    second = encoder.encode(BASE)
    report.check(
        "the same record encodes identically",
        bool(torch.equal(first.bundle, second.bundle)),
    )
    report.check(
        "every component is strictly bipolar",
        all(is_bipolar(component) for component in first.components.values()),
    )

    shuffled = encoder.encode(variant(aroma=["orchid", "honey"]))
    report.check(
        "reordering aroma phrases changes nothing",
        bool(torch.equal(shuffled.bundle, first.bundle)),
        "canonicalization sorts the phrase list",
    )


def missing_signals(report: Report, encoder: TeaEncoder) -> None:
    report.section("Missing signals contribute nothing")

    full = encoder.encode(BASE)
    aroma_only = encoder.encode(variant(taste=[]))
    neither = encoder.encode(variant(aroma=[], taste=[]))

    report.check(
        "both sensory fields missing omits the group",
        "aroma" not in neither.components and "taste" not in neither.components,
    )

    # A missing field must be an absence, not a shared "missing" token that all
    # incomplete teas would have in common.
    report.check(
        "a missing taste is the full bundle minus the taste term",
        bool(torch.allclose(aroma_only.bundle, full.without("taste"), atol=1e-5)),
        "no placeholder vector is substituted",
    )

    sea_level = encoder.encode(variant(elevation_meters=0))
    report.check(
        "elevation 0 m is a real measurement, not a missing value",
        "elevation" in sea_level.components and sea_level.weights["elevation"] > 0,
    )
    no_confidence = encoder.encode(variant(elevation_confidence=0.0))
    report.check(
        "confidence 0 makes elevation contribute nothing",
        no_confidence.weights["elevation"] == 0.0
        and bool(torch.allclose(no_confidence.bundle, full.without("elevation"), atol=1e-5)),
    )


def type_geometry(report: Report, encoder: TeaEncoder, manifest: EncoderManifest) -> None:
    report.section("Each data type has the geometry its type deserves")

    green = encoder.hv.item("class", "green")
    black = encoder.hv.item("class", "black")
    report.check("identical categories are maximally similar", cosine(green, green) > 0.9999)
    report.check(
        "different categories are near-orthogonal",
        abs(cosine(green, black)) < 0.05,
        f"cosine(green, black) = {cosine(green, black):+.4f}  (nominal: no order implied)",
    )

    low, medium, high = (encoder.hv.level("oxidation", 3, i) for i in range(3))
    report.check(
        "ordered levels decay with distance",
        cosine(low, medium) > cosine(low, high),
        f"low~medium = {cosine(low, medium):+.3f}, low~high = {cosine(low, high):+.3f}",
    )

    near, mid, far = (encoder.elevation_component(m) for m in (1200, 1250, 1700))
    report.check(
        "nearby elevations are more similar than distant ones",
        cosine(near, mid) > cosine(near, far),
        f"1200~1250 = {cosine(near, mid):+.3f}, 1200~1700 = {cosine(near, far):+.3f}",
    )

    confident = encoder.encode(BASE)
    unsure = encoder.encode(variant(elevation_confidence=0.5))
    report.check(
        "confidence scales elevation's loudness, not its direction",
        torch.equal(confident.components["elevation"], unsure.components["elevation"])
        and unsure.weights["elevation"] == manifest.elevation_weight(0.5),
        f"weight {confident.weights['elevation']:g} -> {unsure.weights['elevation']:g}",
    )

    as_aroma = encoder.sensory_component("aroma", ("honey",))
    as_taste = encoder.sensory_component("taste", ("honey",))
    report.check(
        "the same phrase means something different in a different field",
        abs(cosine(as_aroma, as_taste)) < 0.05,
        f"cosine(aroma:honey, taste:honey) = {cosine(as_aroma, as_taste):+.4f}",
    )


def counterfactuals(report: Report, encoder: TeaEncoder) -> None:
    report.section("Counterfactuals: change one field, watch similarity move")

    base = encoder.encode(BASE)
    scores = {
        label: cosine(base.bundle, encoder.encode(variant(**changes)).bundle)
        for label, changes in VARIANTS
    }
    width = max(len(label) for label in scores)
    for label, score in sorted(scores.items(), key=lambda item: -item[1]):
        report.note(f"{label:<{width}}  cosine to base = {score:.4f}")

    report.check(
        "a roast change is a smaller nudge than a class or oxidation change",
        scores["roast none -> light"] > scores["oxidation medium -> high"]
        and scores["roast none -> light"] > scores["class oolong -> green"],
    )
    report.check(
        "moving elevation further reduces similarity monotonically",
        scores["elevation +50 m"] > scores["elevation +500 m"] > scores["elevation +1500 m"],
    )
    report.check(
        "lowering elevation confidence reduces its influence monotonically",
        scores["elevation confidence 1.0 -> 0.5"] > scores["elevation confidence 1.0 -> 0.0"],
    )
    report.check(
        "a synonym moves less than an unrelated descriptor",
        scores["aroma honey -> nectar (synonym)"]
        > scores["aroma honey -> diesel fuel (unrelated)"],
    )
    report.check(
        "dropping a whole sensory field matters more than a roast change",
        scores["taste removed"] < scores["roast none -> heavy"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    return parser.parse_args()


def synthetic_embedder(manifest: EncoderManifest, cache: Path) -> PhraseEmbedder:
    """Warm only the phrases the synthetic records use."""
    embedder = PhraseEmbedder(manifest, cache)
    phrases: set[str] = set(canonicalize(BASE["aroma"]) + canonicalize(BASE["taste"]))
    for _, changes in VARIANTS:
        for field in ("aroma", "taste"):
            phrases.update(canonicalize(changes.get(field)))
    embedder.warm(sorted(phrases))
    embedder.save()
    return embedder


def main() -> int:
    args = parse_args()
    manifest = EncoderManifest()
    encoder = TeaEncoder(manifest, synthetic_embedder(manifest, args.cache))

    report = Report("Encoder geometry")
    invariants(report, encoder)
    missing_signals(report, encoder)
    type_geometry(report, encoder, manifest)
    counterfactuals(report, encoder)
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
