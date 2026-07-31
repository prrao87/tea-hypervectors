#!/usr/bin/env python3
"""Demonstrate the three reversibility claims used in the blog post.

    uv run src/verify_invertibility.py

Three operations are easy to conflate:

  1. binding by a bipolar key                -> exact
  2. removing a known addend from the bundle  -> exact to float32 rounding
  3. recovering a bipolarized composite       -> exact

Recovering an unknown bundled member is a different, approximate cleanup task.
The final section checks that compact float16 storage preserves the useful
direction and nearest neighbours.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hdc.core import bind, bundle, is_bipolar
from hdc.dataset import DEFAULT_CACHE, HF_TABLE, HF_URI, load_teas, warm_embedder
from hdc.encoder import EncodedTea, TeaEncoder
from hdc.manifest import EncoderManifest
from hdc.report import Report
from hdc.similarity import cosine
from hdc.storage import from_storage, to_storage


def pick_complete_tea(encodings: list[EncodedTea]) -> EncodedTea:
    """The walkthrough needs one tea that actually uses every component."""
    return next(e for e in encodings if len(e.components) == 6)


def exact_binding(report: Report, encoder: TeaEncoder) -> None:
    report.section("1. Binding by a bipolar key is exactly self-inverse")

    key = encoder.hv.role("aroma")
    value = encoder.hv.item("class", "oolong")
    bound = bind(value, key)
    recovered = bind(bound, key)

    report.check("key is strictly bipolar", is_bipolar(key))
    report.check("key * key == 1 everywhere", bool(torch.equal(key * key, torch.ones_like(key))))
    report.check("bind(bind(v, key), key) == v", bool(torch.equal(recovered, value)))
    report.note(f"cosine(bound, v) = {cosine(bound, value):+.4f}  (binding hides the value)")

    # The self-inverse property comes from {-1, +1}, not from MAP in general.
    weighted = 3.0 * value
    report.check(
        "a non-bipolar key does NOT round-trip",
        not torch.equal(bind(bind(value, weighted), weighted), value),
        "binding twice with 3v multiplies every coordinate by 9",
    )


def additive_removal(report: Report, tea: EncodedTea) -> None:
    report.section("2. A known addend can be subtracted back out of an intact bundle")

    formula = " + ".join(f"{tea.weights[n]:g}{n[0].upper()}" for n in tea.present)
    report.note(f"H = {formula}")
    report.note(
        "Keeping H unnormalized in float32 is what makes this possible. Unlike the "
        "binding round-trip above, this is exact to float32 rounding rather than "
        "bit-identical: float addition is not associative, so summing five terms "
        "and subtracting the sixth lands ~1e-7 away from summing five terms directly."
    )

    name = "roast"
    without = tea.without(name)
    rest = bundle(
        [tea.weights[other] * tea.components[other] for other in tea.present if other != name]
    )
    restored = without + tea.weights[name] * tea.components[name]
    residual = float((without - rest).abs().max())
    report.check(
        f"H - {tea.weights[name]:g}*{name} recovers the sum of the others",
        bool(torch.allclose(without, rest, atol=1e-5)),
        f"largest coordinate residual: {residual:.2e}",
    )
    report.check(
        f"re-adding {name} restores H",
        bool(torch.allclose(restored, tea.bundle, atol=1e-5)),
    )
    report.note(
        "Note what this does NOT say: the sum alone never reveals which unknown "
        "components produced it. Removal works only because we already knew the "
        "component and its weight."
    )


def composite_factor(report: Report, encoder: TeaEncoder, tea: EncodedTea) -> None:
    report.section("3. A bipolarized composite is recoverable from an S-P-O product")

    tea_factor = encoder.tea_factor(tea)
    subject = encoder.hv.item("session", "morning-tasting")
    predicate = encoder.hv.symbol("predicate", "features")

    triple = bind(subject, predicate, tea_factor)
    recovered = bind(triple, subject, predicate)

    report.check("H itself is not bipolar", not is_bipolar(tea.bundle))
    report.check("bipolarize(H) is strictly bipolar", is_bipolar(tea_factor))
    report.check(
        "unbinding the triple recovers bipolarize(H) exactly",
        bool(torch.equal(recovered, tea_factor)),
    )

    # The negative result matters as much as the positive one.
    report.check(
        "the recovered factor is NOT H",
        not torch.equal(recovered, tea.bundle),
        f"cosine(bipolarize(H), H) = {cosine(tea_factor, tea.bundle):+.4f}",
    )
    report.note(
        f"H has {len(tea.bundle.unique())} distinct coordinate values; the bipolar factor "
        f"has {len(tea_factor.unique())}. Bipolarization discards the weights and cannot "
        "be undone."
    )
    report.check(
        "zero-coordinate tie-breaking is deterministic",
        bool(torch.equal(encoder.tea_factor(tea), tea_factor)),
    )


def storage_stability(
    report: Report, encodings: list[EncodedTea], manifest: EncoderManifest
) -> None:
    report.section("Storage: float16 is compact and approximate, and rankings survive it")

    originals = torch.stack([encoded.bundle for encoded in encodings])
    reloaded = torch.stack([from_storage(to_storage(e.bundle), manifest) for e in encodings])

    report.note(
        f"payload per vector: {manifest.dimensions * 2:,} bytes as float16 "
        f"vs {manifest.dimensions * 4:,} bytes as float32"
    )
    report.check(
        "float16 storage is NOT bit-exact, and we do not claim it is",
        not torch.equal(reloaded, originals),
        f"largest coordinate change: {float((reloaded - originals).abs().max()):.2e}",
    )

    # What actually has to survive persistence is the ranking, not the bits.
    worst_self = min(cosine(o, r) for o, r in zip(originals, reloaded, strict=True))
    report.check(
        "every reloaded vector is still essentially the same direction",
        worst_self > 0.9999,
        f"worst cosine(original, reloaded) = {worst_self:.6f}",
    )

    same_nearest = 0
    for index in range(len(encodings)):
        original_scores = torch.nn.functional.cosine_similarity(
            originals, originals[index].unsqueeze(0), dim=1
        )
        reloaded_scores = torch.nn.functional.cosine_similarity(
            reloaded, reloaded[index].unsqueeze(0), dim=1
        )
        original_scores[index] = -2.0
        reloaded_scores[index] = -2.0
        same_nearest += int(original_scores.argmax() == reloaded_scores.argmax())

    total = len(encodings)
    report.check(
        "the nearest neighbour is unchanged for every tea",
        same_nearest == total,
        f"{same_nearest}/{total}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=HF_URI)
    parser.add_argument("--table", default=HF_TABLE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=1_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = EncoderManifest()
    teas = load_teas(args.source, args.table, args.limit)
    encoder = TeaEncoder(manifest, warm_embedder(manifest, teas, args.cache))
    encodings = [encoder.encode(record) for record in teas.to_dicts()]
    tea = pick_complete_tea(encodings)

    report = Report(f"MAP invertibility on {len(encodings)} teas")
    report.note(f"walkthrough tea: id={tea.tea_id} components={tea.present}")

    exact_binding(report, encoder)
    additive_removal(report, tea)
    composite_factor(report, encoder, tea)
    storage_stability(report, encodings, manifest)
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
