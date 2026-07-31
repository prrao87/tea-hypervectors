"""The typed tea encoder: one hypervector per field type, then a weighted sum."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torchhd import MAPTensor

from hdc.core import HypervectorFactory, bind, bundle, is_bipolar
from hdc.manifest import EncoderManifest
from hdc.projection import project_to_bipolar, rademacher_matrix
from hdc.sensory import PhraseEmbedder, canonicalize

COMPONENTS = ("aroma", "taste", "class", "oxidation", "roast", "elevation")


@dataclass(frozen=True)
class EncodedTea:
    """The result of encoding one tea.

    `bundle` is the final vector used for similarity search.

    `components` keeps the individual vector created for each available field,
    such as aroma, class, or elevation. `weights` records how strongly each of
    those fields contributed to `bundle`. Keeping these pieces lets us explain
    a similarity score and ask counterfactual questions such as, "What would
    this tea's vector look like without roast?"

    Only `bundle` is stored as a vector in LanceDB. The components and weights
    are temporary, inspectable details produced while encoding the source row.
    `tea_id` links the result back to that row.
    """

    tea_id: int
    bundle: MAPTensor
    components: dict[str, MAPTensor]
    weights: dict[str, float]

    @property
    def present(self) -> tuple[str, ...]:
        return tuple(name for name in COMPONENTS if name in self.components)

    def without(self, name: str) -> MAPTensor:
        """Return the tea vector with one field's contribution removed."""
        return self.bundle - self.weights[name] * self.components[name]


class TeaEncoder:
    def __init__(self, manifest: EncoderManifest, embedder: PhraseEmbedder) -> None:
        self.manifest = manifest
        self.embedder = embedder
        self.hv = HypervectorFactory(manifest)
        self.projection = rademacher_matrix(manifest, self.hv)

    # -- individual field geometries -------------------------------------

    def sensory_component(self, role: str, phrases: tuple[str, ...]) -> MAPTensor | None:
        """Projected sensory meaning, tagged with its field role.

        Binding to a role is what keeps "honey" as an aroma distinct from
        "honey" as a taste while preserving the same underlying direction.
        """
        embedding = self.embedder.field_embedding(phrases)
        if embedding is None:
            return None
        value = project_to_bipolar(embedding, self.projection, self.hv, context=f"sensory:{role}")
        return bind(self.hv.role(role), value)

    def class_component(self, value: str) -> MAPTensor:
        """Nominal category: no order is implied between black, green, oolong..."""
        if value not in self.manifest.class_vocabulary:
            raise ValueError(f"Unknown class {value!r}")
        return bind(self.hv.role("class"), self.hv.item("class", value))

    def ordinal_component(self, role: str, vocabulary: tuple[str, ...], value: str) -> MAPTensor:
        """Ordered category: adjacent rungs of the scale stay similar."""
        if value not in vocabulary:
            raise ValueError(f"Unknown {role} {value!r}; expected one of {vocabulary}")
        level = self.hv.level(role, len(vocabulary), vocabulary.index(value))
        return bind(self.hv.role(role), level)

    def elevation_component(self, meters: int) -> MAPTensor:
        """Numeric value quantized onto the manifest's metre scale."""
        manifest = self.manifest
        clamped = min(max(meters, manifest.elevation_min_meters), manifest.elevation_max_meters)
        index = round((clamped - manifest.elevation_min_meters) / manifest.elevation_step_meters)
        level = self.hv.level("elevation", manifest.elevation_levels, index)
        return bind(self.hv.role("elevation"), level)

    # -- the whole record -------------------------------------------------

    def encode(self, record: Mapping[str, object]) -> EncodedTea:
        manifest = self.manifest
        weights = manifest.weights

        aroma = canonicalize(record.get("aroma"))
        taste = canonicalize(record.get("taste"))
        components: dict[str, MAPTensor] = {}
        applied: dict[str, float] = {}

        # A missing sensory field contributes nothing and its share is *not*
        # redistributed. Reassigning it would silently amplify the field that
        # happens to survive, turning absent evidence into a similarity signal.
        # The remaining weights then sum to less than 1, which is fine: cosine
        # divides the overall magnitude out.
        for name, role, phrases in (
            ("aroma", "aroma", aroma),
            ("taste", "taste", taste),
        ):
            component = self.sensory_component(role, phrases)
            if component is not None:
                components[name] = component
                applied[name] = getattr(weights, name)

        components["class"] = self.class_component(str(record["class"]))
        applied["class"] = weights.tea_class

        components["oxidation"] = self.ordinal_component(
            "oxidation", manifest.oxidation_vocabulary, str(record["oxidation"])
        )
        applied["oxidation"] = weights.oxidation

        components["roast"] = self.ordinal_component(
            "roast", manifest.roast_vocabulary, str(record["roast"])
        )
        applied["roast"] = weights.roast

        meters = record.get("elevation_meters")
        if meters is not None:
            # Zero metres is a real measurement at the bottom of the scale, so
            # the check above must be `is not None` rather than a truth test.
            confidence = float(record.get("elevation_confidence", 1.0))
            components["elevation"] = self.elevation_component(int(meters))
            applied["elevation"] = manifest.elevation_weight(confidence)

        raw = bundle([applied[name] * components[name] for name in components])
        return EncodedTea(
            tea_id=int(record["id"]),
            bundle=raw,
            components=components,
            weights=applied,
        )

    def tea_factor(self, encoded: EncodedTea) -> MAPTensor:
        """Return a self-invertible version of a tea for use in a relationship.

        A tea's normal vector is a weighted sum whose coordinates can be values
        such as `0.42` or `-0.17`. It works well for similarity search, but it
        cannot be cancelled out of a MAP binding by multiplying with it again:
        those coordinates would be squared rather than become 1.

        This method replaces every coordinate with its sign, producing only
        `-1` and `+1`. Such a bipolar vector is self-inverse because both
        `(-1) * (-1)` and `1 * 1` equal 1. It can therefore be recovered exactly
        from a relationship when the other factors are known:

            triple = subject * predicate * tea_factor
            tea_factor = triple * subject * predicate

        The conversion discards the original weight magnitudes. Use
        `encoded.bundle` for tea similarity; use this bipolar factor only when
        demonstrating binding and unbinding. Recovering it does not recover the
        original weighted bundle.
        """
        return self.hv.bipolarize(encoded.bundle, context=f"tea:{encoded.tea_id}")


def assert_encoder_invariants(encoded: EncodedTea, manifest: EncoderManifest) -> None:
    """Check the properties the rest of the pipeline depends on."""
    for name, component in encoded.components.items():
        assert component.shape == (manifest.dimensions,), f"{name} has the wrong shape"
        assert is_bipolar(component), f"{name} component is not strictly bipolar"
    raw = encoded.bundle
    assert raw.shape == (manifest.dimensions,), "bundle has the wrong shape"
    assert raw.dtype is torch.float32, "bundle is not float32"
    assert torch.isfinite(raw).all(), "bundle contains non-finite coordinates"
