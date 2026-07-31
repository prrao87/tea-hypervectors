"""Deterministic MAP primitives: symbols, ordered levels, and bipolarization.

MAP (Multiply-Add-Permute) uses element-wise multiplication for binding and
element-wise addition for bundling. Multiplication is self-inverse only when
every coordinate is -1 or +1, so this module is careful about exactly which
vectors are strictly bipolar and which are weighted sums.
"""

from __future__ import annotations

import hashlib

import torch
import torchhd
from torchhd import MAPTensor

from hdc.manifest import EncoderManifest


class HypervectorFactory:
    """Generates every atomic hypervector the encoder needs.

    Vectors are derived from a hash of `global_seed | namespace | token` rather
    than drawn from one shared random stream. That means there is no vocabulary
    file to keep in sync.
    """

    def __init__(self, manifest: EncoderManifest) -> None:
        self.manifest = manifest
        self._cache: dict[tuple[str, str], MAPTensor] = {}
        self._level_cache: dict[str, MAPTensor] = {}

    def seed_for(self, namespace: str, token: str) -> int:
        digest = hashlib.sha256(
            f"{self.manifest.global_seed}|{namespace}|{token}".encode()
        ).digest()
        return int.from_bytes(digest[:8], "big") % (2**63 - 1)

    def _generator(self, namespace: str, token: str) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed_for(namespace, token))
        return generator

    def symbol(self, namespace: str, token: str) -> MAPTensor:
        """One strictly bipolar hypervector per (namespace, token) pair."""
        key = (namespace, token)
        if key not in self._cache:
            self._cache[key] = torchhd.random(
                1,
                self.manifest.dimensions,
                self.manifest.vsa,
                generator=self._generator(namespace, token),
                dtype=torch.float32,
            )[0]
        return self._cache[key]

    def role(self, name: str) -> MAPTensor:
        """A field marker, e.g. AROMA. Bound to a value it says 'this is an aroma'."""
        return self.symbol("role", name)

    def item(self, namespace: str, value: str) -> MAPTensor:
        """An unordered categorical value. Distinct items are near-orthogonal."""
        return self.symbol(f"item:{namespace}", value)

    def level(self, namespace: str, count: int, index: int) -> MAPTensor:
        """One rung of an ordered scale.

        TorchHD builds these so neighbouring levels share most coordinates and
        the endpoints are near-orthogonal, which is exactly the geometry an
        ordinal or numeric field wants: `low` should sit closer to `medium`
        than to `high`.
        """
        if namespace not in self._level_cache:
            self._level_cache[namespace] = torchhd.level(
                count,
                self.manifest.dimensions,
                self.manifest.vsa,
                generator=self._generator("level", namespace),
                dtype=torch.float32,
            )
        levels = self._level_cache[namespace]
        if levels.shape[0] != count:
            raise ValueError(
                f"Level namespace {namespace!r} was built with {levels.shape[0]} "
                f"levels but {count} were requested"
            )
        return levels[index]

    def tie_breaker(self, context: str) -> MAPTensor:
        """A deterministic bipolar vector used to resolve exact zeros."""
        return self.symbol("tie", context)

    def bipolarize(self, hypervector: torch.Tensor, context: str) -> MAPTensor:
        """Project any vector into strictly bipolar MAP space.

        Zeros get the corresponding coordinate of a context-specific tie vector
        instead of being pushed to one side, which is what `torchhd.normalize`
        would do. A systematic zero -> -1 rule would make every vector with
        cancelled coordinates lean the same way.
        """
        tie = self.tie_breaker(context).to(hypervector.dtype)
        positive = torch.ones((), dtype=hypervector.dtype)
        return torchhd.ensure_vsa_tensor(
            torch.where(
                hypervector > 0,
                positive,
                torch.where(hypervector < 0, -positive, tie),
            ),
            vsa=self.manifest.vsa,
            dtype=torch.float32,
        )


def is_bipolar(hypervector: torch.Tensor) -> bool:
    """True when every coordinate is exactly -1 or +1."""
    return bool(torch.isin(hypervector, torch.tensor([-1.0, 1.0])).all())


def bind(*factors: torch.Tensor) -> MAPTensor:
    """MAP binding: element-wise multiplication. Self-inverse for bipolar factors."""
    return torchhd.multibind(torch.stack(list(factors)))


def bundle(hypervectors: list[torch.Tensor]) -> MAPTensor:
    """MAP bundling: element-wise addition, kept as a plain sum.

    The sum is deliberately *not* normalized. Normalizing would collapse the
    weights to {-1, +1}, discarding how much each field was meant to matter and
    the ability to subtract a known component back out.
    """
    if not hypervectors:
        raise ValueError("Cannot bundle an empty sequence")
    return torchhd.multiset(torch.stack(hypervectors))
