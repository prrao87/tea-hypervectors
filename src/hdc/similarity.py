"""Cosine similarity over raw bundles, plus a per-component breakdown."""

from __future__ import annotations

import torch

from hdc.encoder import EncodedTea


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    """Cosine over the unnormalized bundle.

    The stored vector is deliberately not L2-normalized: cosine already divides
    out magnitude, and keeping the raw sum is what makes exact component
    removal possible.
    """
    return float(torch.nn.functional.cosine_similarity(left, right, dim=0))


def component_contributions(query: torch.Tensor, neighbour: EncodedTea) -> dict[str, float]:
    """Split one cosine score into the neighbour's individual components.

    Because the neighbour's bundle is `sum(weight_i * C_i)`, the dot product
    distributes over the sum. Dividing each term by the same pair of norms gives
    contributions that add back up to the cosine score, so the breakdown is a
    decomposition rather than an attribution heuristic.
    """
    denominator = float(query.norm()) * float(neighbour.bundle.norm())
    return {
        name: neighbour.weights[name] * float(query @ component) / denominator
        for name, component in neighbour.components.items()
    }
