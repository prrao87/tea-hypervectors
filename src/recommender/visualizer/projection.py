"""Deterministic three-dimensional PCA for the recommender visual."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from recommender.catalog import TeaCatalog
from recommender.visualizer.snapshots import Snapshot

FloatCoordinates = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Projection:
    """Fixed catalogue and prototype coordinates from one PCA projection."""

    catalogue: FloatCoordinates
    positive_centroids: dict[int, FloatCoordinates]
    negative_centroids: dict[int, FloatCoordinates | None]
    axis_labels: tuple[str, str, str] = ("PCA 1", "PCA 2", "PCA 3")
    explained_variance_ratio: tuple[float, float, float] | None = None


def as_numpy(vector: torch.Tensor) -> NDArray[np.float32]:
    """Move one hypervector to a read-only-friendly NumPy representation."""
    return vector.detach().cpu().numpy().astype(np.float32, copy=False)


def fit_pca_3d(catalog: TeaCatalog, snapshots: tuple[Snapshot, ...]) -> Projection:
    """Fit PCA once on the catalogue and transform every prototype into it."""
    if len(catalog.ids) <= 3:
        raise ValueError("PCA needs more than 3 catalogue teas")

    catalogue_hypervectors = as_numpy(catalog.vectors).astype(np.float64)
    mean = catalogue_hypervectors.mean(axis=0)
    centered = catalogue_hypervectors - mean

    # With 166 rows and 10,000 columns, the catalogue Gram matrix is much
    # smaller than the covariance matrix while producing the same exact PCA.
    gram = centered @ centered.T
    eigenvalues, left_vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    leading_values = np.maximum(eigenvalues[order[:3]], 0.0)
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * max(float(leading_values[0]), 1.0)
    if np.any(leading_values <= tolerance):
        raise ValueError("catalogue does not have 3 non-zero principal components")

    components = centered.T @ left_vectors[:, order[:3]]
    components /= np.sqrt(leading_values)

    # Eigenvector signs are arbitrary. Pin each one to its largest loading so
    # repeated exports retain the same orientation.
    for component_index in range(3):
        anchor = int(np.argmax(np.abs(components[:, component_index])))
        if components[anchor, component_index] < 0:
            components[:, component_index] *= -1

    catalogue_coordinates = (centered @ components).astype(np.float32)
    positives: dict[int, FloatCoordinates] = {}
    negatives: dict[int, FloatCoordinates | None] = {}
    for snapshot in snapshots:
        positives[snapshot.sequence] = (
            (as_numpy(snapshot.positive_centroid).astype(np.float64) - mean) @ components
        ).astype(np.float32)
        negatives[snapshot.sequence] = None
        if snapshot.negative_prototype is not None:
            negatives[snapshot.sequence] = (
                (as_numpy(snapshot.negative_prototype).astype(np.float64) - mean) @ components
            ).astype(np.float32)

    total_variance = float(np.maximum(eigenvalues, 0.0).sum())
    explained_variance = tuple(float(value / total_variance) for value in leading_values)
    return Projection(
        catalogue=catalogue_coordinates,
        positive_centroids=positives,
        negative_centroids=negatives,
        explained_variance_ratio=explained_variance,
    )
