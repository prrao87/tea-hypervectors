"""Serialize the recommender story for a renderer-independent web client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from recommender.catalog import TeaCatalog
from recommender.visualizer.metadata import TeaMetadata
from recommender.visualizer.projection import FloatCoordinates, Projection
from recommender.visualizer.snapshots import Snapshot

CLASS_ORDER = ("green", "white", "yellow", "black", "oolong")
CLASS_COLORS = {
    "green": "#54D68B",
    "white": "#F2EDE1",
    "yellow": "#F4CB55",
    "black": "#F05D6C",
    "oolong": "#F2934C",
}
SIGNAL_COLORS = {
    "positive": "#74E5D0",
    "negative": "#FF6B81",
    "liked": "#E5FAF5",
    "neutral": "#83949C",
    "recommendation": "#EAF3F2",
}


def _coordinates(values: FloatCoordinates | None) -> list[float] | None:
    if values is None:
        return None
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("artifact coordinates must contain three finite values")
    return [round(float(value), 8) for value in values]


def build_artifact(
    catalog: TeaCatalog,
    metadata: tuple[TeaMetadata, ...],
    snapshots: tuple[Snapshot, ...],
    projection: Projection,
    *,
    negative_penalty: float = 0.25,
) -> dict[str, Any]:
    """Build the portable JSON document consumed by Three.js."""
    if len(metadata) != len(catalog.ids):
        raise ValueError("metadata and catalogue lengths do not match")
    if projection.catalogue.shape != (len(catalog.ids), 3):
        raise ValueError("catalogue projection must have three coordinates per tea")

    unknown_classes = sorted(set(catalog.classes) - set(CLASS_COLORS))
    if unknown_classes:
        raise ValueError(f"no visualization color exists for {', '.join(unknown_classes)}")

    teas = []
    for index, record in enumerate(metadata):
        if record.tea_id != catalog.ids[index]:
            raise ValueError("metadata order does not match catalogue order")
        tea = record.as_artifact_record()
        tea["position"] = _coordinates(projection.catalogue[index])
        teas.append(tea)

    checkpoints = []
    for snapshot in snapshots:
        checkpoints.append(
            {
                "sequence": snapshot.sequence,
                "event_count": len(snapshot.events),
                "positive_vote_weight": snapshot.positive_weight,
                "negative_vote_weight": snapshot.negative_weight,
                "positive_position": _coordinates(projection.positive_centroids[snapshot.sequence]),
                "negative_position": _coordinates(projection.negative_centroids[snapshot.sequence]),
                "tastings": [
                    {
                        "tea_id": str(event.tea_id),
                        "verdict": event.verdict.value,
                        "strength": event.strength,
                    }
                    for event in snapshot.events
                ],
                "recommendations": [
                    {
                        "rank": rank,
                        "tea_id": str(recommendation.tea_id),
                        "title": recommendation.title,
                        "class": recommendation.tea_class,
                        "score": round(recommendation.score, 8),
                        "positive_similarity": round(recommendation.positive_similarity, 8),
                        "negative_similarity": (
                            round(recommendation.negative_similarity, 8)
                            if recommendation.negative_similarity is not None
                            else None
                        ),
                    }
                    for rank, recommendation in enumerate(snapshot.recommendations, 1)
                ],
            }
        )

    return {
        "schema_version": 1,
        "source": {
            "catalogue_size": len(catalog.ids),
            "hypervector_dimensions": catalog.dimensions,
            "projection": "PCA",
            "projection_dimensions": 3,
            "explained_variance_ratio": [
                round(value, 8) for value in (projection.explained_variance_ratio or ())
            ],
            "ranking_space": "original normalized hypervectors",
            "negative_penalty": negative_penalty,
        },
        "palette": {
            "classes": [
                {"class": tea_class, "label": tea_class.upper(), "color": CLASS_COLORS[tea_class]}
                for tea_class in CLASS_ORDER
            ],
            "signals": SIGNAL_COLORS,
        },
        "teas": teas,
        "checkpoints": checkpoints,
    }


def write_artifact(artifact: dict[str, Any], destination: Path) -> Path:
    """Write stable, human-readable JSON and return its resolved destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination.resolve()
