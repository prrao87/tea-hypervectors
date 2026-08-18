from __future__ import annotations

import json

import numpy as np
import torch

from recommender.catalog import TeaCatalog
from recommender.events import TastingEvent, Verdict
from recommender.visualizer.artifact import CLASS_COLORS, build_artifact, write_artifact
from recommender.visualizer.metadata import TeaMetadata
from recommender.visualizer.projection import Projection
from recommender.visualizer.snapshots import build_snapshots


def test_exports_portable_threejs_data_without_changing_the_ranking_space(tmp_path) -> None:
    ids = (1, 2, 3, 4, 5)
    catalog = TeaCatalog(
        ids=ids,
        titles=("Liked", "Disliked", "Green Match", "Black Match", "Neutral"),
        classes=("oolong", "green", "green", "black", "white"),
        vectors=torch.nn.functional.normalize(torch.eye(5), dim=1),
        _index_by_id={tea_id: index for index, tea_id in enumerate(ids)},
    )
    events = (
        TastingEvent(1, 1, "Liked", Verdict.LIKED, 2),
        TastingEvent(2, 2, "Disliked", Verdict.DISLIKED, 1),
    )
    snapshots = build_snapshots(catalog, events, sequences=(2,), recommendation_limit=2)
    metadata = tuple(
        TeaMetadata(
            tea_id=tea_id,
            title=catalog.titles[index],
            tea_class=catalog.classes[index],
            country="Origin",
            region="Region",
            oxidation="medium",
            roast="none",
            aroma=("floral",),
            taste=("sweet",),
        )
        for index, tea_id in enumerate(ids)
    )
    projection = Projection(
        catalogue=np.arange(15, dtype=np.float32).reshape(5, 3),
        positive_centroids={2: np.array([1.0, 2.0, 3.0], dtype=np.float32)},
        negative_centroids={2: np.array([4.0, 5.0, 6.0], dtype=np.float32)},
        explained_variance_ratio=(0.4, 0.2, 0.1),
    )

    artifact = build_artifact(catalog, metadata, snapshots, projection)
    destination = write_artifact(artifact, tmp_path / "artifact.json")
    stored = json.loads(destination.read_text(encoding="utf-8"))

    assert stored["source"]["ranking_space"] == "original normalized hypervectors"
    assert stored["source"]["projection_dimensions"] == 3
    assert stored["teas"][0] == {
        "id": "1",
        "title": "Liked",
        "class": "oolong",
        "country": "Origin",
        "region": "Region",
        "oxidation": "medium",
        "roast": "none",
        "aroma": ["floral"],
        "taste": ["sweet"],
        "position": [0.0, 1.0, 2.0],
    }
    assert {entry["class"]: entry["color"] for entry in stored["palette"]["classes"]} == (
        CLASS_COLORS
    )
    assert stored["checkpoints"][0]["tastings"][1]["verdict"] == "disliked"
    assert len(stored["checkpoints"][0]["recommendations"]) == 2
