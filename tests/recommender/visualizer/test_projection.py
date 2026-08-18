from __future__ import annotations

import numpy as np
import pytest
import torch

from recommender.catalog import TeaCatalog
from recommender.events import TastingEvent, Verdict
from recommender.visualizer.projection import fit_pca_3d
from recommender.visualizer.snapshots import build_snapshots


def catalog() -> TeaCatalog:
    vectors = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.1],
                [0.8, 0.2, 0.3, 0.0],
                [0.0, 1.0, 0.4, 0.2],
                [0.9, 0.1, -0.5, 0.3],
                [0.7, 0.3, 0.8, -0.2],
                [-1.0, 0.0, 0.2, 0.4],
            ]
        ),
        dim=1,
    )
    ids = (1, 2, 3, 4, 5, 6)
    return TeaCatalog(
        ids=ids,
        titles=("First Like", "Second Like", "Dislike", "Match", "Other", "Far Away"),
        classes=("oolong", "oolong", "green", "oolong", "black", "white"),
        vectors=vectors,
        _index_by_id={tea_id: index for index, tea_id in enumerate(ids)},
    )


def test_projects_catalogue_and_prototypes_into_one_three_dimensional_space() -> None:
    tea_catalog = catalog()
    events = (
        TastingEvent(1, 1, "First Like", Verdict.LIKED, 2),
        TastingEvent(2, 3, "Dislike", Verdict.DISLIKED, 1),
        TastingEvent(3, 2, "Second Like", Verdict.LIKED, 1),
    )
    snapshots = build_snapshots(tea_catalog, events, sequences=(2, 3))

    projection = fit_pca_3d(tea_catalog, snapshots)

    assert projection.catalogue.shape == (6, 3)
    assert projection.axis_labels == ("PCA 1", "PCA 2", "PCA 3")
    assert projection.explained_variance_ratio is not None
    assert len(projection.explained_variance_ratio) == 3
    np.testing.assert_allclose(
        projection.negative_centroids[2],
        projection.catalogue[tea_catalog.index(3)],
        atol=1e-5,
    )


def test_rejects_a_catalogue_without_three_non_zero_components() -> None:
    tea_catalog = catalog()
    planar = TeaCatalog(
        ids=tea_catalog.ids,
        titles=tea_catalog.titles,
        classes=tea_catalog.classes,
        vectors=tea_catalog.vectors[:, :2],
        _index_by_id=dict(tea_catalog._index_by_id),
    )

    with pytest.raises(ValueError, match="3 non-zero principal components"):
        fit_pca_3d(planar, ())
