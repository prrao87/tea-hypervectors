from __future__ import annotations

import pytest
import torch

from recommender.catalog import TeaCatalog
from recommender.events import TastingEvent, Verdict
from recommender.prototype import PreferencePrototype
from recommender.ranking import rank_catalog


def catalog() -> TeaCatalog:
    vectors = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.9, 0.1, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        ),
        dim=1,
    )
    ids = (1, 2, 3, 4, 5)
    return TeaCatalog(
        ids=ids,
        titles=("Liked", "Neutral", "Disliked", "Candidate", "Other"),
        classes=("oolong", "oolong", "green", "black", "green"),
        vectors=vectors,
        _index_by_id={tea_id: index for index, tea_id in enumerate(ids)},
    )


def test_updates_prototypes_and_marks_neutral_as_tried() -> None:
    tea_catalog = catalog()
    events = (
        TastingEvent(1, 1, "Liked", Verdict.LIKED, 2),
        TastingEvent(2, 2, "Neutral", Verdict.NEUTRAL, 1),
        TastingEvent(3, 3, "Disliked", Verdict.DISLIKED, 1),
    )

    model = PreferencePrototype.from_events(tea_catalog, events)

    torch.testing.assert_close(model.positive, tea_catalog.vector(1))
    torch.testing.assert_close(model.negative, tea_catalog.vector(3))
    assert model.tried_ids == frozenset({1, 2, 3})
    assert model.positive_weight == 2
    assert model.negative_weight == 1


def test_ranking_excludes_tried_teas_and_exposes_score_parts() -> None:
    tea_catalog = catalog()
    events = (
        TastingEvent(1, 1, "Liked", Verdict.LIKED, 1),
        TastingEvent(2, 3, "Disliked", Verdict.DISLIKED, 1),
    )
    model = PreferencePrototype.from_events(tea_catalog, events)

    recommendations = rank_catalog(tea_catalog, model, limit=3, negative_weight=0.25)

    assert recommendations[0].tea_id == 4
    assert 1 not in [item.tea_id for item in recommendations]
    assert 3 not in [item.tea_id for item in recommendations]
    assert recommendations[0].negative_similarity is not None
    assert recommendations[0].score == pytest.approx(
        recommendations[0].positive_similarity - 0.25 * recommendations[0].negative_similarity
    )


def test_ranking_requires_a_positive_prototype() -> None:
    tea_catalog = catalog()
    event = TastingEvent(1, 3, "Disliked", Verdict.DISLIKED, 1)
    model = PreferencePrototype.from_events(tea_catalog, (event,))

    with pytest.raises(ValueError, match="liked tea"):
        rank_catalog(tea_catalog, model)
