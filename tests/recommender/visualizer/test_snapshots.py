from __future__ import annotations

import torch

from recommender.catalog import TeaCatalog
from recommender.events import TastingEvent, Verdict
from recommender.prototype import PreferencePrototype
from recommender.visualizer.snapshots import build_snapshots


def catalog() -> TeaCatalog:
    vectors = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.7, 0.3, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        ),
        dim=1,
    )
    ids = (1, 2, 3, 4, 5, 6)
    return TeaCatalog(
        ids=ids,
        titles=("First Like", "Second Like", "Dislike", "Match", "Other", "Far Away"),
        classes=("oolong", "oolong", "green", "oolong", "black", "green"),
        vectors=vectors,
        _index_by_id={tea_id: index for index, tea_id in enumerate(ids)},
    )


def test_builds_incremental_snapshots_without_changing_the_ranking_rule(monkeypatch) -> None:
    tea_catalog = catalog()
    events = (
        TastingEvent(1, 1, "First Like", Verdict.LIKED, 2),
        TastingEvent(2, 3, "Dislike", Verdict.DISLIKED, 1),
        TastingEvent(3, 2, "Second Like", Verdict.LIKED, 1),
    )

    updates: list[int] = []
    original_update = PreferencePrototype.update

    def tracked_update(self, event, vector) -> None:
        updates.append(event.sequence)
        original_update(self, event, vector)

    monkeypatch.setattr(PreferencePrototype, "update", tracked_update)

    first, second = build_snapshots(
        tea_catalog,
        events,
        sequences=(2, 3),
        recommendation_limit=2,
    )

    assert [event.title for event in first.new_likes] == ["First Like"]
    assert [event.title for event in second.new_likes] == ["Second Like"]
    assert updates == [1, 2, 3]
    assert first.positive_weight == 2
    assert second.positive_weight == 3
    assert first.recommendations[0].title == "Match"
    assert second.recommendations[0].title == "Match"
    torch.testing.assert_close(first.negative_prototype, second.negative_prototype)
