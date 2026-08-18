"""Build faithful recommender snapshots for visualization."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch

from recommender.catalog import TeaCatalog
from recommender.events import TastingEvent, Verdict
from recommender.prototype import PreferencePrototype
from recommender.ranking import Recommendation, rank_catalog

DEFAULT_SNAPSHOT_SEQUENCES = (5, 10, 15)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The learner and its outputs after one prefix of the tasting history."""

    sequence: int
    events: tuple[TastingEvent, ...]
    new_likes: tuple[TastingEvent, ...]
    positive_prototype: torch.Tensor
    negative_prototype: torch.Tensor | None
    positive_weight: float
    negative_weight: float
    tried_ids: frozenset[int]
    recommendations: tuple[Recommendation, ...]

    @property
    def positive_centroid(self) -> torch.Tensor:
        """Return the positive prototype direction used by cosine similarity.

        The learner normalizes its weighted sum before ranking. This unit
        hypervector points in the same direction as the weighted centroid of
        the liked tea hypervectors, so a linear projection places both identically.
        """
        return self.positive_prototype

    @property
    def verdict_counts(self) -> Counter[Verdict]:
        return Counter(event.verdict for event in self.events)


def build_snapshots(
    catalog: TeaCatalog,
    events: tuple[TastingEvent, ...],
    *,
    sequences: tuple[int, ...] = DEFAULT_SNAPSHOT_SEQUENCES,
    recommendation_limit: int = 3,
    negative_weight: float = 0.25,
) -> tuple[Snapshot, ...]:
    """Replay chronological events and capture selected learner states."""
    if not sequences or any(sequence < 1 for sequence in sequences):
        raise ValueError("snapshot sequences must contain positive integers")
    if tuple(sorted(set(sequences))) != sequences:
        raise ValueError("snapshot sequences must be unique and increasing")

    catalog.validate_events(events)
    model = PreferencePrototype(catalog.dimensions)
    snapshots: list[Snapshot] = []
    applied_events: list[TastingEvent] = []
    previous_sequence = 0
    for sequence in sequences:
        new_events = tuple(
            event for event in events if previous_sequence < event.sequence <= sequence
        )
        for event in new_events:
            model.update(event, catalog.vector(event.tea_id))
        applied_events.extend(new_events)
        stage_events = tuple(applied_events)
        if not stage_events:
            raise ValueError(f"no tasting events exist at sequence {sequence}")
        positive_prototype = model.positive
        if positive_prototype is None:
            raise ValueError(f"snapshot {sequence} has no liked tea")

        recommendations = rank_catalog(
            catalog,
            model,
            limit=recommendation_limit,
            negative_weight=negative_weight,
        )
        new_likes = tuple(event for event in new_events if event.verdict is Verdict.LIKED)
        snapshots.append(
            Snapshot(
                sequence=sequence,
                events=stage_events,
                new_likes=new_likes,
                positive_prototype=positive_prototype.clone(),
                negative_prototype=(model.negative.clone() if model.negative is not None else None),
                positive_weight=model.positive_weight,
                negative_weight=model.negative_weight,
                tried_ids=model.tried_ids,
                recommendations=recommendations,
            )
        )
        previous_sequence = sequence

    return tuple(snapshots)
