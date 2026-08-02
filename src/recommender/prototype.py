"""Incremental positive and negative prototype hypervectors."""

from __future__ import annotations

import torch

from recommender.catalog import TeaCatalog
from recommender.events import TastingEvent, Verdict


def _unit(vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim != 1:
        raise ValueError(f"expected one vector, got shape {tuple(vector.shape)}")
    if not torch.isfinite(vector).all():
        raise ValueError("preference update vector contains non-finite coordinates")
    norm = vector.norm()
    if norm == 0:
        raise ValueError("preference update vector has zero length")
    return vector.to(torch.float32) / norm


class PreferencePrototype:
    """Online learner whose state is the weighted sum of normalized tea vectors."""

    def __init__(self, dimensions: int) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._positive_sum = torch.zeros(dimensions, dtype=torch.float32)
        self._negative_sum = torch.zeros(dimensions, dtype=torch.float32)
        self._tried_ids: set[int] = set()
        self.positive_weight = 0.0
        self.negative_weight = 0.0
        self.events_seen = 0

    @classmethod
    def from_events(
        cls,
        catalog: TeaCatalog,
        events: tuple[TastingEvent, ...],
    ) -> PreferencePrototype:
        catalog.validate_events(events)
        model = cls(catalog.dimensions)
        for event in events:
            model.update(event, catalog.vector(event.tea_id))
        return model

    @property
    def dimensions(self) -> int:
        return int(self._positive_sum.shape[0])

    @property
    def tried_ids(self) -> frozenset[int]:
        return frozenset(self._tried_ids)

    @property
    def positive(self) -> torch.Tensor | None:
        return self._finalize(self._positive_sum, self.positive_weight)

    @property
    def negative(self) -> torch.Tensor | None:
        return self._finalize(self._negative_sum, self.negative_weight)

    @staticmethod
    def _finalize(total: torch.Tensor, weight: float) -> torch.Tensor | None:
        if weight == 0:
            return None
        norm = total.norm()
        if norm == 0:
            raise ValueError("prototype contributions cancelled to a zero vector")
        return total / norm

    def update(self, event: TastingEvent, vector: torch.Tensor) -> None:
        """Apply one event immediately; neutral events only mark a tea as tried."""
        if vector.shape != (self.dimensions,):
            raise ValueError(
                f"tea vector has shape {tuple(vector.shape)}, expected {(self.dimensions,)}"
            )

        # The user provides /inputs/preferences/tastings.csv with past teas tasted.
        # Strength 0 casts no vote, 1 casts one vote, and 2 casts two votes.
        # Neutral and disliked teas are both filtered as already tried, but only
        # disliked teas vote toward the negative preference summary.
        self._tried_ids.add(event.tea_id)
        self.events_seen += 1
        if event.verdict is Verdict.NEUTRAL:
            return

        contribution = event.strength * _unit(vector)
        if event.verdict is Verdict.LIKED:
            self._positive_sum += contribution
            self.positive_weight += event.strength
        else:
            self._negative_sum += contribution
            self.negative_weight += event.strength
