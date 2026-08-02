"""Rank untried catalog teas against the learned preference prototypes."""

from __future__ import annotations

from dataclasses import dataclass

from recommender.catalog import TeaCatalog
from recommender.prototype import PreferencePrototype


@dataclass(frozen=True, slots=True)
class Recommendation:
    tea_id: int
    title: str
    tea_class: str
    score: float
    positive_similarity: float
    negative_similarity: float | None


def rank_catalog(
    catalog: TeaCatalog,
    model: PreferencePrototype,
    *,
    limit: int = 10,
    negative_weight: float = 0.25,
) -> tuple[Recommendation, ...]:
    """Return the strongest untried matches with an inspectable score split."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if negative_weight < 0:
        raise ValueError("negative_weight must be non-negative")
    if catalog.dimensions != model.dimensions:
        raise ValueError("catalog and preference prototype dimensions do not match")

    positive = model.positive
    if positive is None:
        raise ValueError("at least one liked tea is required before ranking")

    positive_similarities = catalog.vectors @ positive
    negative = model.negative
    if negative is None:
        negative_similarities = None
        scores = positive_similarities
    else:
        negative_similarities = catalog.vectors @ negative
        scores = positive_similarities - negative_weight * negative_similarities

    available = [index for index, tea_id in enumerate(catalog.ids) if tea_id not in model.tried_ids]
    ordered = sorted(available, key=lambda index: float(scores[index]), reverse=True)[:limit]

    return tuple(
        Recommendation(
            tea_id=catalog.ids[index],
            title=catalog.titles[index],
            tea_class=catalog.classes[index],
            score=float(scores[index]),
            positive_similarity=float(positive_similarities[index]),
            negative_similarity=(
                float(negative_similarities[index]) if negative_similarities is not None else None
            ),
        )
        for index in ordered
    )
