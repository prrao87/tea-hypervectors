"""The readable configuration for the tea encoder.

The manifest is the model: it states which fields exist, how ordered values are
represented, and how strongly each field influences similarity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

DIMENSIONS = 10_000
EMBEDDING_DIMENSIONS = 768


@dataclass(frozen=True)
class Weights:
    """Our domain hypothesis about what makes two teas similar.

    Aroma and taste receive 0.25 each because they carry the tea's sensory
    character and should jointly provide half of the signal. They remain
    separate so, for example, a floral aroma is not interchangeable with a
    floral taste.

    Class and oxidation receive 0.16 each. Both are strong style signals, and
    the overlap is intentional: an oolong's class matters, while its oxidation
    level distinguishes lightly oxidized examples from darker ones.

    Elevation receives up to 0.15 because nearby growing elevations should
    contribute useful similarity without overriding sensory character. Its
    actual weight is reduced by `elevation_confidence`.

    Roast receives only 0.03. It is a modifier rather than a tea's identity,
    and it is highly imbalanced in this snapshot: 144 of 166 teas have
    `roast="none"`. A larger weight would give most records an unhelpfully
    strong similarity bonus merely for being unroasted.

    The six weights sum to 1 so they read directly as shares of the similarity
    budget. They are explicit, testable heuristics rather than learned or
    universal constants.
    """

    aroma: float = 0.25
    taste: float = 0.25
    tea_class: float = 0.16
    oxidation: float = 0.16
    roast: float = 0.03
    elevation: float = 0.15

    @property
    def total(self) -> float:
        return (
            self.aroma + self.taste + self.tea_class + self.oxidation + self.roast + self.elevation
        )


@dataclass(frozen=True)
class EncoderManifest:
    global_seed: int = 13
    dimensions: int = DIMENSIONS
    vsa: str = "MAP"

    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = EMBEDDING_DIMENSIONS
    # Nomic requires a task prefix. `clustering` matches symmetric tea-to-tea
    # similarity better than the asymmetric search_document/search_query pair.
    task_prefix: str = "clustering: "

    class_vocabulary: tuple[str, ...] = ("black", "green", "oolong", "white", "yellow")
    oxidation_vocabulary: tuple[str, ...] = ("low", "medium", "high")
    # `medium` is absent from the current data but completes the ordered scale,
    # so adding roasted teas later will not renumber the existing levels.
    roast_vocabulary: tuple[str, ...] = ("none", "light", "medium", "heavy")

    elevation_min_meters: int = 0
    elevation_max_meters: int = 3_000
    elevation_levels: int = 61  # 50 m resolution, both endpoints included

    weights: Weights = Weights()

    def __post_init__(self) -> None:
        if abs(self.weights.total - 1.0) > 1e-9:
            raise ValueError(
                f"Field weights must form a budget summing to 1, got {self.weights.total}"
            )
        span = self.elevation_max_meters - self.elevation_min_meters
        if span <= 0 or self.elevation_levels < 2:
            raise ValueError("Elevation needs a positive range and at least 2 levels")

    @property
    def elevation_step_meters(self) -> float:
        span = self.elevation_max_meters - self.elevation_min_meters
        return span / (self.elevation_levels - 1)

    def elevation_weight(self, confidence: float) -> float:
        """Scale elevation's share of the budget by how much we trust it.

        Confidence changes how loudly elevation votes, not which elevation is
        being voted for. Confidence 0 means elevation contributes nothing.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"elevation_confidence must be in [0, 1], got {confidence}")
        return self.weights.elevation * confidence

    def to_dict(self) -> dict:
        return asdict(self)
