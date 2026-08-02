"""Typed tasting events loaded from the append-only preference CSV."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

REQUIRED_COLUMNS = ("sequence", "tea_id", "title", "verdict", "strength", "notes")


class EventValidationError(ValueError):
    """A tasting row cannot be used safely by the preference learner."""


class Verdict(StrEnum):
    LIKED = "liked"
    DISLIKED = "disliked"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class TastingEvent:
    """One chronological observation about a tea."""

    sequence: int
    tea_id: int
    title: str
    verdict: Verdict
    strength: float
    notes: str = ""

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise EventValidationError("sequence must be a positive integer")
        if self.tea_id < 1:
            raise EventValidationError("tea_id must be a positive integer")
        if not self.title:
            raise EventValidationError("title must not be empty")
        if not math.isfinite(self.strength) or self.strength < 0:
            raise EventValidationError("strength must be a finite, non-negative number")
        if self.verdict is not Verdict.NEUTRAL and self.strength == 0:
            raise EventValidationError("liked and disliked events must have positive strength")


def _event_from_row(row: dict[str, str], line_number: int) -> TastingEvent:
    try:
        verdict = Verdict(row["verdict"].strip().lower())
    except ValueError as error:
        choices = ", ".join(verdict.value for verdict in Verdict)
        raise EventValidationError(
            f"line {line_number}: verdict must be one of {choices}"
        ) from error

    try:
        return TastingEvent(
            sequence=int(row["sequence"]),
            tea_id=int(row["tea_id"]),
            title=row["title"].strip(),
            verdict=verdict,
            strength=float(row["strength"]),
            notes=row["notes"].strip(),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, EventValidationError):
            raise
        raise EventValidationError(f"line {line_number}: {error}") from error


def load_events(path: Path) -> tuple[TastingEvent, ...]:
    """Load, validate, and order the tasting history by sequence."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise EventValidationError(f"missing CSV columns: {', '.join(missing)}")

        events = tuple(
            _event_from_row(row, line_number) for line_number, row in enumerate(reader, 2)
        )

    if not events:
        raise EventValidationError("tasting history is empty")

    sequences = [event.sequence for event in events]
    duplicates = sorted({value for value in sequences if sequences.count(value) > 1})
    if duplicates:
        rendered = ", ".join(str(value) for value in duplicates)
        raise EventValidationError(f"duplicate sequence values: {rendered}")

    return tuple(sorted(events, key=lambda event: event.sequence))
