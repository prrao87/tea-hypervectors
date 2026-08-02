from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from recommender.events import EventValidationError, Verdict, load_events


def load_from_text(contents: str):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tastings.csv"
        path.write_text(contents, encoding="utf-8")
        return load_events(path)


def test_loads_and_sorts_chronological_events() -> None:
    events = load_from_text(
        "sequence,tea_id,title,verdict,strength,notes\n"
        "2,20,Later Tea,neutral,0,\n"
        "1,10,First Tea,LIKED,2,honey\n"
    )

    assert [event.sequence for event in events] == [1, 2]
    assert events[0].verdict is Verdict.LIKED
    assert events[0].strength == 2


def test_rejects_duplicate_sequence() -> None:
    with pytest.raises(EventValidationError, match="duplicate sequence"):
        load_from_text(
            "sequence,tea_id,title,verdict,strength,notes\n"
            "1,10,First Tea,liked,1,\n"
            "1,20,Other Tea,neutral,0,\n"
        )


def test_rejects_zero_strength_like() -> None:
    with pytest.raises(EventValidationError, match="positive strength"):
        load_from_text("sequence,tea_id,title,verdict,strength,notes\n1,10,First Tea,liked,0,\n")
