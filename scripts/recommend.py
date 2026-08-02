#!/usr/bin/env python3
"""Build preference prototypes from tasting history and rank untried teas.

uv run scripts/recommend.py
uv run scripts/recommend.py --limit 15 --negative-weight 0.25
uv run scripts/recommend.py --at-sequence 6
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from recommender.catalog import DEFAULT_DATABASE, DEFAULT_TABLE, TeaCatalog
from recommender.events import Verdict, load_events
from recommender.prototype import PreferencePrototype
from recommender.ranking import rank_catalog

DEFAULT_EVENTS = REPOSITORY / "inputs" / "preferences" / "tastings.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--negative-weight",
        type=float,
        default=0.25,
        help="Penalty applied to similarity with the disliked-tea prototype",
    )
    parser.add_argument(
        "--at-sequence",
        type=int,
        help="Replay only events up to and including this sequence number",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events = load_events(args.events)
    if args.at_sequence is not None:
        events = tuple(event for event in events if event.sequence <= args.at_sequence)
        if not events:
            raise ValueError("no tasting events remain at the requested sequence")

    catalog = TeaCatalog.load(args.database, args.table)
    model = PreferencePrototype.from_events(catalog, events)
    recommendations = rank_catalog(
        catalog,
        model,
        limit=args.limit,
        negative_weight=args.negative_weight,
    )

    counts = Counter(event.verdict for event in events)
    print(
        f"Tastings: {len(events)}  "
        f"liked={counts[Verdict.LIKED]}  "
        f"disliked={counts[Verdict.DISLIKED]}  "
        f"neutral={counts[Verdict.NEUTRAL]}"
    )
    print(
        f"Prototype votes: positive={model.positive_weight:g}  "
        f"negative={model.negative_weight:g}  "
        f"negative penalty={args.negative_weight:g}"
    )
    print()
    print(f"{'score':>8}  {'positive':>8}  {'negative':>8}  {'class':<8} title")
    print("-" * 78)
    for recommendation in recommendations:
        negative = (
            f"{recommendation.negative_similarity:.4f}"
            if recommendation.negative_similarity is not None
            else "n/a"
        )
        print(
            f"{recommendation.score:>8.4f}  "
            f"{recommendation.positive_similarity:>8.4f}  "
            f"{negative:>8}  "
            f"{recommendation.tea_class:<8} "
            f"{recommendation.title}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
