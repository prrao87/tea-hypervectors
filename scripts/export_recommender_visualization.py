#!/usr/bin/env python3
"""Export the tea recommender's shared 3D PCA data as portable JSON.

uv run scripts/export_recommender_visualization.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from recommender.catalog import DEFAULT_DATABASE, DEFAULT_TABLE, TeaCatalog
from recommender.events import load_events
from recommender.visualizer import (
    build_artifact,
    build_snapshots,
    fit_pca_3d,
    load_metadata,
    write_artifact,
)
from recommender.visualizer.metadata import DEFAULT_METADATA
from recommender.visualizer.snapshots import DEFAULT_SNAPSHOT_SEQUENCES

DEFAULT_EVENTS = REPOSITORY / "inputs" / "preferences" / "tastings.csv"
DEFAULT_OUTPUT = (
    REPOSITORY / "src" / "recommender" / "visualizer" / "output" / "tea-recommender-3d.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copy-to", type=Path)
    parser.add_argument("--negative-penalty", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = TeaCatalog.load(args.database, args.table)
    events = load_events(args.events)
    snapshots = build_snapshots(
        catalog,
        events,
        sequences=DEFAULT_SNAPSHOT_SEQUENCES,
        negative_weight=args.negative_penalty,
    )
    projection = fit_pca_3d(catalog, snapshots)
    artifact = build_artifact(
        catalog,
        load_metadata(catalog, args.metadata),
        snapshots,
        projection,
        negative_penalty=args.negative_penalty,
    )
    output = write_artifact(artifact, args.output)
    print(output)

    explained = projection.explained_variance_ratio
    if explained is not None:
        print(
            "PCA explained variance: "
            f"PC1={explained[0]:.2%}  PC2={explained[1]:.2%}  "
            f"PC3={explained[2]:.2%}  total={sum(explained):.2%}"
        )

    if args.copy_to is not None:
        args.copy_to.parent.mkdir(parents=True, exist_ok=True)
        copied = Path(shutil.copy2(output, args.copy_to)).resolve()
        print(copied)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
