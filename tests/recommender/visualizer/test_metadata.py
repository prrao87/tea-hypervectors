from __future__ import annotations

import json

import torch

from recommender.catalog import TeaCatalog
from recommender.visualizer.metadata import load_metadata


def test_loads_only_the_requested_hover_metadata(tmp_path) -> None:
    ids = (1, 2)
    catalog = TeaCatalog(
        ids=ids,
        titles=("Sencha", "Qimen"),
        classes=("green", "black"),
        vectors=torch.eye(2),
        _index_by_id={tea_id: index for index, tea_id in enumerate(ids)},
    )
    source = tmp_path / "metadata.json"
    source.write_text(
        json.dumps(
            {
                "products": [
                    {
                        "id": 1,
                        "title": "Sencha",
                        "category": "green",
                        "country": "Japan",
                        "region": "Shizuoka",
                        "oxidation": "none",
                        "roast": "none",
                        "aroma": ["grass", "sea air"],
                        "taste": ["umami"],
                        "description": "not exported",
                    },
                    {
                        "id": 2,
                        "title": "Qimen",
                        "category": "black",
                        "country": "China",
                        "region": None,
                        "oxidation": "full",
                        "roast": "none",
                        "aroma": ["cocoa"],
                        "taste": ["malty"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    records = load_metadata(catalog, source)

    assert records[0].as_artifact_record() == {
        "id": "1",
        "title": "Sencha",
        "class": "green",
        "country": "Japan",
        "region": "Shizuoka",
        "oxidation": "none",
        "roast": "none",
        "aroma": ["grass", "sea air"],
        "taste": ["umami"],
    }
    assert records[1].region is None
