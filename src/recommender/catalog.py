"""A narrow adapter from the stored tea table to preference-learning tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lancedb
import torch

from recommender.events import TastingEvent

DEFAULT_DATABASE = Path(__file__).resolve().parents[2] / "data" / "tea-hv-db"
DEFAULT_TABLE = "hypervectors"


@dataclass(frozen=True, slots=True)
class TeaCatalog:
    """Tea metadata and an L2-normalized vector matrix."""

    ids: tuple[int, ...]
    titles: tuple[str, ...]
    classes: tuple[str, ...]
    vectors: torch.Tensor
    _index_by_id: dict[int, int]

    @classmethod
    def load(
        cls,
        database: Path = DEFAULT_DATABASE,
        table_name: str = DEFAULT_TABLE,
    ) -> TeaCatalog:
        table = lancedb.connect(database).open_table(table_name)
        rows = (
            table.search()
            .select(["id", "title", "class", "vector_raw"])
            .limit(table.count_rows())
            .to_list()
        )
        if not rows:
            raise ValueError(f"catalog table {table_name!r} is empty")

        vectors = torch.stack(
            [torch.as_tensor(row["vector_raw"], dtype=torch.float32) for row in rows]
        )
        if not torch.isfinite(vectors).all():
            raise ValueError("catalog contains non-finite vector coordinates")
        norms = vectors.norm(dim=1)
        if torch.any(norms == 0):
            raise ValueError("catalog contains a zero-length vector")
        vectors = torch.nn.functional.normalize(vectors, dim=1)

        ids = tuple(int(row["id"]) for row in rows)
        return cls(
            ids=ids,
            titles=tuple(str(row["title"]) for row in rows),
            classes=tuple(str(row["class"]) for row in rows),
            vectors=vectors,
            _index_by_id={tea_id: index for index, tea_id in enumerate(ids)},
        )

    @property
    def dimensions(self) -> int:
        return int(self.vectors.shape[1])

    def index(self, tea_id: int) -> int:
        try:
            return self._index_by_id[tea_id]
        except KeyError as error:
            raise KeyError(f"tea_id {tea_id} is not present in the encoded catalog") from error

    def vector(self, tea_id: int) -> torch.Tensor:
        return self.vectors[self.index(tea_id)]

    def validate_events(self, events: tuple[TastingEvent, ...]) -> None:
        """Ensure preference rows resolve to the intended catalog items."""
        for event in events:
            index = self.index(event.tea_id)
            catalog_title = self.titles[index]
            if event.title != catalog_title:
                raise ValueError(
                    f"sequence {event.sequence}: tea_id {event.tea_id} is titled "
                    f"{catalog_title!r} in the catalog, not {event.title!r}"
                )
