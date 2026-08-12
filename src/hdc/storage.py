"""The LanceDB persistence boundary: float32 to compute, float16 to store.

float16 is compact, approximate persistence. It halves the payload (20 KB per
10,000-d vector instead of 40 KB) and costs about three decimal digits of
precision per coordinate, which is far below the resolution at which cosine
rankings change. It is not a lossless round-trip, so nothing downstream should
assume a reloaded vector is bit-identical to the one that was written.

Everything TorchHD touches is float32, because it doesn't support float16
operations. The narrowing happens immediately before the write and is widened
again immediately after the read.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import torch
import torchhd
from lancedb.schema import vector
from torchhd import MAPTensor

from hdc.encoder import EncodedTea
from hdc.manifest import EncoderManifest

VECTOR_COLUMN = "vector_raw"


def arrow_schema(manifest: EncoderManifest) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.int64()),
            # Display-only columns, carried so query output is readable. They
            # are not encoder inputs.
            pa.field("title", pa.string()),
            pa.field("class", pa.string()),
            pa.field(VECTOR_COLUMN, vector(manifest.dimensions, value_type=pa.float16())),
        ]
    )


def to_storage(hypervector: torch.Tensor) -> np.ndarray:
    """Narrow one float32 hypervector to the stored float16 representation."""
    return hypervector.detach().numpy().astype(np.float16)


def from_storage(values, manifest: EncoderManifest) -> MAPTensor:
    """Widen a stored vector back into a float32 MAP tensor.

    Never run binding, bundling, or subtraction in float16 — widen here, once,
    on the way in.
    """
    array = np.asarray(values, dtype=np.float32)
    return torchhd.ensure_vsa_tensor(torch.from_numpy(array), vsa=manifest.vsa, dtype=torch.float32)


def to_row(
    encoded: EncodedTea,
    record: dict,
) -> dict:
    return {
        "id": encoded.tea_id,
        "title": record["title"],
        "class": record["class"],
        VECTOR_COLUMN: to_storage(encoded.bundle),
    }
