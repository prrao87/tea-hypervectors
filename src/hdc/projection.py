"""Bridge from a 768-dimensional Nomic embedding to a bipolar hypervector.

One fixed Rademacher (random +/-1) matrix is shared by aroma and taste so both
fields land in the same semantic basis. The projection does not create
information; it re-expresses the embedding's *direction* in a form the MAP
algebra can bind and unbind exactly.
"""

from __future__ import annotations

import numpy as np
import torch
from torchhd import MAPTensor

from hdc.core import HypervectorFactory
from hdc.manifest import EncoderManifest


def rademacher_matrix(manifest: EncoderManifest, hv: HypervectorFactory) -> torch.Tensor:
    """A fixed (768, 10000) matrix of -1 and +1 drawn from the manifest seed."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(hv.seed_for("projection", "sensory"))
    draws = torch.randint(
        0,
        2,
        (manifest.embedding_dimensions, manifest.dimensions),
        generator=generator,
        dtype=torch.int8,
    )
    return draws.to(torch.float32) * 2.0 - 1.0


def project_to_bipolar(
    embedding: np.ndarray,
    matrix: torch.Tensor,
    hv: HypervectorFactory,
    context: str,
) -> MAPTensor:
    """Random-hyperplane projection followed by a sign.

    Taking the sign trades some continuous precision for a valid bipolar MAP
    factor. Two semantically close embeddings still disagree on few signs, so
    angular similarity survives while binding stays exactly self-inverse.
    """
    vector = torch.from_numpy(np.asarray(embedding, dtype=np.float32))
    return hv.bipolarize(vector @ matrix, context)
