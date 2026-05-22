"""Central random-number factory for reproducible experiments."""

from __future__ import annotations

import hashlib
from typing import cast

import numpy as np
from numpy.typing import NDArray


def child_seed(master_seed: int, namespace: str) -> int:
    """Derive a stable 32-bit child seed from a master seed and namespace."""

    digest = hashlib.blake2b(f"{master_seed}:{namespace}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (2**32 - 1)


def rng_for(master_seed: int, namespace: str) -> np.random.Generator:
    """Create a NumPy generator for a named deterministic stream."""

    return np.random.default_rng(child_seed(master_seed, namespace))


def correlated_normals(
    rng: np.random.Generator, corr: NDArray[np.float64], size: tuple[int, ...]
) -> NDArray[np.float64]:
    """Draw standard normal vectors with the supplied correlation matrix."""

    chol = np.linalg.cholesky(corr)
    draws = rng.standard_normal((*size, corr.shape[0]))
    return cast(NDArray[np.float64], np.einsum("...j,ij->...i", draws, chol))
