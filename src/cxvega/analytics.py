"""PCA and factor-recovery analytics for swaption cube dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from cxvega.simulator import CubePath


@dataclass(frozen=True)
class PCARecovery:
    """Recovered PCA factors aligned to the simulator loadings."""

    explained_variance: NDArray[np.float64]
    recovered_loadings: NDArray[np.float64]
    true_loadings: NDArray[np.float64]
    loading_similarity: NDArray[np.float64]
    scores: NDArray[np.float64]


def _cosine_similarity(lhs: NDArray[np.float64], rhs: NDArray[np.float64]) -> float:
    """Return cosine similarity between two loading vectors."""

    denominator = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denominator == 0.0:
        return 0.0
    return float(lhs @ rhs / denominator)


def recover_pca_factors(path: CubePath, n_components: int = 3) -> PCARecovery:
    """Recover low-dimensional cube factors from simulated ATM log-vol changes."""

    log_atm = np.log(path.atm_vol)
    returns = np.diff(log_atm, axis=0).reshape(path.atm_vol.shape[0] - 1, -1)
    returns -= np.mean(returns, axis=0, keepdims=True)
    pca = PCA(n_components=n_components, random_state=0)
    scores = cast(NDArray[np.float64], pca.fit_transform(returns))
    components = cast(NDArray[np.float64], pca.components_).T
    true = path.loadings.reshape(-1, 3)
    aligned = np.empty_like(components)
    similarity_matrix = np.empty((n_components, n_components), dtype=float)
    for component_index in range(n_components):
        for factor_index in range(n_components):
            similarity_matrix[component_index, factor_index] = _cosine_similarity(
                components[:, component_index],
                true[:, factor_index],
            )

    best_assignment = max(
        permutations(range(n_components)),
        key=lambda assignment: sum(
            abs(similarity_matrix[component_index, factor_index])
            for component_index, factor_index in enumerate(assignment)
        ),
    )
    similarities = np.empty(n_components, dtype=float)
    for component_index, factor_index in enumerate(best_assignment):
        similarity = similarity_matrix[component_index, factor_index]
        sign = 1.0 if similarity >= 0.0 else -1.0
        aligned[:, factor_index] = sign * components[:, component_index]
        similarities[factor_index] = abs(similarity)
    return PCARecovery(
        explained_variance=cast(NDArray[np.float64], pca.explained_variance_ratio_),
        recovered_loadings=aligned.reshape(path.n_expiries, path.n_tenors, n_components),
        true_loadings=path.loadings,
        loading_similarity=similarities,
        scores=scores,
    )


def factor_exposure_from_vega(
    vega_grid: NDArray[np.float64],
    loadings: NDArray[np.float64],
    skew_exposure: float = 0.0,
) -> NDArray[np.float64]:
    """Project node-level vega into level/slope/curvature plus skew exposure."""

    first_three = np.einsum("ej,ejk->k", vega_grid, loadings)
    return cast(NDArray[np.float64], np.append(first_three, skew_exposure))
