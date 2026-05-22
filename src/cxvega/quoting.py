"""Avellaneda-Stoikov-style quoting controls in implied-volatility space."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray


def reservation_vol(
    mid_vol: float,
    inventory_factor_exposure: NDArray[np.float64],
    factor_covariance: NDArray[np.float64],
    instrument_loading: NDArray[np.float64],
    risk_aversion: float,
    horizon: float,
) -> float:
    """Return a heuristic reservation volatility adjusted for factor inventory."""

    inventory_charge = risk_aversion * float(
        inventory_factor_exposure @ factor_covariance @ instrument_loading
    ) * horizon
    return mid_vol - inventory_charge


def half_spread_vol_bps(
    base_half_spread_vol_bps: float,
    inventory_factor_exposure: NDArray[np.float64],
    factor_covariance: NDArray[np.float64],
    instrument_loading: NDArray[np.float64],
    risk_aversion: float,
) -> float:
    """Return inventory-risk-inflated half-spread in vol basis points."""

    factor_var = max(float(instrument_loading @ factor_covariance @ instrument_loading), 0.0)
    directional = abs(float(inventory_factor_exposure @ factor_covariance @ instrument_loading))
    inventory_addon = float(1.0e4 * risk_aversion * (0.15 * np.sqrt(factor_var) + directional))
    return float(base_half_spread_vol_bps + inventory_addon)


def spread_heatmap(
    base_half_spread_vol_bps: float,
    factor_covariance: NDArray[np.float64],
    instrument_loading: NDArray[np.float64],
    risk_aversion: float,
    grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate half-spread across a two-factor inventory grid."""

    out = np.empty((len(grid), len(grid)), dtype=float)
    loading = np.zeros(4, dtype=float)
    loading[: len(instrument_loading)] = instrument_loading
    for i, level_inventory in enumerate(grid):
        for j, skew_inventory in enumerate(grid):
            q = np.array([level_inventory, 0.0, 0.0, skew_inventory], dtype=float)
            out[i, j] = half_spread_vol_bps(
                base_half_spread_vol_bps,
                q,
                factor_covariance,
                loading,
                risk_aversion,
            )
    return cast(NDArray[np.float64], out)
