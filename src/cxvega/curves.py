"""Deterministic discount and forward swap curves used by the lab."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def discount_factor(
    t: float | NDArray[np.float64],
    flat_rate: float,
) -> float | NDArray[np.float64]:
    """Continuously compounded discount factor for maturity ``t``."""

    return np.exp(-flat_rate * np.asarray(t, dtype=float))


def swap_annuity(expiry: float, tenor: float, flat_rate: float, frequency: int = 1) -> float:
    """Return the fixed-leg annuity for a forward-starting swap."""

    n_payments = max(1, int(round(tenor * frequency)))
    accrual = 1.0 / frequency
    pay_times = expiry + accrual * np.arange(1, n_payments + 1, dtype=float)
    return float(accrual * np.sum(discount_factor(pay_times, flat_rate)))


def forward_swap_rate(
    expiry: float,
    tenor: float,
    level: float,
    slope: float,
    curvature: float,
) -> float:
    """Generate a smooth positive forward swap rate for a cube node."""

    x = np.log1p(expiry)
    y = np.log1p(tenor)
    rate = level + slope * (y - 1.0) + curvature * (x - 0.7) ** 2
    return float(max(0.0025, rate))


def forward_grid(
    expiries: list[float],
    tenors: list[float],
    level: float,
    slope: float,
    curvature: float,
) -> NDArray[np.float64]:
    """Return a forward swap-rate grid with shape ``(expiry, tenor)``."""

    grid = np.empty((len(expiries), len(tenors)), dtype=float)
    for i, expiry in enumerate(expiries):
        for j, tenor in enumerate(tenors):
            grid[i, j] = forward_swap_rate(expiry, tenor, level, slope, curvature)
    return grid


def annuity_grid(
    expiries: list[float],
    tenors: list[float],
    flat_rate: float,
) -> NDArray[np.float64]:
    """Return a swap annuity grid with shape ``(expiry, tenor)``."""

    grid = np.empty((len(expiries), len(tenors)), dtype=float)
    for i, expiry in enumerate(expiries):
        for j, tenor in enumerate(tenors):
            grid[i, j] = swap_annuity(expiry, tenor, flat_rate)
    return grid
