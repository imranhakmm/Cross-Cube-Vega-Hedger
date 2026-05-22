"""End-of-day vega hedging strategies for the market-making lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cxvega.analytics import factor_exposure_from_vega

StrategyName = Literal["delta_only", "single_tenor", "bucketed", "factor_neutral"]


@dataclass(frozen=True)
class HedgeState:
    """Inventory state relevant for vega hedging."""

    vega_grid: NDArray[np.float64]
    skew_exposure: float


@dataclass(frozen=True)
class HedgeResult:
    """Hedge trades and post-hedge state."""

    strategy: StrategyName
    trades: NDArray[np.float64]
    skew_trade: float
    cost: float
    post_state: HedgeState
    post_factor_exposure: NDArray[np.float64]


def expiry_bucket(expiry: float) -> int:
    """Map an expiry to the trader's coarse vega bucket."""

    if expiry <= 1.0:
        return 0
    if expiry <= 3.0:
        return 1
    if expiry <= 7.0:
        return 2
    return 3


def liquid_instrument_indices(
    n_expiry: int,
    n_tenor: int,
    liquid_expiry_indices: list[int],
    liquid_tenor_indices: list[int],
) -> list[tuple[int, int]]:
    """Return valid liquid hedge instrument indices."""

    pairs: list[tuple[int, int]] = []
    for i in liquid_expiry_indices:
        for j in liquid_tenor_indices:
            if 0 <= i < n_expiry and 0 <= j < n_tenor:
                pairs.append((i, j))
    return pairs


def factor_neutral_trades(
    exposure: NDArray[np.float64],
    loadings: NDArray[np.float64],
    liquid_pairs: list[tuple[int, int]],
    ridge: float = 1.0e-6,
) -> tuple[NDArray[np.float64], float]:
    """Solve the small quadratic hedge problem for factor-neutral trades."""

    if not liquid_pairs:
        raise ValueError("At least one liquid hedge instrument is required.")
    a = np.empty((4, len(liquid_pairs) + 1), dtype=float)
    for col, (i, j) in enumerate(liquid_pairs):
        a[:3, col] = loadings[i, j]
        a[3, col] = 0.0
    a[:, -1] = np.array([0.15, -0.05, 0.10, 1.0], dtype=float)
    lhs = a.T @ a + ridge * np.eye(a.shape[1])
    rhs = -a.T @ exposure
    solution = np.linalg.solve(lhs, rhs)
    grid_trades = np.zeros(loadings.shape[:2], dtype=float)
    for value, (i, j) in zip(solution[:-1], liquid_pairs, strict=True):
        grid_trades[i, j] += value
    return grid_trades, float(solution[-1])


def apply_hedge(
    strategy: StrategyName,
    state: HedgeState,
    loadings: NDArray[np.float64],
    expiries: NDArray[np.float64],
    hedge_cost_vol_bps: float,
    liquid_pairs: list[tuple[int, int]],
) -> HedgeResult:
    """Apply one end-of-day hedge strategy and return post-hedge inventory."""

    trades = np.zeros_like(state.vega_grid)
    skew_trade = 0.0
    if strategy == "delta_only":
        pass
    elif strategy == "single_tenor":
        center_expiry = int(np.argmin(np.abs(expiries - 5.0)))
        for tenor_index in range(state.vega_grid.shape[1]):
            trades[center_expiry, tenor_index] = -float(np.sum(state.vega_grid[:, tenor_index]))
    elif strategy == "bucketed":
        for bucket in range(4):
            expiry_indices = [
                i for i, expiry in enumerate(expiries) if expiry_bucket(float(expiry)) == bucket
            ]
            if not expiry_indices:
                continue
            bucket_median = float(np.median(expiries[expiry_indices]))
            representative = min(
                expiry_indices,
                key=lambda i: abs(float(expiries[i]) - bucket_median),
            )
            for tenor_index in range(state.vega_grid.shape[1]):
                trades[representative, tenor_index] -= float(
                    np.sum(state.vega_grid[expiry_indices, tenor_index])
                )
    elif strategy == "factor_neutral":
        exposure = factor_exposure_from_vega(state.vega_grid, loadings, state.skew_exposure)
        trades, skew_trade = factor_neutral_trades(exposure, loadings, liquid_pairs)
    else:
        raise ValueError(f"Unknown hedge strategy: {strategy}")

    post_grid = state.vega_grid + trades
    post_skew = state.skew_exposure + skew_trade
    post_exposure = factor_exposure_from_vega(post_grid, loadings, post_skew)
    cost = hedge_cost_vol_bps * 1.0e-4 * (float(np.sum(np.abs(trades))) + abs(skew_trade))
    return HedgeResult(
        strategy=strategy,
        trades=trades,
        skew_trade=skew_trade,
        cost=cost,
        post_state=HedgeState(post_grid, post_skew),
        post_factor_exposure=post_exposure,
    )


def strategy_names() -> tuple[StrategyName, ...]:
    """Return the four strategies in report order."""

    return ("delta_only", "single_tenor", "bucketed", "factor_neutral")
