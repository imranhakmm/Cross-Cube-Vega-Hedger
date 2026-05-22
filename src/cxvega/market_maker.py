"""Monte Carlo market-making loop for comparing vega hedging policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cxvega.analytics import factor_exposure_from_vega
from cxvega.config import Settings
from cxvega.hedging import HedgeState, apply_hedge, liquid_instrument_indices, strategy_names
from cxvega.pnl import DailyPnL, pnl_summary
from cxvega.quoting import half_spread_vol_bps
from cxvega.rng import rng_for
from cxvega.simulator import CubePath


@dataclass(frozen=True)
class MarketMakerResult:
    """Simulation outputs used by reports and figures."""

    path_pnl: pd.DataFrame
    daily_pnl: pd.DataFrame
    attribution: pd.DataFrame
    representative_path: pd.DataFrame
    summary: pd.DataFrame
    factor_covariance: NDArray[np.float64]


def factor_covariance(settings: Settings) -> NDArray[np.float64]:
    """Return the daily covariance matrix used by quoting and P&L simulation."""

    vols = np.asarray([*settings.simulator.ou_vol, 0.075], dtype=float)
    corr = np.array(
        [
            [1.00, 0.35, -0.15, -0.45],
            [0.35, 1.00, 0.10, -0.20],
            [-0.15, 0.10, 1.00, 0.12],
            [-0.45, -0.20, 0.12, 1.00],
        ],
        dtype=float,
    )
    annual = np.outer(vols, vols) * corr
    return cast(NDArray[np.float64], annual / settings.calendar.steps_per_year)


def _trade_vega(notional: float, annuity: float, forward: float, expiry: float) -> float:
    return float(notional * annuity * forward * np.sqrt(max(expiry, 1.0 / 252.0)) * 0.40)


def _strategy_label(strategy: str) -> str:
    return {
        "delta_only": "Delta-only",
        "single_tenor": "Single-tenor vega-flat",
        "bucketed": "Bucketed vega-flat",
        "factor_neutral": "Factor-neutral",
    }[strategy]


def run_market_maker_simulation(settings: Settings, cube_path: CubePath) -> MarketMakerResult:
    """Run the Monte Carlo market-making experiment for all hedge strategies."""

    mm = settings.market_maker
    rng = rng_for(settings.seed, "market-maker")
    n_paths = mm.n_paths
    n_days = mm.days
    cov = factor_covariance(settings)
    chol = np.linalg.cholesky(cov)
    loadings4 = np.zeros((*cube_path.loadings.shape[:2], 4), dtype=float)
    loadings4[:, :, :3] = cube_path.loadings
    liquid_pairs = liquid_instrument_indices(
        cube_path.n_expiries,
        cube_path.n_tenors,
        mm.liquid_expiry_indices,
        mm.liquid_tenor_indices,
    )
    strategies = strategy_names()

    path_rows: list[dict[str, float | int | str]] = []
    daily_rows: list[dict[str, float | int | str]] = []
    attribution_rows: list[dict[str, float | str]] = []
    representative_rows: list[dict[str, float | int | str]] = []

    for path_index in range(n_paths):
        states = {
            strategy: HedgeState(np.zeros(cube_path.loadings.shape[:2], dtype=float), 0.0)
            for strategy in strategies
        }
        totals = {strategy: 0.0 for strategy in strategies}
        components = {
            strategy: {
                "edge": 0.0,
                "theta": 0.0,
                "delta": 0.0,
                "vega_first_order": 0.0,
                "vega_cross": 0.0,
                "hedge_cost": 0.0,
            }
            for strategy in strategies
        }
        cumulative = {strategy: 0.0 for strategy in strategies}

        for day in range(n_days):
            shock = chol @ rng.standard_normal(4) * mm.shock_scale
            arrivals = int(rng.poisson(mm.daily_arrival_rate))
            daily_edge = {strategy: 0.0 for strategy in strategies}
            for _ in range(arrivals):
                expiry_index = int(rng.integers(0, cube_path.n_expiries))
                tenor_index = int(rng.integers(0, cube_path.n_tenors))
                strike_index = int(rng.integers(0, cube_path.n_strikes))
                side = 1.0 if rng.random() < 0.5 else -1.0
                notional = float(rng.lognormal(np.log(mm.notional_mean), mm.notional_sigma))
                vega = side * _trade_vega(
                    notional,
                    float(cube_path.annuities[expiry_index, tenor_index]),
                    float(cube_path.forwards[expiry_index, tenor_index]),
                    float(cube_path.expiries[expiry_index]),
                )
                moneyness = float(cube_path.strike_offsets_bps[strike_index]) / 100.0
                skew_vega = 0.35 * vega * moneyness
                node_loading = loadings4[expiry_index, tenor_index].copy()
                node_loading[3] = 0.60 * moneyness
                for strategy in strategies:
                    exposure = factor_exposure_from_vega(
                        states[strategy].vega_grid,
                        cube_path.loadings,
                        states[strategy].skew_exposure,
                    )
                    spread = half_spread_vol_bps(
                        mm.base_half_spread_vol_bps,
                        exposure,
                        cov,
                        node_loading,
                        mm.risk_aversion,
                    )
                    daily_edge[strategy] += spread * 1.0e-4 * abs(vega)
                    grid = states[strategy].vega_grid.copy()
                    grid[expiry_index, tenor_index] += vega
                    states[strategy] = HedgeState(grid, states[strategy].skew_exposure + skew_vega)

            for strategy in strategies:
                pre_exposure = factor_exposure_from_vega(
                    states[strategy].vega_grid,
                    cube_path.loadings,
                    states[strategy].skew_exposure,
                )
                theta = -1.5e-6 * float(np.sum(np.abs(states[strategy].vega_grid)))
                delta_scale = 0.006 * max(float(np.linalg.norm(pre_exposure)), 1.0)
                delta = float(rng.normal(0.0, delta_scale))
                vega_first = float(pre_exposure[:3] @ shock[:3])
                cross = float(pre_exposure[3] * shock[3])
                cross_scale = 0.010 * float(
                    np.linalg.norm(pre_exposure[:3]) * np.linalg.norm(shock[:3])
                )
                cross += float(rng.normal(0.0, cross_scale))
                hedge = apply_hedge(
                    strategy,
                    states[strategy],
                    cube_path.loadings,
                    cube_path.expiries,
                    mm.hedge_cost_vol_bps,
                    liquid_pairs,
                )
                states[strategy] = hedge.post_state
                pnl = DailyPnL(
                    edge=daily_edge[strategy],
                    theta=theta,
                    delta=delta,
                    vega_first_order=vega_first,
                    vega_cross=cross,
                    hedge_cost=hedge.cost,
                )
                totals[strategy] += pnl.total
                cumulative[strategy] += pnl.total
                components[strategy]["edge"] += pnl.edge
                components[strategy]["theta"] += pnl.theta
                components[strategy]["delta"] += pnl.delta
                components[strategy]["vega_first_order"] += pnl.vega_first_order
                components[strategy]["vega_cross"] += pnl.vega_cross
                components[strategy]["hedge_cost"] += pnl.hedge_cost
                daily_rows.append(
                    {
                        "path": path_index,
                        "day": day,
                        "strategy": _strategy_label(strategy),
                        "pnl": pnl.total,
                        "edge": pnl.edge,
                        "theta": pnl.theta,
                        "delta": pnl.delta,
                        "vega_first_order": pnl.vega_first_order,
                        "vega_cross": pnl.vega_cross,
                        "hedge_cost": pnl.hedge_cost,
                        "factor_norm": float(np.linalg.norm(hedge.post_factor_exposure)),
                    }
                )
                if path_index == 0:
                    representative_rows.append(
                        {
                            "day": day,
                            "strategy": _strategy_label(strategy),
                            "cumulative_pnl": cumulative[strategy],
                        }
                    )

        for strategy in strategies:
            label = _strategy_label(strategy)
            path_rows.append({"path": path_index, "strategy": label, "pnl": totals[strategy]})
            attribution_rows.append({"strategy": label, **components[strategy]})

    path_pnl = pd.DataFrame(path_rows)
    daily_pnl = pd.DataFrame(daily_rows)
    attribution = pd.DataFrame(attribution_rows).groupby("strategy", as_index=False).mean()
    representative_path = pd.DataFrame(representative_rows)
    summary_rows = []
    for strategy, group in path_pnl.groupby("strategy"):
        stats = pnl_summary(cast(NDArray[np.float64], group["pnl"].to_numpy(dtype=float)))
        summary_rows.append({"strategy": strategy, **stats})
    summary = pd.DataFrame(summary_rows).sort_values("mean", ascending=False)
    return MarketMakerResult(path_pnl, daily_pnl, attribution, representative_path, summary, cov)
