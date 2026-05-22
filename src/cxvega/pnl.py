"""P&L attribution and reconciliation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DailyPnL:
    """Daily P&L decomposition."""

    edge: float
    theta: float
    delta: float
    vega_first_order: float
    vega_cross: float
    hedge_cost: float

    @property
    def total(self) -> float:
        """Return reconciled total P&L."""

        return (
            self.edge
            + self.theta
            + self.delta
            + self.vega_first_order
            + self.vega_cross
            - self.hedge_cost
        )


def reconcile_components(components: DailyPnL, total: float, atol: float = 1.0e-8) -> bool:
    """Check that attribution components sum to total P&L."""

    return bool(np.isclose(components.total, total, atol=atol))


def pnl_summary(values: NDArray[np.float64]) -> dict[str, float]:
    """Return mean, standard deviation, VaR, CVaR, and Sharpe-style statistics."""

    sorted_values = np.sort(values)
    idx_1 = max(0, int(0.01 * len(sorted_values)) - 1)
    idx_5 = max(0, int(0.05 * len(sorted_values)) - 1)
    var_1 = float(sorted_values[idx_1])
    var_5 = float(sorted_values[idx_5])
    cvar_1 = float(np.mean(sorted_values[: idx_1 + 1]))
    cvar_5 = float(np.mean(sorted_values[: idx_5 + 1]))
    std = float(np.std(values, ddof=1))
    mean = float(np.mean(values))
    return {
        "mean": mean,
        "std": std,
        "var_1": var_1,
        "var_5": var_5,
        "cvar_1": cvar_1,
        "cvar_5": cvar_5,
        "sharpe": mean / std if std > 0.0 else 0.0,
    }


def component_matrix(records: list[dict[str, float]]) -> NDArray[np.float64]:
    """Convert component dictionaries to an ordered attribution matrix."""

    keys = ["edge", "theta", "delta", "vega_first_order", "vega_cross", "hedge_cost"]
    matrix = np.array([[record[key] for key in keys] for record in records])
    return cast(NDArray[np.float64], matrix)
