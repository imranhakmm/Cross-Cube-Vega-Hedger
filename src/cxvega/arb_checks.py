"""Static arbitrage diagnostics for simulated and calibrated swaption cubes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cxvega.pricers import black76_price


@dataclass(frozen=True)
class ArbViolation:
    """Single static-arbitrage violation record."""

    kind: str
    expiry_index: int
    tenor_index: int
    strike_index: int
    magnitude: float


@dataclass(frozen=True)
class ArbReport:
    """Structured collection of arbitrage diagnostics."""

    violations: tuple[ArbViolation, ...]

    @property
    def count(self) -> int:
        """Number of detected violations."""

        return len(self.violations)

    def by_kind(self) -> dict[str, int]:
        """Return violation counts grouped by diagnostic type."""

        counts: dict[str, int] = {}
        for violation in self.violations:
            counts[violation.kind] = counts.get(violation.kind, 0) + 1
        return counts


def cube_prices_from_vols(
    forwards: NDArray[np.float64],
    annuities: NDArray[np.float64],
    strikes: NDArray[np.float64],
    expiries: NDArray[np.float64],
    vols: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert a vol cube with shape ``(expiry, tenor, strike)`` into payer prices."""

    prices = np.empty_like(vols)
    for i, expiry in enumerate(expiries):
        for j in range(forwards.shape[1]):
            prices[i, j, :] = black76_price(
                float(forwards[i, j]),
                strikes[i, j, :],
                float(expiry),
                vols[i, j, :],
                float(annuities[i, j]),
            )
    return prices


def check_butterfly(prices: NDArray[np.float64], tol: float = 1.0e-8) -> list[ArbViolation]:
    """Check convexity of payer prices across strikes within each slice."""

    violations: list[ArbViolation] = []
    second_diff = prices[:, :, :-2] - 2.0 * prices[:, :, 1:-1] + prices[:, :, 2:]
    indices = np.argwhere(second_diff < -tol)
    for expiry_index, tenor_index, local_strike_index in indices:
        magnitude = float(-second_diff[expiry_index, tenor_index, local_strike_index])
        violations.append(
            ArbViolation(
                "butterfly",
                int(expiry_index),
                int(tenor_index),
                int(local_strike_index + 1),
                magnitude,
            )
        )
    return violations


def check_calendar(prices: NDArray[np.float64], tol: float = 1.0e-8) -> list[ArbViolation]:
    """Check monotonicity of payer prices across option expiries at common strike index."""

    violations: list[ArbViolation] = []
    diff = np.diff(prices, axis=0)
    indices = np.argwhere(diff < -tol)
    for expiry_index, tenor_index, strike_index in indices:
        magnitude = float(-diff[expiry_index, tenor_index, strike_index])
        violations.append(
            ArbViolation(
                "calendar",
                int(expiry_index + 1),
                int(tenor_index),
                int(strike_index),
                magnitude,
            )
        )
    return violations


def check_cube_static_arbitrage(
    forwards: NDArray[np.float64],
    annuities: NDArray[np.float64],
    strikes: NDArray[np.float64],
    expiries: NDArray[np.float64],
    vols: NDArray[np.float64],
    tol: float = 1.0e-8,
) -> ArbReport:
    """Run butterfly and calendar checks and return a structured report."""

    prices = cube_prices_from_vols(forwards, annuities, strikes, expiries, vols)
    violations = [*check_butterfly(prices, tol=tol), *check_calendar(prices, tol=tol)]
    return ArbReport(tuple(violations))
