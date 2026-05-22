"""SABR calibration routines for per-slice and cube-constrained fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize

from cxvega.arb_checks import check_cube_static_arbitrage
from cxvega.pricers import hagan_lognormal_sabr_vol, sabr_alpha_from_atm_lognormal


@dataclass(frozen=True)
class CalibrationResult:
    """Fitted SABR cube and diagnostics."""

    alpha: NDArray[np.float64]
    rho: NDArray[np.float64]
    nu: NDArray[np.float64]
    vols: NDArray[np.float64]
    residuals: NDArray[np.float64]
    success: bool
    objective: float


def _rho_from_raw(raw: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    value = -0.95 + 0.95 / (1.0 + np.exp(-np.asarray(raw, dtype=float)))
    return float(value) if value.ndim == 0 else cast(NDArray[np.float64], value)


def _raw_from_rho(rho: float) -> float:
    clipped = min(-1.0e-5, max(-0.94999, rho))
    y = (clipped + 0.95) / 0.95
    return float(np.log(y / (1.0 - y)))


def fit_sabr_slice(
    forward: float,
    strikes: NDArray[np.float64],
    expiry: float,
    market_vols: NDArray[np.float64],
    beta: float,
    rho0: float = -0.35,
    nu0: float = 0.55,
) -> tuple[float, float, float, NDArray[np.float64]]:
    """Fit one SABR slice with ATM constrained exactly and free ``rho, nu``."""

    atm_index = int(np.argmin(np.abs(strikes - forward)))
    atm_vol = float(market_vols[atm_index])

    def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
        rho = float(_rho_from_raw(x[0]))
        nu = float(np.exp(x[1]))
        alpha = sabr_alpha_from_atm_lognormal(forward, expiry, atm_vol, beta, rho, nu)
        model = hagan_lognormal_sabr_vol(forward, strikes, expiry, alpha, beta, rho, nu)
        return cast(NDArray[np.float64], np.asarray(model, dtype=float) - market_vols)

    result = least_squares(
        residual,
        x0=np.array([_raw_from_rho(rho0), np.log(nu0)], dtype=float),
        method="lm",
        max_nfev=200,
    )
    rho = float(_rho_from_raw(result.x[0]))
    nu = float(np.exp(result.x[1]))
    alpha = sabr_alpha_from_atm_lognormal(forward, expiry, atm_vol, beta, rho, nu)
    final_residual = residual(result.x)
    return alpha, rho, nu, final_residual


def calibrate_cube_per_slice(
    forwards: NDArray[np.float64],
    strikes: NDArray[np.float64],
    expiries: NDArray[np.float64],
    market_vols: NDArray[np.float64],
    beta: float,
) -> CalibrationResult:
    """Fit SABR independently at each expiry-tenor slice."""

    n_expiry, n_tenor, n_strike = market_vols.shape
    alpha = np.empty((n_expiry, n_tenor), dtype=float)
    rho = np.empty_like(alpha)
    nu = np.empty_like(alpha)
    residuals = np.empty_like(market_vols)
    fitted = np.empty_like(market_vols)
    for i in range(n_expiry):
        for j in range(n_tenor):
            a, r, n, res = fit_sabr_slice(
                float(forwards[i, j]),
                strikes[i, j],
                float(expiries[i]),
                market_vols[i, j],
                beta,
            )
            alpha[i, j] = a
            rho[i, j] = r
            nu[i, j] = n
            residuals[i, j] = res
            fitted[i, j] = hagan_lognormal_sabr_vol(
                float(forwards[i, j]),
                strikes[i, j],
                float(expiries[i]),
                a,
                beta,
                r,
                n,
            )
    objective = float(np.mean(residuals**2))
    return CalibrationResult(alpha, rho, nu, fitted, residuals, True, objective)


def _smoothness_penalty(grid: NDArray[np.float64]) -> float:
    penalty = 0.0
    if grid.shape[0] > 1:
        penalty += float(np.mean(np.diff(grid, axis=0) ** 2))
    if grid.shape[1] > 1:
        penalty += float(np.mean(np.diff(grid, axis=1) ** 2))
    return penalty


def calibrate_cube_joint(
    forwards: NDArray[np.float64],
    annuities: NDArray[np.float64],
    strikes: NDArray[np.float64],
    expiries: NDArray[np.float64],
    market_vols: NDArray[np.float64],
    beta: float,
    smoothness_weight: float = 0.035,
    butterfly_weight: float = 0.50,
    calendar_weight: float = 0.25,
    maxiter: int = 80,
) -> CalibrationResult:
    """Fit the cube with residual, smoothness, and static-arbitrage penalties."""

    initial = calibrate_cube_per_slice(forwards, strikes, expiries, market_vols, beta)
    n_expiry, n_tenor, n_strike = market_vols.shape
    atm_indices = np.argmin(np.abs(strikes - forwards[:, :, None]), axis=2)
    atm_vols = market_vols[np.arange(n_expiry)[:, None], np.arange(n_tenor)[None, :], atm_indices]
    x0 = np.concatenate(
        [np.vectorize(_raw_from_rho)(initial.rho).ravel(), np.log(initial.nu).ravel()]
    )

    def unpack(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        raw_rho = x[: n_expiry * n_tenor].reshape(n_expiry, n_tenor)
        log_nu = x[n_expiry * n_tenor :].reshape(n_expiry, n_tenor)
        rho = cast(NDArray[np.float64], _rho_from_raw(raw_rho))
        nu = cast(NDArray[np.float64], np.exp(log_nu))
        return rho, nu

    def build(
        x: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        rho, nu = unpack(x)
        alpha = np.empty((n_expiry, n_tenor), dtype=float)
        vols = np.empty((n_expiry, n_tenor, n_strike), dtype=float)
        for i in range(n_expiry):
            for j in range(n_tenor):
                alpha[i, j] = sabr_alpha_from_atm_lognormal(
                    float(forwards[i, j]),
                    float(expiries[i]),
                    float(atm_vols[i, j]),
                    beta,
                    float(rho[i, j]),
                    float(nu[i, j]),
                )
                vols[i, j] = hagan_lognormal_sabr_vol(
                    float(forwards[i, j]),
                    strikes[i, j],
                    float(expiries[i]),
                    float(alpha[i, j]),
                    beta,
                    float(rho[i, j]),
                    float(nu[i, j]),
                )
        return alpha, rho, vols

    def objective(x: NDArray[np.float64]) -> float:
        _alpha, rho, vols = build(x)
        _, nu = unpack(x)
        residual_loss = float(np.mean((vols - market_vols) ** 2))
        smooth_loss = _smoothness_penalty(rho) + _smoothness_penalty(np.log(nu))
        report = check_cube_static_arbitrage(
            forwards, annuities, strikes, expiries, vols, tol=1.0e-7
        )
        arb_loss = sum(v.magnitude**2 for v in report.violations)
        return residual_loss + smoothness_weight * smooth_loss + (
            butterfly_weight + calendar_weight
        ) * arb_loss

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        options={"maxiter": maxiter, "ftol": 1.0e-10},
    )
    alpha, rho, vols = build(cast(NDArray[np.float64], result.x))
    _, nu = unpack(cast(NDArray[np.float64], result.x))
    residuals = vols - market_vols
    return CalibrationResult(
        alpha=alpha,
        rho=rho,
        nu=nu,
        vols=vols,
        residuals=residuals,
        success=bool(result.success),
        objective=float(result.fun),
    )
