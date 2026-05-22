"""Swaption pricing and SABR implied-volatility utilities."""

from __future__ import annotations

from collections.abc import Callable
from math import sqrt
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import ndtr

OptionType = str

MIN_VOL = 1.0e-8
MIN_TIME = 1.0e-8


def _as_float_array(value: float | NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(value, dtype=float)


def _normal_pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return cast(NDArray[np.float64], np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi))


def _signed(option_type: OptionType) -> float:
    if option_type == "payer":
        return 1.0
    if option_type == "receiver":
        return -1.0
    raise ValueError("option_type must be 'payer' or 'receiver'.")


def black76_price(
    forward: float,
    strike: float | NDArray[np.float64],
    expiry: float,
    vol: float | NDArray[np.float64],
    annuity: float = 1.0,
    option_type: OptionType = "payer",
) -> float | NDArray[np.float64]:
    """Price a payer or receiver swaption under Black-76."""

    k = _as_float_array(strike)
    sigma = np.maximum(_as_float_array(vol), MIN_VOL)
    tau = max(expiry, MIN_TIME)
    sqrt_tau = sqrt(tau)
    sign = _signed(option_type)
    f = max(forward, 1.0e-12)
    d1 = (np.log(f / k) + 0.5 * sigma * sigma * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    value = annuity * sign * (f * ndtr(sign * d1) - k * ndtr(sign * d2))
    return float(value) if value.ndim == 0 else value


def bachelier_price(
    forward: float,
    strike: float | NDArray[np.float64],
    expiry: float,
    normal_vol: float | NDArray[np.float64],
    annuity: float = 1.0,
    option_type: OptionType = "payer",
) -> float | NDArray[np.float64]:
    """Price a payer or receiver swaption under the Bachelier normal model."""

    k = _as_float_array(strike)
    sigma_n = np.maximum(_as_float_array(normal_vol), MIN_VOL)
    tau = max(expiry, MIN_TIME)
    sqrt_tau = sqrt(tau)
    sign = _signed(option_type)
    d = (forward - k) / (sigma_n * sqrt_tau)
    value = annuity * (sign * (forward - k) * ndtr(sign * d) + sigma_n * sqrt_tau * _normal_pdf(d))
    return float(value) if value.ndim == 0 else value


def sabr_atm_lognormal_vol(
    forward: float,
    expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Return Hagan's ATM lognormal SABR volatility."""

    f = max(forward, 1.0e-12)
    one_minus_beta = 1.0 - beta
    f_factor = f ** one_minus_beta
    correction = (
        (one_minus_beta**2 / 24.0) * alpha**2 / (f ** (2.0 * one_minus_beta))
        + (rho * beta * nu * alpha) / (4.0 * f_factor)
        + ((2.0 - 3.0 * rho**2) * nu**2) / 24.0
    )
    return float(alpha / f_factor * (1.0 + correction * expiry))


def sabr_alpha_from_atm_lognormal(
    forward: float,
    expiry: float,
    atm_vol: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Invert Hagan's ATM formula by solving the exact cubic in ``alpha``."""

    f = max(forward, 1.0e-12)
    one_minus_beta = 1.0 - beta
    f_factor = f ** one_minus_beta
    a3 = expiry * (one_minus_beta**2 / 24.0) / (f ** (2.0 * one_minus_beta))
    a2 = expiry * (rho * beta * nu) / (4.0 * f_factor)
    a1 = 1.0 + expiry * ((2.0 - 3.0 * rho**2) * nu**2 / 24.0)
    a0 = -atm_vol * f_factor
    roots = np.roots(np.array([a3, a2, a1, a0], dtype=float))
    real_positive = [
        float(root.real) for root in roots if abs(root.imag) < 1.0e-8 and root.real > 0.0
    ]
    if not real_positive:
        raise ValueError("ATM SABR inversion failed to find a positive real alpha.")
    linear_guess = atm_vol * f_factor
    return min(real_positive, key=lambda value: abs(value - linear_guess))


def hagan_lognormal_sabr_vol(
    forward: float,
    strike: float | NDArray[np.float64],
    expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float | NDArray[np.float64]:
    """Return Hagan et al. (2002) lognormal SABR implied volatility."""

    k = _as_float_array(strike)
    f = max(forward, 1.0e-12)
    if np.any(k <= 0.0):
        raise ValueError("Lognormal SABR requires positive strikes.")

    one_minus_beta = 1.0 - beta
    log_fk = np.log(f / k)
    fk_beta = (f * k) ** (0.5 * one_minus_beta)
    log_correction = 1.0 + (one_minus_beta**2 / 24.0) * log_fk**2 + (
        one_minus_beta**4 / 1920.0
    ) * log_fk**4
    z = (nu / alpha) * fk_beta * log_fk
    root = np.sqrt(np.maximum(1.0 - 2.0 * rho * z + z * z, 1.0e-16))
    x_z = np.log((root + z - rho) / (1.0 - rho))
    z_over_x = np.divide(z, x_z, out=np.ones_like(z), where=np.abs(x_z) > 1.0e-7)
    time_correction = 1.0 + expiry * (
        (one_minus_beta**2 / 24.0) * alpha**2 / ((f * k) ** one_minus_beta)
        + (rho * beta * nu * alpha) / (4.0 * fk_beta)
        + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
    )
    vol = alpha / (fk_beta * log_correction) * z_over_x * time_correction
    atm = sabr_atm_lognormal_vol(f, expiry, alpha, beta, rho, nu)
    vol = np.where(np.abs(log_fk) < 1.0e-8, atm, vol)
    return float(vol) if vol.ndim == 0 else vol


def shifted_hagan_lognormal_sabr_vol(
    forward: float,
    strike: float | NDArray[np.float64],
    expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float,
) -> float | NDArray[np.float64]:
    """Evaluate Hagan lognormal SABR after an additive rate shift."""

    shifted_forward = forward + shift
    shifted_strike = _as_float_array(strike) + shift
    if shifted_forward <= 0.0 or np.any(shifted_strike <= 0.0):
        raise ValueError("Shifted SABR requires forward + shift and strike + shift to be positive.")
    return hagan_lognormal_sabr_vol(
        shifted_forward, shifted_strike, expiry, alpha, beta, rho, nu
    )


def _solve_vol(target_price: float, fn: Callable[[float], float], upper: float) -> float:
    low = MIN_VOL
    high = upper
    while fn(high) < target_price and high < 100.0:
        high *= 2.0
    return float(brentq(lambda vol: fn(vol) - target_price, low, high, maxiter=100))


def lognormal_to_normal_vol(
    forward: float,
    strike: float,
    expiry: float,
    lognormal_vol: float,
    annuity: float = 1.0,
) -> float:
    """Convert Black vol to normal vol by matching option value with Brent root finding."""

    target = float(black76_price(forward, strike, expiry, lognormal_vol, annuity))
    return _solve_vol(
        target,
        lambda vol: float(bachelier_price(forward, strike, expiry, vol, annuity)),
        upper=max(0.05, 5.0 * forward * lognormal_vol),
    )


def normal_to_lognormal_vol(
    forward: float,
    strike: float,
    expiry: float,
    normal_vol: float,
    annuity: float = 1.0,
) -> float:
    """Convert normal vol to Black vol by matching option value with Brent root finding."""

    target = float(bachelier_price(forward, strike, expiry, normal_vol, annuity))
    return _solve_vol(
        target,
        lambda vol: float(black76_price(forward, strike, expiry, vol, annuity)),
        upper=2.0,
    )
