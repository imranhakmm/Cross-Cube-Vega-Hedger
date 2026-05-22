"""Analytical, bumped, and Bartlett Greeks for vanilla swaptions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.special import ndtr

from cxvega.pricers import bachelier_price, black76_price, hagan_lognormal_sabr_vol


@dataclass(frozen=True)
class GreekTriple:
    """Delta, gamma, and vega container."""

    delta: float
    gamma: float
    vega: float


def _normal_pdf(x: float) -> float:
    return float(np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi))


def black76_greeks(
    forward: float,
    strike: float,
    expiry: float,
    vol: float,
    annuity: float = 1.0,
    option_type: str = "payer",
) -> GreekTriple:
    """Return analytical Black-76 delta, gamma, and vega."""

    tau = max(expiry, 1.0e-8)
    sigma = max(vol, 1.0e-8)
    sqrt_tau = sqrt(tau)
    sign = 1.0 if option_type == "payer" else -1.0
    d1 = (np.log(forward / strike) + 0.5 * sigma * sigma * tau) / (sigma * sqrt_tau)
    delta = annuity * ndtr(d1) if sign > 0.0 else annuity * (ndtr(d1) - 1.0)
    gamma = annuity * _normal_pdf(float(d1)) / (forward * sigma * sqrt_tau)
    vega = annuity * forward * sqrt_tau * _normal_pdf(float(d1))
    return GreekTriple(float(delta), float(gamma), float(vega))


def bachelier_greeks(
    forward: float,
    strike: float,
    expiry: float,
    normal_vol: float,
    annuity: float = 1.0,
    option_type: str = "payer",
) -> GreekTriple:
    """Return analytical Bachelier delta, gamma, and vega."""

    tau = max(expiry, 1.0e-8)
    sigma = max(normal_vol, 1.0e-8)
    sqrt_tau = sqrt(tau)
    sign = 1.0 if option_type == "payer" else -1.0
    d = (forward - strike) / (sigma * sqrt_tau)
    delta = annuity * ndtr(d) if sign > 0.0 else annuity * (ndtr(d) - 1.0)
    gamma = annuity * _normal_pdf(float(d)) / (sigma * sqrt_tau)
    vega = annuity * sqrt_tau * _normal_pdf(float(d))
    return GreekTriple(float(delta), float(gamma), float(vega))


def bumped_greeks(
    price_fn: Callable[[float, float], float],
    forward: float,
    vol: float,
    forward_bump: float = 1.0e-5,
    vol_bump: float = 1.0e-4,
) -> GreekTriple:
    """Return central finite-difference delta, gamma, and vega for a price function."""

    p_up = price_fn(forward + forward_bump, vol)
    p_mid = price_fn(forward, vol)
    p_down = price_fn(forward - forward_bump, vol)
    delta = (p_up - p_down) / (2.0 * forward_bump)
    gamma = (p_up - 2.0 * p_mid + p_down) / (forward_bump**2)
    vega = (price_fn(forward, vol + vol_bump) - price_fn(forward, vol - vol_bump)) / (
        2.0 * vol_bump
    )
    return GreekTriple(delta=float(delta), gamma=float(gamma), vega=float(vega))


def sabr_bartlett_delta(
    forward: float,
    strike: float,
    expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    annuity: float = 1.0,
) -> float:
    """Return Bartlett delta, including the SABR-implied alpha response to a forward move."""

    bump = max(1.0e-5, forward * 1.0e-4)
    alpha_slope = rho * nu / (forward**beta)

    def value(fwd: float, local_alpha: float) -> float:
        vol = float(hagan_lognormal_sabr_vol(fwd, strike, expiry, local_alpha, beta, rho, nu))
        return float(black76_price(fwd, strike, expiry, vol, annuity))

    up = value(forward + bump, max(alpha + alpha_slope * bump, 1.0e-6))
    down = value(forward - bump, max(alpha - alpha_slope * bump, 1.0e-6))
    return float((up - down) / (2.0 * bump))


def sabr_bartlett_vega(
    forward: float,
    strike: float,
    expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    annuity: float = 1.0,
) -> float:
    """Return the option sensitivity to SABR alpha, expressed as Bartlett vega."""

    bump = max(1.0e-5, alpha * 1.0e-4)

    def value(local_alpha: float) -> float:
        vol = float(hagan_lognormal_sabr_vol(forward, strike, expiry, local_alpha, beta, rho, nu))
        return float(black76_price(forward, strike, expiry, vol, annuity))

    return float((value(alpha + bump) - value(max(alpha - bump, 1.0e-8))) / (2.0 * bump))


def black_price_from_vol(
    forward: float,
    strike: float,
    expiry: float,
    annuity: float = 1.0,
) -> Callable[[float, float], float]:
    """Create a Black-76 price function for finite-difference checks."""

    def price(local_forward: float, local_vol: float) -> float:
        return float(black76_price(local_forward, strike, expiry, local_vol, annuity))

    return price


def bachelier_price_from_vol(
    forward: float,
    strike: float,
    expiry: float,
    annuity: float = 1.0,
) -> Callable[[float, float], float]:
    """Create a Bachelier price function for finite-difference checks."""

    def price(local_forward: float, local_vol: float) -> float:
        return float(bachelier_price(local_forward, strike, expiry, local_vol, annuity))

    return price

