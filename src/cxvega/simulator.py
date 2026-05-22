"""Multi-factor swaption cube simulator.

The simulator is deliberately explicit.  A latent three-factor OU process moves
the ATM log-volatility surface through smooth loadings: a level mode, a slope in
log-expiry, and a humped curvature mode.  Separate bounded OU dynamics drive
SABR rho, while log-nu follows another OU process.  Both skew/wing processes are
correlated with the level shock, reflecting the stylised fact that swaption skew
often steepens when rates volatility rises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from cxvega.config import Settings
from cxvega.curves import annuity_grid, forward_grid
from cxvega.pricers import hagan_lognormal_sabr_vol, sabr_alpha_from_atm_lognormal
from cxvega.rng import correlated_normals, rng_for


@dataclass(frozen=True)
class CubePath:
    """Simulated cube path with dense NumPy arrays and labelled axes."""

    times: NDArray[np.float64]
    expiries: NDArray[np.float64]
    tenors: NDArray[np.float64]
    expiry_labels: tuple[str, ...]
    tenor_labels: tuple[str, ...]
    strike_offsets_bps: NDArray[np.float64]
    forwards: NDArray[np.float64]
    annuities: NDArray[np.float64]
    strikes: NDArray[np.float64]
    loadings: NDArray[np.float64]
    factors: NDArray[np.float64]
    atm_vol: NDArray[np.float64]
    alpha: NDArray[np.float64]
    rho: NDArray[np.float64]
    nu: NDArray[np.float64]
    vols: NDArray[np.float64]
    beta: float

    @property
    def n_expiries(self) -> int:
        """Return the number of option expiries."""

        return len(self.expiries)

    @property
    def n_tenors(self) -> int:
        """Return the number of underlying swap tenors."""

        return len(self.tenors)

    @property
    def n_strikes(self) -> int:
        """Return the number of strike offsets."""

        return len(self.strike_offsets_bps)


def factor_loadings(
    expiries: NDArray[np.float64],
    tenors: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build smooth level, slope, and curvature loadings over the cube grid."""

    log_expiry = np.log(expiries)
    centered = (log_expiry - np.mean(log_expiry)) / np.std(log_expiry)
    hump = np.exp(-0.5 * ((log_expiry - np.log(2.0)) / 0.95) ** 2)
    hump = (hump - np.mean(hump)) / np.std(hump)
    tenor_tilt = (np.log1p(tenors) - np.mean(np.log1p(tenors))) / np.std(np.log1p(tenors))

    loadings = np.empty((len(expiries), len(tenors), 3), dtype=float)
    for i in range(len(expiries)):
        for j in range(len(tenors)):
            loadings[i, j, 0] = 1.0 + 0.08 * tenor_tilt[j]
            loadings[i, j, 1] = centered[i] * (1.0 + 0.05 * tenor_tilt[j])
            loadings[i, j, 2] = hump[i] * (1.0 - 0.10 * tenor_tilt[j])
    norms = np.sqrt(np.mean(loadings**2, axis=(0, 1)))
    return cast(NDArray[np.float64], loadings / norms)


def base_atm_lognormal_vol(
    expiries: NDArray[np.float64],
    tenors: NDArray[np.float64],
    forwards: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return a plausible static ATM Black-vol cube implied by normal-vol levels."""

    normal_bps = np.empty((len(expiries), len(tenors)), dtype=float)
    for i, expiry in enumerate(expiries):
        front_decay = np.exp(-expiry / 0.75)
        back_anchor = 78.0 + 4.0 * np.tanh(np.log1p(expiry) - 1.0)
        for j, tenor in enumerate(tenors):
            tenor_adjust = 5.0 * np.tanh(np.log1p(tenor) - 1.1)
            normal_bps[i, j] = back_anchor + 16.0 * front_decay + tenor_adjust
    normal_vol = normal_bps * 1.0e-4
    return cast(NDArray[np.float64], np.maximum(normal_vol / np.maximum(forwards, 0.01), 0.05))


def strike_grid(
    forwards: NDArray[np.float64],
    strike_offsets_bps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Create a positive strike grid from forward rates and absolute bp offsets."""

    strikes = forwards[:, :, None] + strike_offsets_bps[None, None, :] * 1.0e-4
    return cast(NDArray[np.float64], np.maximum(strikes, 0.001))


def _bounded_rho(raw: NDArray[np.float64]) -> NDArray[np.float64]:
    return -0.95 + 0.95 / (1.0 + np.exp(-raw))


def _rho_inverse(rho: float) -> float:
    clipped = min(-1.0e-5, max(-0.94999, rho))
    y = (clipped + 0.95) / 0.95
    return float(np.log(y / (1.0 - y)))


def _simulate_latent_factors(settings: Settings, n_steps: int) -> NDArray[np.float64]:
    sim = settings.simulator
    rng = rng_for(settings.seed, "cube-factors")
    dt = 1.0 / settings.calendar.steps_per_year
    corr = np.asarray(sim.ou_corr, dtype=float)
    shocks = correlated_normals(rng, corr, (n_steps,))
    factors = np.zeros((n_steps + 1, 3), dtype=float)
    kappa = np.asarray(sim.ou_kappa, dtype=float)
    vol = np.asarray(sim.ou_vol, dtype=float)
    for t in range(n_steps):
        factors[t + 1] = factors[t] - kappa * factors[t] * dt + vol * np.sqrt(dt) * shocks[t]
    return factors


def simulate_cube_path(settings: Settings, n_steps: int | None = None) -> CubePath:
    """Simulate a full SABR swaption cube path from the configured latent model."""

    cube = settings.cube
    sim = settings.simulator
    steps = n_steps or int(settings.calendar.steps_per_year * settings.calendar.simulation_years)
    expiries = np.asarray(cube.expiries_years, dtype=float)
    tenors = np.asarray(cube.tenors_years, dtype=float)
    offsets = np.asarray(cube.strike_offsets_bps, dtype=float)
    forwards = forward_grid(
        cube.expiries_years,
        cube.tenors_years,
        settings.rates.forward_level,
        settings.rates.forward_slope,
        settings.rates.forward_curvature,
    )
    annuities = annuity_grid(
        cube.expiries_years,
        cube.tenors_years,
        settings.rates.flat_discount_rate,
    )
    strikes = strike_grid(forwards, offsets)
    loadings = factor_loadings(expiries, tenors)
    base_atm = base_atm_lognormal_vol(expiries, tenors, forwards)
    factors = _simulate_latent_factors(settings, steps)
    times = np.arange(steps + 1, dtype=float) / settings.calendar.steps_per_year

    n_expiry, n_tenor = forwards.shape
    n_strike = len(offsets)
    atm_vol = np.empty((steps + 1, n_expiry, n_tenor), dtype=float)
    alpha = np.empty_like(atm_vol)
    rho = np.empty_like(atm_vol)
    nu = np.empty_like(atm_vol)
    vols = np.empty((steps + 1, n_expiry, n_tenor, n_strike), dtype=float)

    rng = rng_for(settings.seed, "sabr-params")
    dt = 1.0 / settings.calendar.steps_per_year
    rho_raw = np.full((n_expiry, n_tenor), _rho_inverse(sim.rho_mean), dtype=float)
    log_nu = np.full((n_expiry, n_tenor), np.log(sim.nu_mean), dtype=float)
    rho_mean_raw = _rho_inverse(sim.rho_mean)
    level_shocks = np.diff(factors[:, 0], prepend=factors[0, 0]) / max(
        sim.ou_vol[0] * np.sqrt(dt),
        1.0e-8,
    )

    for t in range(steps + 1):
        if t > 0:
            eps_rho = rng.standard_normal((n_expiry, n_tenor))
            eps_nu = rng.standard_normal((n_expiry, n_tenor))
            rho_driver = sim.rho_level_corr * level_shocks[t] + np.sqrt(
                max(1.0 - sim.rho_level_corr**2, 0.0)
            ) * eps_rho
            nu_driver = sim.nu_level_corr * level_shocks[t] + np.sqrt(
                max(1.0 - sim.nu_level_corr**2, 0.0)
            ) * eps_nu
            rho_raw += sim.rho_kappa * (rho_mean_raw - rho_raw) * dt
            rho_raw += sim.rho_vol * np.sqrt(dt) * rho_driver
            log_nu += sim.nu_kappa * (np.log(sim.nu_mean) - log_nu) * dt
            log_nu += sim.nu_vol * np.sqrt(dt) * nu_driver

        log_atm = np.log(base_atm) + np.einsum("ejk,k->ej", loadings, factors[t])
        atm_vol[t] = np.exp(log_atm)
        rho[t] = _bounded_rho(rho_raw)
        nu[t] = np.exp(log_nu)
        for i in range(n_expiry):
            for j in range(n_tenor):
                alpha[t, i, j] = sabr_alpha_from_atm_lognormal(
                    float(forwards[i, j]),
                    float(expiries[i]),
                    float(atm_vol[t, i, j]),
                    cube.beta,
                    float(rho[t, i, j]),
                    float(nu[t, i, j]),
                )
                vols[t, i, j] = hagan_lognormal_sabr_vol(
                    float(forwards[i, j]),
                    strikes[i, j],
                    float(expiries[i]),
                    float(alpha[t, i, j]),
                    cube.beta,
                    float(rho[t, i, j]),
                    float(nu[t, i, j]),
                )

    return CubePath(
        times=times,
        expiries=expiries,
        tenors=tenors,
        expiry_labels=tuple(cube.expiry_labels),
        tenor_labels=tuple(cube.tenor_labels),
        strike_offsets_bps=offsets,
        forwards=forwards,
        annuities=annuities,
        strikes=strikes,
        loadings=loadings,
        factors=factors,
        atm_vol=atm_vol,
        alpha=alpha,
        rho=rho,
        nu=nu,
        vols=vols,
        beta=cube.beta,
    )
