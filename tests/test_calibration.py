import numpy as np

from cxvega.arb_checks import check_cube_static_arbitrage
from cxvega.calibration import (
    apply_observation_noise,
    calibrate_cube_joint,
    calibrate_cube_per_slice,
    fit_sabr_slice,
)
from cxvega.config import load_settings
from cxvega.rng import rng_for
from cxvega.simulator import simulate_cube_path


def test_fit_sabr_slice_recovers_synthetic_smile() -> None:
    settings = load_settings("configs/default.yaml")
    path = simulate_cube_path(settings, n_steps=1)
    alpha, rho, nu, residual = fit_sabr_slice(
        float(path.forwards[2, 1]),
        path.strikes[2, 1],
        float(path.expiries[2]),
        path.vols[0, 2, 1],
        path.beta,
    )
    assert np.isclose(alpha, path.alpha[0, 2, 1], rtol=1.0e-4)
    assert np.isclose(rho, path.rho[0, 2, 1], atol=1.0e-3)
    assert np.isclose(nu, path.nu[0, 2, 1], rtol=1.0e-3)
    assert float(np.max(np.abs(residual))) < 1.0e-6


def test_per_slice_calibration_shapes() -> None:
    settings = load_settings("configs/default.yaml")
    path = simulate_cube_path(settings, n_steps=1)
    result = calibrate_cube_per_slice(
        path.forwards,
        path.strikes,
        path.expiries,
        path.vols[0],
        path.beta,
    )
    assert result.alpha.shape == (7, 4)
    assert result.vols.shape == (7, 4, 7)
    assert result.success
    assert float(np.max(np.abs(result.residuals))) < 1.0e-5


def test_observation_noise_is_seeded_and_multiplicative() -> None:
    settings = load_settings("configs/default.yaml")
    path = simulate_cube_path(settings, n_steps=1)
    noisy_a = apply_observation_noise(
        path.vols[0],
        rng_for(settings.seed, "calibration-observation-noise"),
        settings.observation_noise.calibration_vol_log_sigma,
    )
    noisy_b = apply_observation_noise(
        path.vols[0],
        rng_for(settings.seed, "calibration-observation-noise"),
        settings.observation_noise.calibration_vol_log_sigma,
    )
    assert np.allclose(noisy_a, noisy_b)
    assert np.all(noisy_a > 0.0)
    assert not np.allclose(noisy_a, path.vols[0])


def test_joint_calibration_reduces_material_calendar_violations() -> None:
    settings = load_settings("configs/default.yaml")
    path = simulate_cube_path(settings, n_steps=8)
    market = apply_observation_noise(
        path.vols[0],
        rng_for(settings.seed, "calibration-observation-noise"),
        settings.observation_noise.calibration_vol_log_sigma,
    )
    per_slice = calibrate_cube_per_slice(
        path.forwards, path.strikes, path.expiries, market, path.beta
    )
    joint = calibrate_cube_joint(
        path.forwards,
        path.annuities,
        path.strikes,
        path.expiries,
        market,
        path.beta,
        smoothness_weight=settings.calibration.joint_smoothness_weight,
        butterfly_weight=settings.calibration.joint_butterfly_weight,
        calendar_weight=settings.calibration.joint_calendar_weight,
        maxiter=30,
    )
    per_report = check_cube_static_arbitrage(
        path.forwards, path.annuities, path.strikes, path.expiries, per_slice.vols, tol=1.0e-4
    )
    joint_report = check_cube_static_arbitrage(
        path.forwards, path.annuities, path.strikes, path.expiries, joint.vols, tol=1.0e-4
    )
    assert joint_report.count < per_report.count
