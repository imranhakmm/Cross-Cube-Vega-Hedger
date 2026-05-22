import numpy as np

from cxvega.calibration import calibrate_cube_per_slice, fit_sabr_slice
from cxvega.config import load_settings
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
