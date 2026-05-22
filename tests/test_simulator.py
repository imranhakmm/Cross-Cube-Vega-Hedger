import numpy as np

from cxvega.config import load_settings
from cxvega.simulator import simulate_cube_path


def test_simulator_shapes_and_reproducibility() -> None:
    settings = load_settings("configs/default.yaml")
    path_a = simulate_cube_path(settings, n_steps=4)
    path_b = simulate_cube_path(settings, n_steps=4)
    assert path_a.vols.shape == (5, 7, 4, 7)
    assert path_a.atm_vol.shape == (5, 7, 4)
    assert path_a.loadings.shape == (7, 4, 3)
    assert np.allclose(path_a.vols, path_b.vols)
    assert np.all(path_a.vols > 0.0)
