import numpy as np

from cxvega.analytics import recover_pca_factors
from cxvega.config import load_settings
from cxvega.simulator import simulate_cube_path


def test_pca_recovery_uses_loading_similarity_for_level_mode() -> None:
    settings = load_settings("configs/default.yaml")
    path = simulate_cube_path(settings, n_steps=252)
    recovery = recover_pca_factors(path)
    assert np.min(recovery.loading_similarity) > 0.90
