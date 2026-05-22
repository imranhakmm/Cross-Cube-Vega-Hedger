import numpy as np

from cxvega.analytics import factor_exposure_from_vega
from cxvega.hedging import HedgeState, apply_hedge, factor_neutral_trades
from cxvega.simulator import factor_loadings


def test_factor_neutral_trades_reduce_exposure() -> None:
    expiries = np.array([0.25, 1.0, 2.0, 5.0, 10.0])
    tenors = np.array([1.0, 2.0, 5.0])
    loadings = factor_loadings(expiries, tenors)
    exposure = np.array([100.0, -40.0, 25.0, 15.0])
    pairs = [(i, j) for i in range(len(expiries)) for j in range(len(tenors))]
    trades, skew_trade = factor_neutral_trades(exposure, loadings, pairs)
    post = factor_exposure_from_vega(trades, loadings, skew_trade) + exposure
    assert np.linalg.norm(post) < np.linalg.norm(exposure) * 0.05


def test_bucketed_hedge_reduces_raw_vega() -> None:
    expiries = np.array([0.25, 1.0, 2.0, 5.0, 10.0])
    tenors = np.array([1.0, 2.0, 5.0])
    loadings = factor_loadings(expiries, tenors)
    state = HedgeState(np.ones((5, 3)) * 10.0, 5.0)
    result = apply_hedge("bucketed", state, loadings, expiries, 0.25, [(3, 1)])
    assert abs(float(np.sum(result.post_state.vega_grid))) < 1.0e-9
    assert result.cost > 0.0
