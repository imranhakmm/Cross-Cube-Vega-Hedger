import numpy as np

from cxvega.greeks import (
    bachelier_greeks,
    bachelier_price_from_vol,
    black76_greeks,
    black_price_from_vol,
    bumped_greeks,
    sabr_bartlett_delta,
    sabr_bartlett_vega,
)


def test_black76_greeks_match_bumps() -> None:
    forward = 0.035
    strike = 0.034
    expiry = 2.0
    vol = 0.30
    annuity = 4.5
    analytical = black76_greeks(forward, strike, expiry, vol, annuity)
    bumped = bumped_greeks(black_price_from_vol(forward, strike, expiry, annuity), forward, vol)
    assert np.isclose(analytical.delta, bumped.delta, rtol=1.0e-4)
    assert np.isclose(analytical.gamma, bumped.gamma, rtol=5.0e-3)
    assert np.isclose(analytical.vega, bumped.vega, rtol=1.0e-5)


def test_bachelier_greeks_match_bumps() -> None:
    forward = 0.031
    strike = 0.034
    expiry = 1.25
    vol = 0.008
    annuity = 6.0
    analytical = bachelier_greeks(forward, strike, expiry, vol, annuity)
    bumped = bumped_greeks(bachelier_price_from_vol(forward, strike, expiry, annuity), forward, vol)
    assert np.isclose(analytical.delta, bumped.delta, rtol=1.0e-4)
    assert np.isclose(analytical.gamma, bumped.gamma, rtol=5.0e-3)
    assert np.isclose(analytical.vega, bumped.vega, rtol=1.0e-5)


def test_bartlett_greeks_are_finite_and_nonzero() -> None:
    delta = sabr_bartlett_delta(0.035, 0.033, 2.0, 0.05, 0.5, -0.3, 0.55, 5.0)
    vega = sabr_bartlett_vega(0.035, 0.033, 2.0, 0.05, 0.5, -0.3, 0.55, 5.0)
    assert np.isfinite(delta)
    assert np.isfinite(vega)
    assert abs(delta) > 0.0
    assert vega > 0.0
