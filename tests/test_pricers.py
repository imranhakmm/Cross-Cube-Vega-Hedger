import numpy as np

from cxvega.pricers import (
    bachelier_price,
    black76_price,
    hagan_lognormal_sabr_vol,
    lognormal_to_normal_vol,
    normal_to_lognormal_vol,
    sabr_alpha_from_atm_lognormal,
    sabr_atm_lognormal_vol,
    shifted_hagan_lognormal_sabr_vol,
)


def test_black76_put_call_parity() -> None:
    forward = 0.035
    strike = 0.033
    expiry = 2.0
    vol = 0.35
    annuity = 4.7
    payer = float(black76_price(forward, strike, expiry, vol, annuity, "payer"))
    receiver = float(black76_price(forward, strike, expiry, vol, annuity, "receiver"))
    assert np.isclose(payer - receiver, annuity * (forward - strike))


def test_bachelier_put_call_parity() -> None:
    forward = 0.035
    strike = 0.037
    expiry = 1.5
    vol = 0.008
    annuity = 7.5
    payer = float(bachelier_price(forward, strike, expiry, vol, annuity, "payer"))
    receiver = float(bachelier_price(forward, strike, expiry, vol, annuity, "receiver"))
    assert np.isclose(payer - receiver, annuity * (forward - strike))


def test_hagan_atm_matches_atm_formula() -> None:
    forward = 0.0325
    alpha = 0.05
    beta = 0.5
    rho = -0.35
    nu = 0.55
    expiry = 3.0
    direct = float(hagan_lognormal_sabr_vol(forward, forward, expiry, alpha, beta, rho, nu))
    atm = sabr_atm_lognormal_vol(forward, expiry, alpha, beta, rho, nu)
    assert np.isclose(direct, atm, rtol=1.0e-12)


def test_sabr_alpha_inverts_atm_formula() -> None:
    forward = 0.041
    expiry = 5.0
    beta = 0.5
    rho = -0.25
    nu = 0.45
    alpha = 0.055
    atm = sabr_atm_lognormal_vol(forward, expiry, alpha, beta, rho, nu)
    recovered = sabr_alpha_from_atm_lognormal(forward, expiry, atm, beta, rho, nu)
    assert np.isclose(recovered, alpha, rtol=1.0e-10)


def test_shifted_sabr_matches_unshifted_on_shifted_inputs() -> None:
    forward = -0.002
    strike = 0.001
    shift = 0.035
    shifted = float(
        shifted_hagan_lognormal_sabr_vol(forward, strike, 2.0, 0.04, 0.5, -0.3, 0.4, shift)
    )
    direct = float(
        hagan_lognormal_sabr_vol(forward + shift, strike + shift, 2.0, 0.04, 0.5, -0.3, 0.4)
    )
    assert np.isclose(shifted, direct)


def test_vol_conversion_round_trip_by_price_matching() -> None:
    forward = 0.035
    strike = 0.033
    expiry = 4.0
    black_vol = 0.28
    normal_vol = lognormal_to_normal_vol(forward, strike, expiry, black_vol)
    recovered = normal_to_lognormal_vol(forward, strike, expiry, normal_vol)
    assert np.isclose(recovered, black_vol, rtol=1.0e-8)


def test_sabr_vectorises_over_strike() -> None:
    strikes = np.array([0.025, 0.03, 0.035, 0.04])
    vols = hagan_lognormal_sabr_vol(0.033, strikes, 1.0, 0.045, 0.5, -0.25, 0.5)
    assert isinstance(vols, np.ndarray)
    assert vols.shape == strikes.shape
    assert np.all(vols > 0.0)
