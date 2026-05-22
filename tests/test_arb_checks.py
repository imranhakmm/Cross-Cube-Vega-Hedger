import numpy as np

from cxvega.arb_checks import check_butterfly, check_calendar


def test_butterfly_check_flags_concavity() -> None:
    prices = np.array([[[3.0, 4.0, 2.0]]])
    violations = check_butterfly(prices, tol=1.0e-12)
    assert len(violations) == 1
    assert violations[0].kind == "butterfly"


def test_calendar_check_flags_decreasing_value() -> None:
    prices = np.array([[[2.0, 1.5]], [[1.9, 1.6]]])
    violations = check_calendar(prices, tol=1.0e-12)
    assert len(violations) == 1
    assert violations[0].kind == "calendar"
