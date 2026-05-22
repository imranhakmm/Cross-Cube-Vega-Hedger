import numpy as np

from cxvega.pnl import DailyPnL, pnl_summary, reconcile_components


def test_pnl_reconciles_exactly() -> None:
    components = DailyPnL(
        edge=10.0,
        theta=-1.0,
        delta=0.5,
        vega_first_order=3.0,
        vega_cross=-2.0,
        hedge_cost=0.25,
    )
    assert reconcile_components(components, 10.25)


def test_pnl_summary_tail_metrics() -> None:
    values = np.array([-5.0, -2.0, 0.0, 1.0, 4.0])
    stats = pnl_summary(values)
    assert stats["mean"] == float(np.mean(values))
    assert stats["var_5"] <= stats["mean"]
    assert "sharpe" in stats
