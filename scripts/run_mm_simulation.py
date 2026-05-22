from __future__ import annotations

import argparse

from cxvega.config import load_settings
from cxvega.reporting import generate_cube, generate_market_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the market-maker Monte Carlo simulation.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    cube_path = generate_cube(settings, steps=settings.market_maker.days)
    result = generate_market_results(settings, cube_path)
    print(f"wrote {len(result.path_pnl)} path P&L rows to outputs/simulations/path_pnl.csv")


if __name__ == "__main__":
    main()
