from __future__ import annotations

import argparse

from cxvega.config import load_settings
from cxvega.reporting import calibration_diagnostics, generate_cube


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SABR calibration diagnostics.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    path = generate_cube(settings, steps=12)
    figure, table = calibration_diagnostics(path, settings)
    print(f"wrote {figure} and {len(table)} calibration diagnostic rows")


if __name__ == "__main__":
    main()
