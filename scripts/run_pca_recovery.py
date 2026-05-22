from __future__ import annotations

import argparse

from cxvega.config import load_settings
from cxvega.reporting import generate_cube, pca_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCA factor-recovery diagnostics.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    path = generate_cube(settings, steps=504)
    figure = pca_figure(path)
    print(f"wrote {figure}")


if __name__ == "__main__":
    main()
