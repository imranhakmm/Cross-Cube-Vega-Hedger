from __future__ import annotations

import argparse

from cxvega.config import load_settings
from cxvega.reporting import generate_all_figures, generate_cube


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all report and site figures.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    path = generate_cube(settings)
    figures = generate_all_figures(settings, path)
    print(f"wrote {len(figures)} figures under outputs/figures")


if __name__ == "__main__":
    main()
