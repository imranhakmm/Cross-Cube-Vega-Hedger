"""Central visual style for report, site, and generated figures."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import plotly.graph_objects as go


@dataclass(frozen=True)
class Palette:
    """Project colour palette."""

    charcoal: str = "#1a1a1a"
    graphite: str = "#3a3a3a"
    silver: str = "#b8b8b8"
    off_white: str = "#f5f5f5"
    oxblood: str = "#7a2e2e"
    steel: str = "#6f7f89"
    pine: str = "#2f5d56"


PALETTE = Palette()


def apply_style() -> None:
    """Apply the silver sell-side research-note style to Matplotlib."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE.silver,
            "axes.labelcolor": PALETTE.graphite,
            "axes.titlecolor": PALETTE.charcoal,
            "axes.grid": False,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.color": PALETTE.graphite,
            "ytick.color": PALETTE.graphite,
            "legend.frameon": False,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "svg.hashsalt": "cross-cube-vega",
        }
    )


def strategy_colours() -> dict[str, str]:
    """Return consistent colours for the four hedging strategies."""

    return {
        "Delta-only": PALETTE.graphite,
        "Single-tenor vega-flat": PALETTE.silver,
        "Bucketed vega-flat": PALETTE.pine,
        "Factor-neutral": PALETTE.oxblood,
    }


def plotly_template() -> go.layout.Template:
    """Return the Plotly template matching the PDF figure palette."""

    return go.layout.Template(
        layout={
            "font": {
                "family": "Source Sans Pro, Inter, Arial, sans-serif",
                "color": PALETTE.charcoal,
            },
            "paper_bgcolor": PALETTE.off_white,
            "plot_bgcolor": "white",
            "colorway": [PALETTE.graphite, PALETTE.silver, PALETTE.pine, PALETTE.oxblood],
            "xaxis": {
                "showgrid": False,
                "linecolor": PALETTE.silver,
                "tickcolor": PALETTE.graphite,
                "zerolinecolor": PALETTE.silver,
            },
            "yaxis": {
                "gridcolor": "#dddddd",
                "linecolor": PALETTE.silver,
                "tickcolor": PALETTE.graphite,
                "zerolinecolor": PALETTE.silver,
            },
            "legend": {"orientation": "h", "y": -0.18},
            "title": {"font": {"family": "Source Sans Pro, Inter, Arial, sans-serif"}},
        }
    )
