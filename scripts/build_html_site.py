from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
from jinja2 import Template

from cxvega.config import load_settings
from cxvega.plotting import PALETTE, plotly_template, strategy_colours
from cxvega.reporting import (
    ensure_dirs,
    generate_all_figures,
    generate_cube,
    load_or_run_market_results,
)

STYLE = """
:root {
  --charcoal: #1a1a1a;
  --graphite: #3a3a3a;
  --silver: #b8b8b8;
  --off-white: #f5f5f5;
  --oxblood: #7a2e2e;
  --teal: #2f5d56;
}
body {
  background: var(--off-white);
  color: var(--charcoal);
  font-family: "EB Garamond", "Source Serif Pro", Georgia, serif;
  line-height: 1.45;
}
main {
  max-width: 72rem;
  margin-left: 8%;
  padding: 3rem 2rem 5rem 0;
}
nav, footer {
  font-family: "Source Sans Pro", Inter, Arial, sans-serif;
  color: var(--graphite);
  padding: 1rem 8%;
}
nav { border-bottom: 1px solid var(--silver); }
footer {
  border-top: 1px solid var(--silver);
  font-size: 0.9rem;
}
nav a, footer a {
  color: var(--oxblood);
  margin-right: 1.2rem;
  text-decoration: none;
}
h1, h2 {
  font-family: "Source Sans Pro", Inter, Arial, sans-serif;
  font-weight: 600;
  letter-spacing: 0;
  color: var(--oxblood);
}
h1 {
  font-size: 2.6rem;
  margin: 2rem 0 0.8rem;
}
h2 {
  margin-top: 2.8rem;
  border-top: 1px solid var(--silver);
  padding-top: 1rem;
}
.lede {
  max-width: 60ch;
  font-size: 1.25rem;
  line-height: 1.45;
  color: var(--graphite);
}
.note {
  float: right;
  width: 220px;
  margin: 0 0 24px 28px;
  font-size: 14px;
  line-height: 1.45;
  color: var(--graphite);
  border-left: 3px solid var(--silver);
  padding-left: 14px;
}
p {
  max-width: 60ch;
}
table {
  border-collapse: collapse;
  max-width: 64rem;
  font-family: "Source Sans Pro", Inter, Arial, sans-serif;
  font-size: 14px;
}
th, td {
  border-bottom: 1px solid #ddd;
  padding: 8px 10px;
  text-align: right;
}
th:first-child, td:first-child {
  text-align: left;
}
img {
  max-width: 100%;
  height: auto;
  margin: 1.2rem 0;
}
.plotly-graph-div {
  max-width: 64rem;
}
"""

TUFTE_CSS = """
html { font-size: 15px; }
body { margin: 0; counter-reset: sidenote-counter; }
article { padding: 5rem 0; }
section { padding-top: 1rem; padding-bottom: 1rem; }
.sidenote, .marginnote {
  float: right;
  clear: right;
  margin-right: -32%;
  width: 28%;
  margin-top: 0;
  margin-bottom: 0;
  font-size: 0.9rem;
  line-height: 1.35;
  vertical-align: baseline;
  position: relative;
}
"""

PAGE_TEMPLATE = Template(
    """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <link rel="stylesheet" href="assets/tufte.css">
  <style>{{ style }}</style>
</head>
<body>
<nav>
  <a href="index.html">Executive summary</a>
  <a href="model.html">Model</a>
  <a href="results.html">Results</a>
  <a href="appendix.html">Appendix</a>
</nav>
<main>{{ body }}</main>
<footer>
  <a href="../report/report.pdf">Read the full report (PDF)</a>
  <a href="https://github.com/imranhakmm/Cross-Cube-Vega-Hedger">GitHub repo</a>
  <span>Imran Hakim - v0.1 - 22 May 2026</span>
</footer>
</body>
</html>
"""
)


def _money_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for column in ["mean", "std", "var_1", "var_5", "cvar_1", "cvar_5"]:
        if column in display:
            display[column] = display[column].map(lambda x: f"{x / 1.0e6:,.2f}")
    if "sharpe" in display:
        display["sharpe"] = display["sharpe"].map(lambda x: f"{x:.2f}")
    return display.to_html(index=False, escape=False)


def _write_page(name: str, title: str, body: str) -> None:
    Path("docs/site").mkdir(parents=True, exist_ok=True)
    Path("docs/site", name).write_text(
        PAGE_TEMPLATE.render(title=title, style=STYLE, body=body),
        encoding="utf-8",
    )


def _write_site_css() -> None:
    assets = Path("docs/site/assets")
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "tufte.css").write_text(TUFTE_CSS.strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static HTML mini-site.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    ensure_dirs()
    _write_site_css()
    path = generate_cube(settings, steps=settings.market_maker.days)
    result = load_or_run_market_results(settings, path)
    generate_all_figures(settings, path)

    for figure in Path("outputs/figures").glob("*.png"):
        shutil.copyfile(figure, Path("docs/site/assets") / figure.name)

    hist = px.histogram(
        result.path_pnl,
        x="pnl",
        color="strategy",
        color_discrete_map=strategy_colours(),
        nbins=44,
        template=plotly_template(),
        title="Terminal P&L by strategy",
    )
    hist.update_layout(
        font_family="Source Sans Pro, Inter, Arial, sans-serif",
        paper_bgcolor=PALETTE.off_white,
        plot_bgcolor="white",
    )
    hist_html = pio.to_html(
        hist,
        include_plotlyjs="cdn",
        full_html=False,
        div_id="cxvega-terminal-pnl-histogram",
    )

    line = px.line(
        result.representative_path,
        x="day",
        y="cumulative_pnl",
        color="strategy",
        color_discrete_map=strategy_colours(),
        template=plotly_template(),
        title="Representative cumulative P&L path",
    )
    line.update_layout(
        font_family="Source Sans Pro, Inter, Arial, sans-serif",
        paper_bgcolor=PALETTE.off_white,
        plot_bgcolor="white",
    )
    line_html = pio.to_html(
        line,
        include_plotlyjs=False,
        full_html=False,
        div_id="cxvega-representative-path",
    )

    summary_html = _money_table(result.summary)
    _write_page(
        "index.html",
        "Cross-Cube Vega Hedging",
        f"""
        <h1>Cross-Cube Vega Hedging</h1>
        <p class="lede">Single-tenor vega is a mirage: the cube moves in factors, and naive hedges
        leave paid-for edge exposed to level, slope, curvature, and skew moves.</p>
        <p class="note">All data are simulated. The point is not to claim live-market calibration,
        but to make the residual risk mechanically visible and reproducible.</p>
        {hist_html}
        <h2>Results Summary</h2>
        {summary_html}
        """,
    )
    _write_page(
        "model.html",
        "Model",
        """
        <h1>The True Cube Model</h1>
        <p>The simulator evolves ATM log-volatility through smooth level, slope, and curvature
        loadings over the expiry-tenor grid. SABR rho is bounded in [-0.95, 0.0], log-nu mean
        reverts, and both are correlated with the level shock.</p>
        <img src="assets/atm_cube_snapshots.png" alt="ATM cube snapshots">
        <h2>Calibration Diagnostics</h2>
        <img src="assets/calibration_residual_heatmaps.png" alt="Calibration residuals">
        """,
    )
    _write_page(
        "results.html",
        "Results",
        f"""
        <h1>Market-Making Results</h1>
        {line_html}
        <img src="assets/pnl_attribution.png" alt="P&L attribution">
        <img src="assets/quoting_half_spread_heatmap.png" alt="Quoting heatmap">
        """,
    )
    _write_page(
        "appendix.html",
        "Appendix",
        """
        <h1>Appendix</h1>
        <p>The PCA recovery check verifies that the latent level, slope, and curvature modes can be
        recovered from simulated cube moves before the hedging strategies are compared.</p>
        <img src="assets/pca_recovery.png" alt="PCA recovery">
        <p>References: Andersen and Piterbarg (2010), Hagan et al. (2002), Rebonato (2002),
        Bergomi (2016), Avellaneda and Stoikov (2008), Gatheral (2006).</p>
        """,
    )
    print("wrote docs/site/index.html")


if __name__ == "__main__":
    main()
