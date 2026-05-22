from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
from jinja2 import Template

from cxvega.config import load_settings
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
}
body {
  margin: 0;
  background: var(--off-white);
  color: var(--charcoal);
  font-family: Georgia, "Times New Roman", serif;
}
main {
  max-width: 1040px;
  margin: 0 auto;
  padding: 48px 28px 72px;
  background: white;
}
nav, footer {
  font-family: Arial, sans-serif;
  color: var(--graphite);
  border-bottom: 1px solid var(--silver);
  padding: 16px 28px;
}
footer {
  border-top: 1px solid var(--silver);
  border-bottom: 0;
}
nav a, footer a {
  color: var(--oxblood);
  margin-right: 18px;
  text-decoration: none;
}
h1, h2 {
  font-weight: 500;
  letter-spacing: 0;
}
h1 {
  font-size: 42px;
  margin: 32px 0 14px;
}
h2 {
  margin-top: 42px;
  border-top: 1px solid var(--silver);
  padding-top: 20px;
}
.lede {
  font-size: 20px;
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
table {
  border-collapse: collapse;
  width: 100%;
  font-family: Arial, sans-serif;
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static HTML mini-site.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    ensure_dirs()
    path = generate_cube(settings, steps=settings.market_maker.days)
    result = load_or_run_market_results(settings, path)
    generate_all_figures(settings, path)

    for figure in Path("outputs/figures").glob("*.png"):
        shutil.copyfile(figure, Path("docs/site/assets") / figure.name)

    hist = px.histogram(
        result.path_pnl,
        x="pnl",
        color="strategy",
        nbins=44,
        template="simple_white",
        title="Terminal P&L by strategy",
    )
    hist.update_layout(font_family="Arial", paper_bgcolor="#f5f5f5", plot_bgcolor="white")
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
        template="simple_white",
        title="Representative cumulative P&L path",
    )
    line.update_layout(font_family="Arial", paper_bgcolor="#f5f5f5", plot_bgcolor="white")
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
