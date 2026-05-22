from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import subprocess
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from matplotlib.backends.backend_pdf import PdfPages

from cxvega.config import Settings, load_settings
from cxvega.plotting import PALETTE, apply_style
from cxvega.reporting import (
    ensure_dirs,
    generate_all_figures,
    generate_cube,
    load_or_run_market_results,
)

REPORT_DIR = Path("docs/report")
FIGURE_DIR = REPORT_DIR / "figures"
OUTPUT_FIGURE_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")
REPORT_DATE = date(2026, 5, 22)
TITLE = "Cross-Cube Vega Hedging: A Swaption Market-Making Lab"
TEMPLATE_DIR = REPORT_DIR / "templates"
RENDERED_HTML = REPORT_DIR / "report.rendered.html"


@dataclass(frozen=True)
class TableSpec:
    """Lightweight table description for the Matplotlib report."""

    headers: Sequence[str]
    rows: Sequence[Sequence[str]]
    widths: Sequence[float]


def _write_latex_sources(settings: Settings) -> None:
    report_tex = r"""
\documentclass[11pt]{article}
\usepackage{style/cxvega}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{float}
\usepackage{graphicx}
\usepackage{multicol}
\usepackage[authoryear,backend=biber]{biblatex}
\addbibresource{references.bib}
\title{Cross-Cube Vega Hedging: A Swaption Market-Making Lab}
\author{Imran Hakim}
\date{22 May 2026}

\begin{document}
\cxcover
\section*{Abstract}
Node vega is a useful operating report, but it is not the risk basis of a
swaption cube.  This lab simulates a SABR cube whose ATM level, slope, curvature
and skew-wing dynamics move together, then runs a market-maker through the cube
with four hedging policies.  The calibration exercise deliberately adds
observation noise, so per-slice fits no longer recover the data-generating model
mechanically.  The constrained fit pays a small local-residual cost to remove
material cube-level static-arbitrage violations.  The market-making experiment
then shows the trading consequence: factor-neutral hedging compresses terminal
P\&L dispersion and improves the left tail.  The limitation is equally explicit:
all market data are simulated, beta is fixed, and the quoting overlay is a
factor-space heuristic rather than a full HJB solution.

\section*{Executive Summary}
Single-tenor vega is a mirage.  A book that is flat by node can still be long
the level factor, short the expiry slope, and exposed to skew steepening.  The
headline figure uses a same-axis P\&L overlay: delta-only is a broad distribution,
while factor-neutral hedging is a much tighter spike.  The result is not free;
the factor-neutral rule pays more transaction cost and depends on liquid proxy
instruments.  It is still the hedge that best matches the cube's realised risk
basis.

\section{Introduction and Motivation}
Market-makers manage swaption risk in cube coordinates because that is how
quotes, controls, and end-of-day marks arrive.  The cube, however, moves in
factors.  Bartlett's SABR delta correction is the same warning in miniature:
the obvious hedge is wrong once model parameters co-move with the underlying.

\section{Architecture}
\input{style/architecture.tikz}
The Python pipeline is deterministic: simulator, calibration, PCA analytics,
market-maker loop, P\&L attribution, static site, and this report.

\section{Model: The True Cube}
\[
\log \sigma_{\mathrm{ATM}}(T,\tau,t)=\mu(T,\tau)+
  \sum_{i=1}^3 L_i(T,\tau)X_i(t).
\]
The factors follow correlated OU dynamics
\[
dX_i=-\kappa_i X_i\,dt+\eta_i\,dW_i,\qquad
\mathrm{corr}(dW)=R.
\]
The loading shapes are a level mode, a centred log-expiry slope, and a hump
around intermediate expiries.  Rho is bounded in $[-0.95,0]$ and log-nu
mean-reverts, with both skew and wing shocks correlated to the level factor.

\section{Calibration}
The per-slice fit fixes beta, enforces the ATM point exactly, and fits rho and
nu.  The joint fit adds butterfly, calendar, and smoothness penalties.  The
arbitrage penalties dominate smoothness; this is intentionally a risk-control
fit, not a local least-squares beauty contest.

\section{Factor Recovery via PCA}
PCA is run on simulated ATM log-vol changes, then the recovered subspace is
rotated into the simulator's true factor labels for diagnostics.  The cosine
similarities in the generated table quantify whether the analytics pipeline has
found the intended level, slope, and curvature modes.

\section{Hedging Strategies}
The four policies are delta-only, single-tenor vega-flat, bucketed vega-flat, and
factor-neutral.  The last solves a small constrained hedge over liquid cube
points to reduce exposures to level, slope, curvature, and skew.

\section{Market-Maker Simulation and Quoting}
Client trades arrive from a Poisson process and cross the market-maker's quote.
The reservation vol uses an Avellaneda-Stoikov-style inventory charge in factor
space.  This is a heuristic adaptation, not an end-to-end HJB derivation.

\section{Results and P\&L Attribution}
The report tables reconcile edge, theta, delta, first-order vega, cross-vega
slippage, and hedge transaction cost to total realised P\&L.  The result to care
about is the distributional one: hedging the factors tightens the tail.

\section{Assumptions and Limitations}
The lab uses simulated data only, single-curve discounting, fixed beta, no
jumps, daily hedging, perfectly observed mids, and no strategic client response.

\section{Extensions}
Natural extensions are real-data overlays, rough-vol dynamics, multi-curve
discounting, joint cap-swaption calibration, Bermudan exercise, IDB-aware
quoting, and a learning layer over the factor state.

\printbibliography
\end{document}
"""
    references = r"""
@book{andersenpiterbarg2010,
  author={Andersen, Leif B. G. and Piterbarg, Vladimir V.},
  title={Interest Rate Modeling},
  publisher={Atlantic Financial Press},
  year={2010}
}
@article{hagan2002,
  author={Hagan, Patrick S. and Kumar, Deep and Lesniewski, Andrew S. and Woodward, Diana E.},
  title={Managing Smile Risk},
  journal={Wilmott Magazine},
  year={2002}
}
@article{bartlett2006,
  author={Bartlett, Bruce},
  title={Hedging Under SABR Model},
  journal={Wilmott Magazine},
  year={2006}
}
@book{rebonato2002,
  author={Rebonato, Riccardo},
  title={Modern Pricing of Interest-Rate Derivatives},
  publisher={Princeton University Press},
  year={2002}
}
@book{bergomi2016,
  author={Bergomi, Lorenzo},
  title={Stochastic Volatility Modeling},
  publisher={Chapman and Hall/CRC},
  year={2016}
}
@article{avellanedastoikov2008,
  author={Avellaneda, Marco and Stoikov, Sasha},
  title={High-frequency trading in a limit order book},
  journal={Quantitative Finance},
  volume={8},
  number={3},
  pages={217--224},
  year={2008}
}
@book{gatheral2006,
  author={Gatheral, Jim},
  title={The Volatility Surface: A Practitioner's Guide},
  publisher={Wiley},
  year={2006}
}
@book{brigomercurio2006,
  author={Brigo, Damiano and Mercurio, Fabio},
  title={Interest Rate Models: Theory and Practice},
  publisher={Springer},
  year={2006}
}
@book{glasserman2004,
  author={Glasserman, Paul},
  title={Monte Carlo Methods in Financial Engineering},
  publisher={Springer},
  year={2004}
}
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.tex").write_text(report_tex.strip() + "\n", encoding="utf-8")
    (REPORT_DIR / "references.bib").write_text(references.strip() + "\n", encoding="utf-8")
    _ = settings


def _try_xelatex() -> bool:
    if shutil.which("xelatex") is None:
        return False
    try:
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "report.tex"],
            cwd=REPORT_DIR,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return (REPORT_DIR / "report.pdf").exists()
    except subprocess.CalledProcessError:
        return False


def _fmt_mm(value: float) -> str:
    return f"{value / 1.0e6:,.1f}"


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    row_html = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        row_html.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(row_html)}</tbody></table>"


def _summary_table(summary: pd.DataFrame, strategies: Sequence[str]) -> str:
    frame = summary.set_index("strategy").loc[list(strategies)]
    rows = [
        [
            str(strategy),
            _fmt_mm(float(row["mean"])),
            _fmt_mm(float(row["std"])),
            _fmt_mm(float(row["var_5"])),
            f"{float(row['sharpe']):.2f}",
        ]
        for strategy, row in frame.iterrows()
    ]
    return _html_table(["Strategy", "Mean", "Std", "5% VaR", "Sharpe"], rows)


def _arb_table(arb: pd.DataFrame) -> str:
    frame = arb.copy()
    for column in ["butterfly", "calendar"]:
        if column not in frame:
            frame[column] = 0
    rows = [
        [
            str(row["fit"]),
            str(int(row["violations"])),
            str(int(row["butterfly"])),
            str(int(row["calendar"])),
        ]
        for _, row in frame.iterrows()
    ]
    return _html_table(["Fit", "Total", "Butterfly", "Calendar"], rows)


def _pca_table(pca: pd.DataFrame) -> str:
    rows = [
        [
            str(row["factor"]).title(),
            f"{float(row['loading_similarity']):.3f}",
            f"{100.0 * float(row['explained_variance']):.1f}%",
        ]
        for _, row in pca.iterrows()
    ]
    return _html_table(["Factor", "Cosine", "Expl. var"], rows)


def _tail_table(summary: pd.DataFrame) -> str:
    rows = [
        [
            str(row["strategy"]),
            _fmt_mm(float(row["var_1"])),
            _fmt_mm(float(row["var_5"])),
            _fmt_mm(float(row["cvar_5"])),
            _fmt_mm(float(row["std"])),
        ]
        for _, row in summary.iterrows()
    ]
    return _html_table(["Strategy", "1% VaR", "5% VaR", "5% CVaR", "Std"], rows)


def _figure(filename: str, caption: str, css_class: str = "") -> str:
    class_attr = f' class="{css_class}"' if css_class else ""
    return (
        f"<figure{class_attr}>"
        f'<img src="figures/{filename}" alt="{caption}">'
        f"<figcaption>{caption}</figcaption>"
        "</figure>"
    )


def _equation(name: str) -> str:
    return f'<img class="equation" src="figures/eq_{name}.svg" alt="{name} equation">'


def _render_equations() -> None:
    equations = {
        "atm": r"\log \sigma_{\mathrm{ATM}}(T,\tau,t)=\mu(T,\tau)+\sum_i L_i(T,\tau)X_i(t)",
        "ou": r"dX_i=-\kappa_i X_i\,dt+\eta_i\,dW_i,\quad \mathrm{corr}(dW)=R",
        "calibration": r"J=\Vert\sigma_m-\sigma_q\Vert^2+\lambda_b B+\lambda_c C+\lambda_s S",
        "hedge": r"\min_h \Vert A(q+Bh)\Vert^2+\lambda\Vert h\Vert^2+c^\top |h|",
        "quoting": r"r_i=\mathrm{mid}_i-\gamma q^\top\Sigma l_i(T_{\max}-t)",
    }
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"mathtext.fontset": "stix", "svg.fonttype": "path"}):
        for name, expression in equations.items():
            fig = plt.figure(figsize=(7.0, 0.56))
            fig.patch.set_alpha(0.0)
            fig.text(0.5, 0.5, f"${expression}$", ha="center", va="center", fontsize=17)
            fig.savefig(
                FIGURE_DIR / f"eq_{name}.svg",
                bbox_inches="tight",
                pad_inches=0.02,
                transparent=True,
                metadata={"Date": None},
            )
            plt.close(fig)


def _build_pages(summary: pd.DataFrame, arb: pd.DataFrame, pca: pd.DataFrame) -> list[dict[str, str | bool]]:
    return [
        {
            "title": "Abstract and Executive Summary",
            "architecture": False,
            "body": f"""
              <h2>Abstract</h2>
              <p>A swaption market-maker sees vega by cube point, but the realised cube does not move
              one point at a time. It moves through correlated factors: ATM level, expiry slope,
              curvature, and skew-wing dynamics. This lab simulates that cube, calibrates noisy SABR
              smiles, runs a client-flow market-maker, and compares four hedging policies. The
              factor-neutral rule compresses terminal P&amp;L dispersion and improves the left tail.
              The limitation is explicit: no live market data, fixed beta, daily hedging, and a
              quoting overlay that is a factor-space heuristic rather than a full HJB solution.</p>
              <h2>Executive Summary</h2>
              <p class="lede">Single-tenor vega is a mirage. It is a convenient control number,
              but it is not the basis in which the cube actually moves.</p>
              <p>The same-axis terminal P&amp;L overlay below is the central exhibit: delta-only hedging
              leaves a fat distribution, while factor-neutral hedging concentrates the outcome around
              a tighter risk budget. The result is not free. It trades more, pays more hedge cost,
              and depends on liquid proxy instruments. Those costs buy a cleaner residual.</p>
              {_summary_table(summary, ["Delta-only", "Factor-neutral"])}
              {_figure("headline_pnl_dist.png", "Figure 1. Same-axis terminal P&L overlay; dashed lines mark strategy means.", "figure-small")}
            """,
        },
        {
            "title": "Introduction and Motivation",
            "architecture": False,
            "body": """
              <div class="columns">
                <div>
                  <h2>Why Node Vega Exists</h2>
                  <p>Node vega is the reporting standard because it is operationally legible. Traders
                  quote points, risk systems aggregate points, and end-of-day controls ask whether a
                  trader is long or short a recognisable part of the cube. For a desk, that convention
                  is useful. For a hedge, it is incomplete.</p>
                  <h2>The Risk Basis</h2>
                  <p>A block in 2y into 10y exposes the book to more than the 2y x 10y mark. It has
                  projection onto level, onto the front-back expiry slope, onto an intermediate-expiry
                  hump, and onto skew and wing dynamics. A local vega hedge can be flat at the booked
                  point and still be long the level factor.</p>
                </div>
                <div>
                  <h2>The Bartlett Pattern</h2>
                  <p>Bartlett's SABR delta correction is a useful mental model. The Black delta is the
                  obvious hedge only if volatility parameters stand still. Once alpha co-moves with the
                  forward, the hedge needs the parameter response too. Cross-cube vega has the same
                  shape: the obvious node hedge ignores nearby smiles moving together.</p>
                  <h2>Trader's-Day Framing</h2>
                  <p>The market-maker takes down flow, earns edge, and goes home with inventory. The
                  next morning the cube has moved. The question is not whether the single node was
                  hedged, but whether the hedge was short the same factor the book was long.</p>
                </div>
              </div>
            """,
        },
        {
            "title": "Architecture",
            "architecture": True,
            "body": """
              <h2>Pipeline Walk-Through</h2>
              <p>The simulator produces a full SABR cube path: forwards, annuities, strikes, ATM vols,
              rho, nu, alpha, and quoted smile vols. Calibration consumes a noisy snapshot and writes
              residual heatmaps plus static-arbitrage counts. PCA analytics recover the latent risk
              basis. The market-maker loop samples client flow, applies the quoting overlay, hedges at
              end-of-day, and emits path-level P&amp;L. Attribution then reconciles edge, theta, delta,
              first-order vega, cross-vega, and hedge cost.</p>
              <h2>Reproducibility Contract</h2>
              <p>The master seed sits in YAML. The observation-noise stream is separate from the
              simulator streams, so calibrating a noisy market snapshot does not perturb the true cube
              path. The Makefile rebuilds simulations, figures, the static site, and this PDF.</p>
            """,
        },
        {
            "title": "Model: True Cube I",
            "architecture": False,
            "body": f"""
              <div class="columns">
                <div>
                  <h2>ATM Factor Surface</h2>
                  <p>The cube is indexed by option expiry T in 1m, 3m, 6m, 1y, 2y, 5y, and 10y, and
                  underlying swap tenor tau in 1y, 2y, 5y, and 10y. The ATM Black-vol surface is driven
                  by three smooth loadings over this grid.</p>
                  {_equation("atm")}
                  <h2>Loading Shapes</h2>
                  <p>The level loading is nearly constant across the cube with a small tenor tilt. The
                  slope loading is the centred log-expiry coordinate. The curvature loading is a Gaussian
                  hump around intermediate expiries, centred near two years, then normalised.</p>
                </div>
                <div>
                  <h2>OU Dynamics</h2>
                  {_equation("ou")}
                  <p>The OU factors mean-revert daily. Their instantaneous covariance is configurable
                  in YAML, and the quoting overlay uses the full covariance matrix including off-diagonal
                  terms. That matters because level inventory and skew inventory should interact.</p>
                  <h2>Stylised Scale</h2>
                  <p>The front of the normal-vol cube is set around the 60-90bp region and the back is
                  anchored near 80bp, consistent with broad interest-rate volatility modelling examples.
                  Absolute levels are laboratory choices, not market claims.</p>
                </div>
              </div>
            """,
        },
        {
            "title": "Model: True Cube II",
            "architecture": False,
            "body": """
              <div class="columns">
                <div>
                  <h2>Skew and Wing Dynamics</h2>
                  <p>Each expiry-tenor slice carries SABR parameters alpha, beta, rho, and nu. Beta is
                  fixed at the configured value. Rho is generated through a transformed OU process, so
                  it stays inside [-0.95, 0.0]. Log-nu is another OU process. Both shocks are correlated
                  with the level factor, reflecting the stylised observation that skew and volatility
                  level do not move independently.</p>
                  <h2>ATM Inversion</h2>
                  <p>The simulator specifies ATM vol first, then recovers alpha from Hagan's ATM SABR
                  formula. This keeps the factor model attached to the observable surface rather than
                  to an abstract alpha process.</p>
                </div>
                <div>
                  <h2>SABR and Vol Conversion</h2>
                  <p>The pricer implements Hagan's 2002 lognormal SABR formula with a shifted extension
                  for negative-rate laboratory cases. Black-76 and Bachelier pricing are both present.
                  Normal/lognormal conversion is done by Brent root-finding on price equality rather
                  than by a closed-form approximation.</p>
                  <h2>Parameter Honesty</h2>
                  <p>The configuration file is not a calibration file. It records a plausible set of
                  stylised facts: mean-reverting factors, correlated level/skew shocks, bounded rho,
                  and liquid hedge points. The report is a controlled experiment, not a claim about
                  today's swaption market.</p>
                </div>
              </div>
            """,
        },
        {
            "title": "Model: Cube Snapshots",
            "architecture": False,
            "body": f"""
              {_figure("atm_cube_snapshots.png", "Figure 2. Three ATM cube snapshots from the simulated path.")}
              <h2>Reading the Heatmaps</h2>
              <p>The snapshots are deliberately boring in the right way: the whole cube changes, and
              nearby expiries and tenors change together. A single isolated node shock would make the
              hedging problem artificial. The thesis needs a cube that moves in common modes, because
              that is where local vega reports become a lossy compression.</p>
            """,
        },
        {
            "title": "Calibration I",
            "architecture": False,
            "body": f"""
              <div class="columns">
                <div>
                  <h2>Noisy Market Snapshot</h2>
                  <p>The calibration target is no longer the exact data-generating SABR surface. Each
                  strike vol receives independent multiplicative lognormal observation noise:
                  sigma_market = sigma_true exp(eps), eps ~ N(0, sigma_obs^2), with sigma_obs = 0.005
                  by default. The random stream is separate from the simulator stream.</p>
                  <h2>Per-Slice Fit</h2>
                  <p>Per-slice calibration fixes beta, matches ATM exactly through alpha inversion, and
                  fits rho and nu by least squares. It gives low local residuals, but it has no reason
                  to respect calendar consistency across neighbouring expiries.</p>
                </div>
                <div>
                  <h2>Cube-Constrained Fit</h2>
                  <p>The joint objective keeps the same ATM discipline and adds penalties for butterfly
                  and calendar violations, plus a smaller smoothness penalty on rho and log-nu over the
                  cube. The no-arbitrage penalties dominate smoothness by design.</p>
                  {_equation("calibration")}
                  <p>This is not a local least-squares beauty contest; it is a risk-control fit. The
                  diagnostic question is whether a small residual cost buys fewer static-arbitrage
                  breaks.</p>
                </div>
              </div>
            """,
        },
        {
            "title": "Calibration II",
            "architecture": False,
            "body": f"""
              {_figure("calibration_residual_heatmaps.png", "Figure 3. Residual RMS in vol bp. Joint calibration pays local residual for consistency.", "figure-small")}
              <h2>Static-Arbitrage Counts</h2>
              {_arb_table(arb)}
              <p>The noisy per-slice fit now has genuine residuals instead of a meaningless 1e-12
              colourbar. The constrained fit is slightly worse locally, but it removes the material
              calendar breaks under the same checker. This is the calibration analogue of the hedging
              thesis: a local optimum can be the wrong portfolio object.</p>
            """,
        },
        {
            "title": "Factor Recovery via PCA I",
            "architecture": False,
            "body": f"""
              <div class="columns">
                <div>
                  <h2>Method</h2>
                  <p>PCA is run on daily changes in the simulated ATM log-vol surface. The first three
                  components explain the designed level, slope, and curvature subspace. Because PCA
                  basis vectors are sign-ambiguous, the diagnostic aligns them to the known simulator
                  loadings and flips signs consistently.</p>
                  <h2>Cosine Similarity</h2>
                  {_pca_table(pca)}
                </div>
                <div>
                  <h2>What This Validates</h2>
                  <p>The table does not claim that PCA can name economic factors in real data without
                  judgement. It says something narrower and important: in this controlled cube, the
                  analytics recover the latent risk subspace used by the hedger.</p>
                  <h2>What It Does Not Validate</h2>
                  <p>It does not validate the parameter levels, the absence of jumps, or the stability
                  of factor loadings through regimes. A live implementation would rerun this diagnostic
                  over rolling windows.</p>
                </div>
              </div>
            """,
        },
        {
            "title": "Factor Recovery via PCA II",
            "architecture": False,
            "body": f"""
              {_figure("pca_recovery.png", "Figure 4. Dashed true loadings and solid recovered loadings are normalised consistently.")}
              <h2>Reading the Figure</h2>
              <p>The recovered curves are plotted against the same expiry axis and colour as the true
              factor labels. Both are unit-normalised before plotting, and the sign ambiguity is
              resolved before the cosine table is written.</p>
            """,
        },
        {
            "title": "Hedging Strategies I",
            "architecture": False,
            "body": """
              <div class="columns">
                <div>
                  <h2>Delta-Only</h2>
                  <p>The control hedge removes underlying swap delta and leaves vega to run. It is
                  useful as a lower bound because it captures the cost of pretending the cube is not the
                  main risk. Delta P&amp;L should be small in expectation, but vega and cross-vega dominate.</p>
                  <h2>Single-Tenor Vega-Flat</h2>
                  <p>Each new trade's vega is snapped to the nearest cube node and offset by trading
                  that node. This looks clean in a blotter and can be deeply wrong in factor space if
                  the offset node loads on the wrong combination of level, slope, and curvature.</p>
                </div>
                <div>
                  <h2>Bucketed Vega-Flat</h2>
                  <p>Bucketed hedging nets vega within expiry buckets: &lt;=1y, 1y-3y, 3y-7y, and &gt;7y.
                  This practical compromise cannot fully neutralise factor exposure, but it stops the
                  most obvious front/back leakage and uses fewer hedge instruments.</p>
                  <h2>Portfolio Greek Matrix</h2>
                  <p>Let V be node vega and L be the loading matrix. Raw vega reports V. Factor risk is
                  closer to L'V, augmented by skew exposure. The hedging question is not only whether
                  sum(V)=0, but whether the projected vector is small.</p>
                </div>
              </div>
            """,
        },
        {
            "title": "Hedging Strategies II",
            "architecture": False,
            "body": f"""
              <div class="columns">
                <div>
                  <h2>Factor-Neutral Hedge</h2>
                  <p>The factor-neutral rule selects liquid cube points and solves a small constrained
                  hedge to bring level, slope, curvature, and skew exposures near zero. It minimises
                  residual factor exposure subject to trading-cost and instrument-size penalties.</p>
                  {_equation("hedge")}
                  <p>In trader language: use the liquid points that actually load on the factors you
                  are trying to offset, and avoid over-trading an ill-conditioned local hedge.</p>
                </div>
                <div>
                  <h2>Instrument Selection</h2>
                  <p>The hedge set is smaller than the risk grid. That is intentional. A market-maker
                  normally has a handful of executable liquid points and must project the book onto
                  that basis. The factor-neutral hedge is a practical proxy hedge, not a fantasy where
                  every node trades frictionlessly.</p>
                  <h2>Cost Accounting</h2>
                  <p>Every hedge trade pays a configurable bid-ask cost in vol bp. The attribution table
                  keeps that cost separate from cross-vega residuals.</p>
                </div>
              </div>
            """,
        },
        {
            "title": "Market-Maker Simulation",
            "architecture": False,
            "body": """
              <div class="columns">
                <div>
                  <h2>Client Flow</h2>
                  <p>Client trades arrive from a Poisson process. Cube point, strike offset, side, and
                  notional are sampled from simple distributions. The client crosses the quote with
                  probability near one. That simplification keeps the experiment focused on inventory
                  and hedging rather than on a separate client-response model.</p>
                  <h2>Book State</h2>
                  <p>After each trade, inventory Greeks and factor exposures update. Hedging fires at
                  end-of-day. The simulation records the book before and after hedging so P&amp;L can be
                  attributed to edge, theta, delta, first-order vega, cross-vega, and hedge cost.</p>
                </div>
                <div>
                  <h2>Monte Carlo Scale</h2>
                  <p>The default run uses 1,000 market-making paths over 252 trading days. Tests use
                  small deterministic seeds and avoid Monte Carlo. Generated simulations are cached
                  under outputs/simulations, while make clean removes them to prove reproducibility.</p>
                  <h2>Simplifications</h2>
                  <p>There is no adverse-selection model, no funding spread, no exchange margin, and
                  no jump risk. Those choices make the experiment narrower, but they also make the
                  hedge comparison easier to audit.</p>
                </div>
              </div>
            """,
        },
        {
            "title": "Quoting Overlay",
            "architecture": False,
            "body": f"""
              <h2>Reservation Vol in Factor Space</h2>
              <p>The quote mid is adjusted by an Avellaneda-Stoikov-style inventory charge, but the
              inventory vector is factor exposure rather than a single scalar position.</p>
              {_equation("quoting")}
              <div class="columns">
                <div>{_figure("quoting_half_spread_heatmap.png", "Figure 5. Half-spread responds to level and skew inventory together.", "figure-tiny")}</div>
                <div>
                  <p>The heatmap is intentionally no longer flat. The displayed grid spans a wide enough
                  inventory range for the risk-aversion term to matter, and the covariance matrix is
                  used with off-diagonal terms intact. The gradient is therefore not purely axis-aligned.
                  This remains a heuristic overlay, not a full dynamic-programming derivation.</p>
                </div>
              </div>
            """,
        },
        {
            "title": "Results I: Representative Path",
            "architecture": False,
            "body": f"""
              {_figure("representative_mm_path.png", "Figure 6. One path shows how the same client flow diverges under different hedges.")}
              <h2>Path-Level Read</h2>
              <p>A single path should not be over-interpreted, but it is useful for sanity. The
              strategies see the same simulated flow and cube path. Divergence comes from the hedge
              state, hedge costs, and residual factor exposures rather than from different market draws.</p>
            """,
        },
        {
            "title": "Results II: Distributions and Tail Risk",
            "architecture": False,
            "body": f"""
              {_figure("pnl_dist_small_multiples.png", "Figure 7. Small multiples keep the same x-axis while showing each strategy separately.", "figure-small")}
              <h2>Tail-Risk Table ($mm)</h2>
              {_tail_table(summary)}
              <p>The same-axis overlay in the executive summary is the headline picture. These small
              multiples make the same point strategy by strategy: delta-only carries the broadest
              distribution, bucketed hedging improves the shape, and factor-neutral hedging is the
              tightest in this experiment.</p>
            """,
        },
        {
            "title": "Results III: P&L Attribution",
            "architecture": False,
            "body": f"""
              {_figure("pnl_attribution.png", "Figure 8. Attribution reconciles spread edge, Greeks, residual cross-vega, and cost.")}
              <h2>Trader-Tone Read</h2>
              <p>The factor-neutral hedge does not create edge; it protects the edge already earned.
              Its cost is visible and should be challenged. The reason to like it is that the
              cross-vega residual is less dominant, so the P&amp;L is less dependent on being lucky about
              which cube factor moved after the trade.</p>
            """,
        },
        {
            "title": "Results IV: What the Experiment Says",
            "architecture": False,
            "body": """
              <div class="columns">
                <div>
                  <h2>Headline</h2>
                  <p>The result supports the core thesis in the controlled setting: a hedge constructed
                  in the cube's factor basis produces a tighter realised P&amp;L distribution than a hedge
                  constructed only in local node vega. The improvement is clearest in the left tail and
                  in the residual attribution.</p>
                  <h2>Calibration Link</h2>
                  <p>The calibration section tells the same story in model space. Per-slice fitting wins
                  the local residual contest. The cube-constrained fit is the better surface object
                  because it removes static-arbitrage breaks under a symmetric checker.</p>
                </div>
                <div>
                  <h2>Caveat</h2>
                  <p>This is not a live trading backtest. It is a market-making lab. The aim is to make
                  the mechanics transparent enough that a reviewer can disagree with the parameters,
                  rerun the project, and see how the result changes.</p>
                  <h2>Desk Use</h2>
                  <p>The immediate desk lesson is not to throw away node vega. Keep it for operations,
                  but report the factor projection beside it. A position that is node-flat and
                  factor-long should not be called flat.</p>
                </div>
              </div>
            """,
        },
        {
            "title": "Assumptions and Limitations",
            "architecture": False,
            "body": """
              <p><strong>Simulated data only.</strong> No conclusion depends on a hidden vendor feed,
              but absolute P&amp;L levels are not live-market forecasts.</p>
              <p><strong>Single-curve discounting.</strong> The lab ignores multi-curve basis and
              collateral details that would matter in production pricing.</p>
              <p><strong>Fixed beta.</strong> SABR beta is configurable but not calibrated; this keeps
              the exercise focused on cross-cube vega.</p>
              <p><strong>No jumps.</strong> Daily OU dynamics miss event risk, central-bank gap moves,
              and regime shifts.</p>
              <p><strong>Daily hedging.</strong> Intraday inventory management is compressed into an
              end-of-day hedge.</p>
              <p><strong>Perfectly observed mids.</strong> There is no mark uncertainty or broker-quality
              hierarchy.</p>
              <p><strong>Heuristic quoting.</strong> The factor-space reservation vol borrows the
              Avellaneda-Stoikov shape but is not a full HJB solution.</p>
            """,
        },
        {
            "title": "Extensions",
            "architecture": False,
            "body": """
              <h2>Real-data overlay</h2><p>Use exchange settles or broker indicative marks to initialise
              the cube and compare factor stability.</p>
              <h2>Rough-vol dynamics</h2><p>Replace OU factors with rougher latent drivers and test
              whether hedging cadence becomes more important.</p>
              <h2>Multi-curve pricing</h2><p>Add OIS discounting and forward curves so the swaption lab
              better resembles production infrastructure.</p>
              <h2>Joint cap-swaption calibration</h2><p>Bring caplets into the smile calibration and ask
              whether short-rate consistency changes the hedge basis.</p>
              <h2>Bermudan extension</h2><p>Use least-squares Monte Carlo to push the factor hedge into
              callable-exotics inventory.</p>
              <h2>Strategic quoting</h2><p>Let IDB visibility and client response feed back into the
              reservation-vol rule.</p>
              <h2>Learning overlay</h2><p>Use reinforcement learning as a policy layer on top of the
              transparent factor state, not as a black-box replacement.</p>
            """,
        },
        {
            "title": "References I",
            "architecture": False,
            "body": """
              <div class="references">
              <p>Andersen, Leif B. G. and Vladimir V. Piterbarg (2010). <em>Interest Rate Modeling</em>.
              Atlantic Financial Press.</p>
              <p>Hagan, Patrick S., Deep Kumar, Andrew S. Lesniewski, and Diana E. Woodward (2002).
              "Managing Smile Risk." <em>Wilmott Magazine</em>.</p>
              <p>Bartlett, Bruce (2006). "Hedging Under SABR Model." <em>Wilmott Magazine</em>.</p>
              <p>Rebonato, Riccardo (2002). <em>Modern Pricing of Interest-Rate Derivatives</em>.
              Princeton University Press.</p>
              <p>Bergomi, Lorenzo (2016). <em>Stochastic Volatility Modeling</em>. Chapman and Hall/CRC.</p>
              </div>
            """,
        },
        {
            "title": "References II",
            "architecture": False,
            "body": """
              <div class="references">
              <p>Avellaneda, Marco and Sasha Stoikov (2008). "High-frequency trading in a limit order
              book." <em>Quantitative Finance</em> 8(3), 217-224.</p>
              <p>Gatheral, Jim (2006). <em>The Volatility Surface: A Practitioner's Guide</em>. Wiley.</p>
              <p>Brigo, Damiano and Fabio Mercurio (2006). <em>Interest Rate Models: Theory and
              Practice</em>. Springer.</p>
              <p>Glasserman, Paul (2004). <em>Monte Carlo Methods in Financial Engineering</em>. Springer.</p>
              <h2>Build Note</h2>
              <p>The preferred renderer is WeasyPrint. In this sandbox WeasyPrint's Python package
              installs, but native Pango/GObject libraries are unavailable, so the active renderer is
              headless Chromium via Playwright. Matplotlib remains available only behind the explicit
              emergency flag.</p>
              </div>
            """,
        },
    ]


def _render_report_html(settings: Settings) -> str:
    apply_style()
    path = generate_cube(settings, steps=settings.market_maker.days)
    load_or_run_market_results(settings, path)
    generate_all_figures(settings, path)
    _render_equations()
    summary = pd.read_csv(TABLE_DIR / "results_summary.csv")
    arb = pd.read_csv(TABLE_DIR / "arb_violation_counts.csv")
    pca = pd.read_csv(TABLE_DIR / "pca_recovery.csv")
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(("html", "xml")),
    )
    template = environment.get_template("report.html.j2")
    html = template.render(pages=_build_pages(summary, arb, pca))
    RENDERED_HTML.write_text(html, encoding="utf-8")
    return html


def _write_pdf_weasyprint(html: str) -> str:
    stderr_buffer = io.StringIO()
    stdout_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            from weasyprint import HTML

            HTML(string=html, base_url=str(REPORT_DIR.resolve())).write_pdf(
                REPORT_DIR / "report.pdf"
            )
        return "weasyprint"
    except Exception as exc:
        reason = str(exc).splitlines()[0]
        if "libgobject" in reason or "pango" in reason.lower():
            reason = "native Pango/GObject libraries unavailable"
        print(f"WeasyPrint unavailable; falling back to Playwright ({type(exc).__name__}: {reason})")
        return _write_pdf_playwright()


def _find_chrome() -> str | None:
    candidates = [
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _write_pdf_playwright() -> str:
    from playwright.sync_api import sync_playwright

    chrome_path = _find_chrome()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=chrome_path,
        )
        page = browser.new_page(viewport={"width": 816, "height": 1056})
        page.emulate_media(media="print")
        page.goto(RENDERED_HTML.resolve().as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(REPORT_DIR / "report.pdf"),
            format="Letter",
            print_background=True,
            prefer_css_page_size=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        browser.close()
    return "playwright"


def _page(pdf: PdfPages, title: str, page_no: int) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.text(
        0.08,
        0.965,
        "CROSS-CUBE VEGA HEDGING",
        fontsize=7.6,
        color=PALETTE.graphite,
        family="DejaVu Sans",
        weight="bold",
    )
    ax.text(0.92, 0.965, str(page_no), fontsize=7.6, ha="right", color=PALETTE.graphite)
    ax.plot([0.08, 0.92], [0.948, 0.948], color=PALETTE.silver, linewidth=0.6)
    ax.text(
        0.08,
        0.918,
        title,
        fontsize=14,
        color=PALETTE.charcoal,
        family="DejaVu Sans",
        weight="bold",
    )
    ax.text(0.08, 0.032, "Imran Hakim - v0.1", fontsize=7.4, color=PALETTE.graphite)
    ax.plot([0.08, 0.92], [0.056, 0.056], color="#e4e4e4", linewidth=0.45)
    _ = pdf
    return fig, ax


def _save(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def _draw_wrapped(
    ax: plt.Axes,
    text: str,
    x: float,
    y: float,
    width: int,
    *,
    size: float = 8.6,
    leading: float = 0.018,
    color: str = PALETTE.charcoal,
    family: str = "DejaVu Serif",
    weight: str = "normal",
) -> float:
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    for paragraph in paragraphs:
        for line in textwrap.wrap(paragraph, width=width):
            ax.text(
                x,
                y,
                line,
                fontsize=size,
                color=color,
                family=family,
                weight=weight,
                va="top",
            )
            y -= leading
        y -= leading * 0.55
    return y


def _draw_subhead(ax: plt.Axes, text: str, x: float, y: float) -> float:
    ax.text(
        x,
        y,
        text,
        fontsize=10.3,
        color=PALETTE.oxblood,
        family="DejaVu Sans",
        weight="bold",
        va="top",
    )
    return y - 0.030


def _draw_equation(ax: plt.Axes, text: str, x: float, y: float, width: float = 0.38) -> float:
    box = plt.Rectangle(
        (x, y - 0.052),
        width,
        0.070,
        facecolor=PALETTE.off_white,
        edgecolor=PALETTE.silver,
        linewidth=0.5,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.014,
        y - 0.013,
        text,
        fontsize=9.0,
        color=PALETTE.charcoal,
        family="DejaVu Serif",
        va="center",
    )
    return y - 0.087


def _draw_table(ax: plt.Axes, spec: TableSpec, x: float, y: float, row_h: float = 0.030) -> float:
    ax.plot([x, x + sum(spec.widths)], [y, y], color=PALETTE.charcoal, linewidth=0.8)
    cursor = x
    for header, width in zip(spec.headers, spec.widths, strict=True):
        ax.text(
            cursor + 0.004,
            y - 0.020,
            header,
            fontsize=7.7,
            family="DejaVu Sans",
            weight="bold",
            color=PALETTE.charcoal,
            va="top",
        )
        cursor += width
    y -= row_h
    ax.plot([x, x + sum(spec.widths)], [y + 0.006, y + 0.006], color=PALETTE.silver, linewidth=0.5)
    for row_index, row in enumerate(spec.rows):
        if row_index % 2 == 1:
            ax.add_patch(
                plt.Rectangle(
                    (x, y - row_h + 0.004),
                    sum(spec.widths),
                    row_h,
                    facecolor="#f0f0f0",
                    edgecolor="none",
                    alpha=0.75,
                )
            )
        cursor = x
        for cell, width in zip(row, spec.widths, strict=True):
            ha = "left" if cursor == x else "right"
            offset = 0.004 if ha == "left" else width - 0.004
            ax.text(
                cursor + offset,
                y - 0.010,
                cell,
                fontsize=7.6,
                family="DejaVu Sans",
                color=PALETTE.charcoal,
                ha=ha,
                va="top",
            )
            cursor += width
        y -= row_h
    ax.plot([x, x + sum(spec.widths)], [y + 0.007, y + 0.007], color=PALETTE.charcoal, linewidth=0.7)
    return y - 0.016


def _draw_image(
    ax: plt.Axes,
    image_path: Path,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    caption: str,
) -> None:
    image = mpimg.imread(image_path)
    ax.imshow(image, extent=[x0, x1, y0, y1], aspect="auto")
    ax.text(
        x0,
        y0 - 0.018,
        caption,
        fontsize=7.4,
        family="DejaVu Serif",
        style="italic",
        color=PALETTE.graphite,
        va="top",
    )


def _money_mm(value: float) -> str:
    return f"{value / 1.0e6:,.1f}"


def _mpl_metric_table(summary: pd.DataFrame, strategies: Sequence[str]) -> TableSpec:
    frame = summary.set_index("strategy").loc[list(strategies)]
    rows = [
        [
            name,
            _money_mm(float(row["mean"])),
            _money_mm(float(row["std"])),
            _money_mm(float(row["var_5"])),
            f"{float(row['sharpe']):.2f}",
        ]
        for name, row in frame.iterrows()
    ]
    return TableSpec(
        headers=["Strategy", "Mean", "Std", "5% VaR", "Sharpe"],
        rows=rows,
        widths=[0.155, 0.070, 0.070, 0.075, 0.065],
    )


def _mpl_arb_table(arb: pd.DataFrame) -> TableSpec:
    columns = [column for column in ["fit", "violations", "butterfly", "calendar"] if column in arb]
    rows = [[str(row[column]) for column in columns] for _, row in arb.iterrows()]
    return TableSpec(
        headers=["Fit", "Total", "Butterfly", "Calendar"][: len(columns)],
        rows=rows,
        widths=[0.155, 0.060, 0.080, 0.080][: len(columns)],
    )


def _mpl_pca_table(pca: pd.DataFrame) -> TableSpec:
    rows = [
        [
            str(row["factor"]).title(),
            f"{float(row['loading_similarity']):.3f}",
            f"{100.0 * float(row['explained_variance']):.1f}%",
        ]
        for _, row in pca.iterrows()
    ]
    return TableSpec(
        headers=["Factor", "Cosine", "Expl. var"],
        rows=rows,
        widths=[0.120, 0.075, 0.085],
    )


def _mpl_tail_table(summary: pd.DataFrame) -> TableSpec:
    frame = summary.set_index("strategy")
    rows = [
        [
            name,
            _money_mm(float(row["var_1"])),
            _money_mm(float(row["var_5"])),
            _money_mm(float(row["cvar_5"])),
            _money_mm(float(row["std"])),
        ]
        for name, row in frame.iterrows()
    ]
    return TableSpec(
        headers=["Strategy", "1% VaR", "5% VaR", "5% CVaR", "Std"],
        rows=rows,
        widths=[0.175, 0.065, 0.065, 0.075, 0.060],
    )


def _cover(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.patch.set_facecolor(PALETTE.charcoal)
    ax.set_facecolor(PALETTE.charcoal)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.text(
        0.08,
        0.735,
        "Cross-Cube Vega Hedging",
        fontsize=34,
        color=PALETTE.silver,
        family="DejaVu Serif",
        va="top",
    )
    ax.text(
        0.08,
        0.675,
        "A Swaption Market-Making Lab",
        fontsize=18,
        color=PALETTE.off_white,
        family="DejaVu Serif",
        va="top",
    )
    ax.plot([0.08, 0.73], [0.624, 0.624], color=PALETTE.silver, linewidth=1.0)
    ax.text(0.08, 0.565, "Imran Hakim", fontsize=12.5, color=PALETTE.off_white)
    ax.text(
        0.08,
        0.532,
        "Independent research - produced as a portfolio artefact",
        fontsize=9.5,
        color=PALETTE.silver,
    )
    ax.text(0.08, 0.504, REPORT_DATE.strftime("%d %B %Y"), fontsize=9.5, color=PALETTE.silver)

    front = [(0.65, 0.15), (0.88, 0.15), (0.88, 0.38), (0.65, 0.38)]
    back = [(0.73, 0.24), (0.96, 0.24), (0.96, 0.47), (0.73, 0.47)]
    for square in [front, back]:
        xs = [point[0] for point in [*square, square[0]]]
        ys = [point[1] for point in [*square, square[0]]]
        ax.plot(xs, ys, color=PALETTE.silver, alpha=0.30, linewidth=1.2)
    for p0, p1 in zip(front, back, strict=True):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=PALETTE.silver, alpha=0.30, linewidth=1.2)
    _save(pdf, fig)


def _body_pages(pdf: PdfPages, summary: pd.DataFrame, arb: pd.DataFrame, pca: pd.DataFrame) -> None:
    page_no = 2

    fig, ax = _page(pdf, "Abstract and Executive Summary", page_no)
    y = 0.875
    y = _draw_subhead(ax, "Abstract", 0.08, y)
    y = _draw_wrapped(
        ax,
        "A swaption market-maker sees vega by cube point, but the realised cube does not move "
        "one point at a time. It moves through correlated factors: ATM level, expiry slope, "
        "curvature, and skew-wing dynamics. This lab simulates that cube, calibrates noisy SABR "
        "smiles, runs a client-flow market-maker, and compares four hedging policies. The "
        "factor-neutral rule compresses terminal P&L dispersion and improves the left tail. "
        "The limitation is explicit: no live market data, fixed beta, daily hedging, and a "
        "quoting overlay that is a factor-space heuristic rather than a full HJB solution.",
        0.08,
        y,
        84,
        size=8.7,
    )
    y = _draw_subhead(ax, "Executive Summary", 0.08, y - 0.004)
    y = _draw_wrapped(
        ax,
        "Single-tenor vega is a mirage. It is a convenient control number, but it is not the "
        "basis in which the cube actually moves. The same-axis terminal P&L overlay below is "
        "the central exhibit: delta-only hedging leaves a fat distribution, while factor-neutral "
        "hedging concentrates the outcome around a tighter risk budget. The point is not that "
        "factor-neutral hedging is free. It trades more, pays more hedge cost, and depends on "
        "liquid proxy instruments. The point is that those costs buy a cleaner residual.",
        0.08,
        y,
        84,
        size=8.7,
    )
    _draw_table(ax, _mpl_metric_table(summary, ["Delta-only", "Factor-neutral"]), 0.08, y - 0.008)
    _draw_image(
        ax,
        OUTPUT_FIGURE_DIR / "headline_pnl_dist.png",
        0.08,
        0.092,
        0.92,
        0.365,
        "Figure 1. Same-axis terminal P&L overlay; dashed lines mark strategy means.",
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Introduction and Motivation", page_no)
    y_left = 0.875
    y_left = _draw_subhead(ax, "Why Node Vega Exists", 0.08, y_left)
    y_left = _draw_wrapped(
        ax,
        "Node vega is the reporting standard because it is operationally legible. Traders quote "
        "points, risk systems aggregate points, and end-of-day controls ask whether a trader is "
        "long or short a recognisable part of the cube. For a desk, that convention is useful. "
        "For a hedge, it is incomplete. The cube does not wait for one node to move; the whole "
        "surface breathes through common shocks.",
        0.08,
        y_left,
        47,
    )
    y_left = _draw_subhead(ax, "The Risk Basis", 0.08, y_left)
    y_left = _draw_wrapped(
        ax,
        "A block in 2y into 10y exposes the book to more than the 2y x 10y mark. It has "
        "projection onto level, onto the front-back expiry slope, onto an intermediate-expiry "
        "hump, and onto skew and wing dynamics. A local vega hedge can be flat at the booked "
        "point and still be long the level factor. That is exactly the leakage this project "
        "tries to isolate.",
        0.08,
        y_left,
        47,
    )
    y_right = 0.875
    y_right = _draw_subhead(ax, "The Bartlett Pattern", 0.54, y_right)
    y_right = _draw_wrapped(
        ax,
        "Bartlett's SABR delta correction is a useful mental model. The Black delta is the "
        "obvious hedge only if volatility parameters stand still. Once alpha co-moves with the "
        "forward, the hedge needs the parameter response too. Cross-cube vega has the same "
        "shape: the obvious node hedge ignores the way nearby smiles move together.",
        0.54,
        y_right,
        47,
    )
    y_right = _draw_subhead(ax, "Trader's-Day Framing", 0.54, y_right)
    _draw_wrapped(
        ax,
        "The market-maker takes down flow, earns edge, and goes home with an inventory. The "
        "next morning the cube has moved. The question is not whether the single node was "
        "hedged, but whether the hedge was short the same factor the book was long. If it was "
        "not, the desk has paid spread for a residual it did not intend to warehouse.",
        0.54,
        y_right,
        47,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Architecture", page_no)
    _draw_image(
        ax,
        OUTPUT_FIGURE_DIR / "architecture.png",
        0.08,
        0.590,
        0.92,
        0.815,
        "Figure 2. Deterministic pipeline: simulator, calibration, analytics, market-maker, P&L.",
    )
    y = _draw_subhead(ax, "Pipeline Walk-Through", 0.08, 0.545)
    _draw_wrapped(
        ax,
        "The simulator produces a full SABR cube path: forwards, annuities, strikes, ATM vols, "
        "rho, nu, alpha, and quoted smile vols. Calibration consumes a noisy snapshot of those "
        "market vols and writes residual heatmaps plus static-arbitrage counts. PCA analytics "
        "recover the latent risk basis. The market-maker loop samples client flow, applies the "
        "quoting overlay, hedges at end-of-day, and emits path-level P&L. Attribution then "
        "reconciles edge, theta, delta, first-order vega, cross-vega, and hedge cost.",
        0.08,
        y,
        87,
    )
    y = _draw_subhead(ax, "Reproducibility Contract", 0.08, 0.315)
    _draw_wrapped(
        ax,
        "The master seed sits in YAML. The observation-noise stream is separate from the "
        "simulator streams, so calibrating a noisy market snapshot does not perturb the true "
        "cube path. The Makefile rebuilds simulations, figures, the static site, and this PDF "
        "from scratch.",
        0.08,
        y,
        87,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Model: True Cube I", page_no)
    y_left = _draw_subhead(ax, "ATM Factor Surface", 0.08, 0.875)
    y_left = _draw_wrapped(
        ax,
        "The cube is indexed by option expiry T in {1m, 3m, 6m, 1y, 2y, 5y, 10y} and "
        "underlying swap tenor tau in {1y, 2y, 5y, 10y}. The ATM Black-vol surface is driven "
        "by three smooth loadings over this grid.",
        0.08,
        y_left,
        47,
    )
    y_left = _draw_equation(
        ax,
        r"$\log\sigma_{ATM}(T,\tau,t)=\mu(T,\tau)+\sum_i L_i(T,\tau)X_i(t)$",
        0.08,
        y_left,
    )
    y_left = _draw_subhead(ax, "Loading Shapes", 0.08, y_left)
    _draw_wrapped(
        ax,
        "The level loading is nearly constant across the cube with a small tenor tilt. The slope "
        "loading is the centred log-expiry coordinate. The curvature loading is a Gaussian hump "
        "around the intermediate expiries, centred near two years, then normalised. These are "
        "not calibrated to live data; they are deliberately simple shapes that a trader would "
        "recognise from daily vol-surface moves.",
        0.08,
        y_left,
        47,
    )
    y_right = _draw_subhead(ax, "OU Dynamics", 0.54, 0.875)
    y_right = _draw_equation(
        ax,
        r"$dX_i=-\kappa_iX_i\,dt+\eta_i\,dW_i,\quad corr(dW)=R$",
        0.54,
        y_right,
    )
    y_right = _draw_wrapped(
        ax,
        "The OU factors mean-revert daily. Their instantaneous covariance is configurable in "
        "YAML, and the market-maker quoting overlay uses the full covariance matrix, including "
        "off-diagonal terms. That matters because level inventory and skew inventory should "
        "interact rather than widen quotes independently.",
        0.54,
        y_right,
        47,
    )
    y_right = _draw_subhead(ax, "Stylised Scale", 0.54, y_right)
    _draw_wrapped(
        ax,
        "The front of the normal-vol cube is set around the 60-90bp region and the back is "
        "anchored near 80bp, consistent with the broad scale used in interest-rate volatility "
        "modelling examples. Absolute levels are laboratory choices, not market claims.",
        0.54,
        y_right,
        47,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Model: True Cube II", page_no)
    y_left = _draw_subhead(ax, "Skew and Wing Dynamics", 0.08, 0.875)
    y_left = _draw_wrapped(
        ax,
        "Each expiry-tenor slice carries SABR parameters alpha, beta, rho, and nu. Beta is fixed "
        "at the configured value. Rho is generated through a transformed OU process, so it "
        "stays inside [-0.95, 0.0]. Log-nu is another OU process. Both shocks are correlated "
        "with the level factor, reflecting the stylised observation that skew and volatility "
        "level do not move independently.",
        0.08,
        y_left,
        47,
    )
    y_left = _draw_subhead(ax, "ATM Inversion", 0.08, y_left)
    _draw_wrapped(
        ax,
        "The simulator specifies ATM vol first, then recovers alpha from Hagan's ATM SABR "
        "formula. This keeps the factor model attached to the observable surface rather than "
        "to an abstract alpha process. Calibration mirrors market practice by enforcing the "
        "ATM point and fitting smile shape through rho and nu.",
        0.08,
        y_left,
        47,
    )
    y_right = _draw_subhead(ax, "SABR and Vol Conversion", 0.54, 0.875)
    y_right = _draw_wrapped(
        ax,
        "The pricer implements Hagan's 2002 lognormal SABR formula with a shifted extension for "
        "negative-rate laboratory cases. Black-76 and Bachelier pricing are both present. "
        "Normal/lognormal conversion is done by Brent root-finding on price equality rather "
        "than by a closed-form approximation; the extra cost is small and the audit trail is "
        "clean.",
        0.54,
        y_right,
        47,
    )
    y_right = _draw_subhead(ax, "Parameter Honesty", 0.54, y_right)
    _draw_wrapped(
        ax,
        "The configuration file is not a calibration file. It records a plausible set of "
        "stylised facts: mean-reverting factors, correlated level/skew shocks, bounded rho, "
        "and liquid hedge points. The report should be read as a controlled experiment, not as "
        "a claim about today's swaption market.",
        0.54,
        y_right,
        47,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Model: Cube Snapshots", page_no)
    _draw_image(
        ax,
        OUTPUT_FIGURE_DIR / "atm_cube_snapshots.png",
        0.08,
        0.455,
        0.92,
        0.825,
        "Figure 3. Three ATM cube snapshots from the simulated path.",
    )
    y = _draw_subhead(ax, "Reading the Heatmaps", 0.08, 0.405)
    _draw_wrapped(
        ax,
        "The snapshots are deliberately boring in the right way: the whole cube changes, and "
        "nearby expiries and tenors change together. A single isolated node shock would make "
        "the hedging problem artificial. The thesis needs a cube that moves in common modes, "
        "because that is where local vega reports become a lossy compression.",
        0.08,
        y,
        87,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Calibration I", page_no)
    y_left = _draw_subhead(ax, "Noisy Market Snapshot", 0.08, 0.875)
    y_left = _draw_wrapped(
        ax,
        "The calibration target is no longer the exact data-generating SABR surface. Each strike "
        "vol receives independent multiplicative lognormal observation noise: "
        "sigma_market = sigma_true exp(eps), eps ~ N(0, sigma_obs^2), with sigma_obs = 0.005 by "
        "default. The random stream is separate from the simulator stream.",
        0.08,
        y_left,
        47,
    )
    y_left = _draw_subhead(ax, "Per-Slice Fit", 0.08, y_left)
    _draw_wrapped(
        ax,
        "Per-slice calibration fixes beta, matches ATM exactly through alpha inversion, and "
        "fits rho and nu by least squares. That is close to the workflow a desk would use when "
        "bootstrapping individual smile slices. It gives low local residuals, but it has no "
        "reason to respect calendar consistency across neighbouring expiries.",
        0.08,
        y_left,
        47,
    )
    y_right = _draw_subhead(ax, "Cube-Constrained Fit", 0.54, 0.875)
    y_right = _draw_wrapped(
        ax,
        "The joint objective keeps the same ATM discipline and adds penalties for butterfly "
        "and calendar violations, plus a smaller smoothness penalty on rho and log-nu over "
        "the cube. The no-arbitrage penalties dominate smoothness by design. This is not a "
        "local least-squares beauty contest; it is a risk-control fit.",
        0.54,
        y_right,
        47,
    )
    y_right = _draw_equation(
        ax,
        r"$J=||\sigma_m-\sigma_q||^2+\lambda_bB+\lambda_cC+\lambda_sS$",
        0.54,
        y_right,
    )
    _draw_wrapped(
        ax,
        "The diagnostic question is whether a small residual cost buys fewer static-arbitrage "
        "breaks. If it does not, the constrained calibration story fails. The table on the "
        "next page is therefore a core acceptance test, not decoration.",
        0.54,
        y_right,
        47,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Calibration II", page_no)
    _draw_image(
        ax,
        OUTPUT_FIGURE_DIR / "calibration_residual_heatmaps.png",
        0.08,
        0.505,
        0.92,
        0.830,
        "Figure 4. Residual RMS in vol bp. Joint calibration pays local residual for consistency.",
    )
    y = _draw_subhead(ax, "Static-Arbitrage Counts", 0.08, 0.455)
    y = _draw_table(ax, _mpl_arb_table(arb), 0.08, y)
    _draw_wrapped(
        ax,
        "The noisy per-slice fit now has genuine residuals instead of a meaningless 1e-12 "
        "colourbar. The constrained fit is slightly worse locally, but it removes the material "
        "calendar breaks under the same checker. This is the calibration analogue of the "
        "hedging thesis: a local optimum can be the wrong portfolio object.",
        0.08,
        y,
        87,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Factor Recovery via PCA I", page_no)
    y_left = _draw_subhead(ax, "Method", 0.08, 0.875)
    y_left = _draw_wrapped(
        ax,
        "PCA is run on daily changes in the simulated ATM log-vol surface. The first three "
        "components explain the designed level, slope, and curvature subspace. Because PCA "
        "basis vectors are sign-ambiguous, the diagnostic aligns them to the known simulator "
        "loadings and flips signs consistently.",
        0.08,
        y_left,
        47,
    )
    y_left = _draw_subhead(ax, "Cosine Similarity", 0.08, y_left)
    _draw_table(ax, _mpl_pca_table(pca), 0.08, y_left)
    y_right = _draw_subhead(ax, "What This Validates", 0.54, 0.875)
    y_right = _draw_wrapped(
        ax,
        "The table is not claiming that PCA can name economic factors in real data without "
        "judgement. It says something narrower and important: in this controlled cube, the "
        "analytics recover the latent risk subspace used by the hedger. That is enough to make "
        "the hedging comparison internally coherent.",
        0.54,
        y_right,
        47,
    )
    y_right = _draw_subhead(ax, "What It Does Not Validate", 0.54, y_right)
    _draw_wrapped(
        ax,
        "It does not validate the parameter levels, the absence of jumps, or the stability of "
        "factor loadings through regimes. A live implementation would rerun this diagnostic "
        "over rolling windows and ask whether factor labels are stable enough to trade.",
        0.54,
        y_right,
        47,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Factor Recovery via PCA II", page_no)
    _draw_image(
        ax,
        OUTPUT_FIGURE_DIR / "pca_recovery.png",
        0.08,
        0.445,
        0.92,
        0.825,
        "Figure 5. Dashed true loadings and solid recovered loadings are normalised consistently.",
    )
    y = _draw_subhead(ax, "Reading the Figure", 0.08, 0.395)
    _draw_wrapped(
        ax,
        "The recovered curves are plotted against the same expiry axis and colour as the true "
        "factor labels. The diagnostic was previously misleading because the true loadings and "
        "recovered eigenvectors lived on different scales. Both are now unit-normalised before "
        "plotting, and the sign ambiguity is resolved before the cosine table is written.",
        0.08,
        y,
        87,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Hedging Strategies I", page_no)
    y_left = _draw_subhead(ax, "Delta-Only", 0.08, 0.875)
    y_left = _draw_wrapped(
        ax,
        "The control hedge removes underlying swap delta and leaves vega to run. It is useful "
        "as a lower bound because it captures the cost of pretending the cube is not the main "
        "risk. Delta P&L should be small in expectation, but vega and cross-vega dominate.",
        0.08,
        y_left,
        47,
    )
    y_left = _draw_subhead(ax, "Single-Tenor Vega-Flat", 0.08, y_left)
    _draw_wrapped(
        ax,
        "Each new trade's vega is snapped to the nearest cube node and offset by trading that "
        "node. This looks clean in a blotter and can be deeply wrong in factor space if the "
        "offset node loads on the wrong combination of level, slope, and curvature.",
        0.08,
        y_left,
        47,
    )
    y_right = _draw_subhead(ax, "Bucketed Vega-Flat", 0.54, 0.875)
    y_right = _draw_wrapped(
        ax,
        "Bucketed hedging nets vega within expiry buckets: <=1y, 1y-3y, 3y-7y, and >7y. This "
        "is a practical compromise. It cannot fully neutralise factor exposure, but it stops "
        "the most obvious front/back leakage and uses fewer hedge instruments.",
        0.54,
        y_right,
        47,
    )
    y_right = _draw_subhead(ax, "Portfolio Greek Matrix", 0.54, y_right)
    _draw_wrapped(
        ax,
        "Let V be node vega and L be the loading matrix. Raw vega reports V. Factor risk is "
        "closer to L'V, augmented by a skew exposure. The hedging question is therefore not "
        "only whether sum(V)=0, but whether the projected vector is small.",
        0.54,
        y_right,
        47,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Hedging Strategies II", page_no)
    y_left = _draw_subhead(ax, "Factor-Neutral Hedge", 0.08, 0.875)
    y_left = _draw_wrapped(
        ax,
        "The factor-neutral rule selects liquid cube points and solves a small constrained "
        "hedge to bring level, slope, curvature, and skew exposures near zero. It minimises "
        "residual factor exposure subject to trading-cost and instrument-size penalties.",
        0.08,
        y_left,
        47,
    )
    y_left = _draw_equation(
        ax,
        r"$\min_h ||A(q+Bh)||^2+\lambda||h||^2+c^\top |h|$",
        0.08,
        y_left,
    )
    _draw_wrapped(
        ax,
        "In trader language: use the liquid points that actually load on the factors you are "
        "trying to offset, and avoid over-trading an ill-conditioned local hedge.",
        0.08,
        y_left,
        47,
    )
    y_right = _draw_subhead(ax, "Instrument Selection", 0.54, 0.875)
    y_right = _draw_wrapped(
        ax,
        "The available hedge set is smaller than the risk grid. That is intentional. A market "
        "maker normally has a handful of executable liquid points and must project the book "
        "onto that basis. The factor-neutral hedge is therefore a practical proxy hedge, not a "
        "fantasy where every node trades frictionlessly.",
        0.54,
        y_right,
        47,
    )
    y_right = _draw_subhead(ax, "Cost Accounting", 0.54, y_right)
    _draw_wrapped(
        ax,
        "Every hedge trade pays a configurable bid-ask cost in vol bp. The attribution table "
        "keeps that cost separate from cross-vega residuals, so the reader can see whether the "
        "tighter distribution was bought with too much turnover.",
        0.54,
        y_right,
        47,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Market-Maker Simulation", page_no)
    y_left = _draw_subhead(ax, "Client Flow", 0.08, 0.875)
    y_left = _draw_wrapped(
        ax,
        "Client trades arrive from a Poisson process. Cube point, strike offset, side, and "
        "notional are sampled from simple distributions. The client crosses the quote with "
        "probability near one. That simplification keeps the experiment focused on inventory "
        "and hedging rather than on a separate client-response model.",
        0.08,
        y_left,
        47,
    )
    y_left = _draw_subhead(ax, "Book State", 0.08, y_left)
    _draw_wrapped(
        ax,
        "After each trade, inventory Greeks and factor exposures update. Hedging fires at "
        "end-of-day. The simulation records the book before and after hedging so P&L can be "
        "attributed to edge, theta, delta, first-order vega, cross-vega, and hedge cost.",
        0.08,
        y_left,
        47,
    )
    y_right = _draw_subhead(ax, "Monte Carlo Scale", 0.54, 0.875)
    y_right = _draw_wrapped(
        ax,
        "The default run uses 1,000 market-making paths over 252 trading days. Tests use small "
        "deterministic seeds and avoid Monte Carlo. Generated simulations are cached under "
        "outputs/simulations, while make clean removes them to prove reproducibility.",
        0.54,
        y_right,
        47,
    )
    y_right = _draw_subhead(ax, "Simplifications", 0.54, y_right)
    _draw_wrapped(
        ax,
        "There is no adverse-selection model, no funding spread, no exchange margin, and no "
        "jump risk. Those choices make the experiment narrower, but they also make the hedge "
        "comparison easier to audit.",
        0.54,
        y_right,
        47,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Quoting Overlay", page_no)
    y = _draw_subhead(ax, "Reservation Vol in Factor Space", 0.08, 0.875)
    y = _draw_wrapped(
        ax,
        "The quote mid is adjusted by an Avellaneda-Stoikov-style inventory charge, but the "
        "inventory vector is factor exposure rather than a single scalar position.",
        0.08,
        y,
        87,
    )
    y = _draw_equation(
        ax,
        r"$r_i=mid_i-\gamma q^\top\Sigma l_i(T_{max}-t)$",
        0.08,
        y,
        width=0.84,
    )
    _draw_image(
        ax,
        OUTPUT_FIGURE_DIR / "quoting_half_spread_heatmap.png",
        0.08,
        0.115,
        0.50,
        0.520,
        "Figure 6. Half-spread responds to level and skew inventory together.",
    )
    _draw_wrapped(
        ax,
        "The heatmap is intentionally no longer flat. The displayed grid spans a wide enough "
        "inventory range for the risk-aversion term to matter, and the covariance matrix is "
        "used with off-diagonal terms intact. The gradient is therefore not purely axis-aligned. "
        "This remains a heuristic overlay, not a full dynamic-programming derivation.",
        0.56,
        0.505,
        40,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Results I: Representative Path", page_no)
    _draw_image(
        ax,
        OUTPUT_FIGURE_DIR / "representative_mm_path.png",
        0.08,
        0.405,
        0.92,
        0.825,
        "Figure 7. One path shows how the same client flow diverges under different hedges.",
    )
    y = _draw_subhead(ax, "Path-Level Read", 0.08, 0.355)
    _draw_wrapped(
        ax,
        "A single path should not be over-interpreted, but it is useful for sanity. The "
        "strategies see the same simulated flow and cube path. Divergence comes from the hedge "
        "state, hedge costs, and residual factor exposures rather than from different market "
        "draws.",
        0.08,
        y,
        87,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Results II: Distributions and Tail Risk", page_no)
    _draw_image(
        ax,
        OUTPUT_FIGURE_DIR / "pnl_dist_small_multiples.png",
        0.08,
        0.555,
        0.92,
        0.820,
        "Figure 8. Small multiples keep the same x-axis while showing each strategy separately.",
    )
    y = _draw_subhead(ax, "Tail-Risk Table ($mm)", 0.08, 0.500)
    _draw_table(ax, _mpl_tail_table(summary), 0.08, y, row_h=0.027)
    _draw_wrapped(
        ax,
        "The same-axis overlay in the executive summary is the headline picture. These small "
        "multiples make the same point strategy by strategy: delta-only carries the broadest "
        "distribution, bucketed hedging improves the shape, and factor-neutral hedging is the "
        "tightest in this experiment.",
        0.08,
        0.218,
        87,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Results III: P&L Attribution", page_no)
    _draw_image(
        ax,
        OUTPUT_FIGURE_DIR / "pnl_attribution.png",
        0.08,
        0.445,
        0.92,
        0.820,
        "Figure 9. Attribution reconciles spread edge, Greeks, residual cross-vega, and cost.",
    )
    y = _draw_subhead(ax, "Trader-Tone Read", 0.08, 0.395)
    _draw_wrapped(
        ax,
        "The factor-neutral hedge does not create edge; it protects the edge already earned. "
        "Its cost is visible and should be challenged. The reason to like it is that the "
        "cross-vega residual is less dominant, so the P&L is less dependent on being lucky "
        "about which cube factor moved after the trade.",
        0.08,
        y,
        87,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Results IV: What the Experiment Says", page_no)
    y_left = _draw_subhead(ax, "Headline", 0.08, 0.875)
    y_left = _draw_wrapped(
        ax,
        "The result supports the core thesis in the controlled setting: a hedge constructed in "
        "the cube's factor basis produces a tighter realised P&L distribution than a hedge "
        "constructed only in local node vega. The improvement is clearest in the left tail and "
        "in the residual attribution.",
        0.08,
        y_left,
        47,
    )
    y_left = _draw_subhead(ax, "Calibration Link", 0.08, y_left)
    _draw_wrapped(
        ax,
        "The calibration section tells the same story in model space. Per-slice fitting wins "
        "the local residual contest. The cube-constrained fit is the better surface object "
        "because it removes static-arbitrage breaks under a symmetric checker.",
        0.08,
        y_left,
        47,
    )
    y_right = _draw_subhead(ax, "Caveat", 0.54, 0.875)
    y_right = _draw_wrapped(
        ax,
        "This is not a live trading backtest. It is a market-making lab. The aim is to make the "
        "mechanics transparent enough that a reviewer can disagree with the parameters, rerun "
        "the project, and see how the result changes.",
        0.54,
        y_right,
        47,
    )
    y_right = _draw_subhead(ax, "Desk Use", 0.54, y_right)
    _draw_wrapped(
        ax,
        "The immediate desk lesson is not to throw away node vega. Keep it for operations, but "
        "report the factor projection beside it. A position that is node-flat and factor-long "
        "should not be called flat.",
        0.54,
        y_right,
        47,
    )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Assumptions and Limitations", page_no)
    limits = [
        ("Simulated data only", "No conclusion depends on a hidden vendor feed, but absolute P&L levels are not live-market forecasts."),
        ("Single-curve discounting", "The lab ignores multi-curve basis and collateral details that would matter in production pricing."),
        ("Fixed beta", "SABR beta is configurable but not calibrated; this keeps the exercise focused on cross-cube vega."),
        ("No jumps", "Daily OU dynamics miss event risk, central-bank gap moves, and regime shifts."),
        ("Daily hedging", "Intraday inventory management is compressed into an end-of-day hedge."),
        ("Perfectly observed mids", "There is no mark uncertainty or broker-quality hierarchy."),
        ("Heuristic quoting", "The factor-space reservation vol borrows the Avellaneda-Stoikov shape but is not a full HJB solution."),
    ]
    y = 0.875
    for title, body in limits:
        y = _draw_wrapped(
            ax,
            f"{title}. {body}",
            0.08,
            y,
            88,
            size=8.4,
            leading=0.016,
        )
    _save(pdf, fig)
    page_no += 1

    fig, ax = _page(pdf, "Extensions", page_no)
    extensions = [
        ("Real-data overlay", "Use exchange settles or broker indicative marks to initialise the cube and compare factor stability."),
        ("Rough-vol dynamics", "Replace OU factors with rougher latent drivers and test whether hedging cadence becomes more important."),
        ("Multi-curve pricing", "Add OIS discounting and forward curves so the swaption lab better resembles production infrastructure."),
        ("Joint cap-swaption calibration", "Bring caplets into the smile calibration and ask whether short-rate consistency changes the hedge basis."),
        ("Bermudan extension", "Use least-squares Monte Carlo to push the factor hedge into callable-exotics inventory."),
        ("Strategic quoting", "Let IDB visibility and client response feed back into the reservation-vol rule."),
        ("Learning overlay", "Use reinforcement learning as a policy layer on top of the transparent factor state, not as a black-box replacement."),
    ]
    y = 0.875
    for title, body in extensions:
        y = _draw_subhead(ax, title, 0.08, y)
        y = _draw_wrapped(ax, body, 0.08, y, 88, size=8.4, leading=0.016)
    _save(pdf, fig)
    page_no += 1

    references = [
        ("Andersen and Piterbarg (2010)", "Interest Rate Modeling. Atlantic Financial Press. The project borrows its interest-rate modelling discipline from this reference: explicit curves, model assumptions, and no mystery calibration knobs."),
        ("Hagan, Kumar, Lesniewski, and Woodward (2002)", "Managing Smile Risk. Wilmott Magazine. The SABR implementation follows the lognormal asymptotic implied-vol formula and the ATM treatment used throughout the lab."),
        ("Bartlett (2006)", "Hedging Under SABR Model. Wilmott Magazine. The Bartlett delta/vega discussion motivates the broader point that parameters moving with the underlying change the hedge."),
        ("Rebonato (2002)", "Modern Pricing of Interest-Rate Derivatives. Princeton University Press. Used as background for swaption smile practice and interest-rate volatility intuition."),
        ("Bergomi (2016)", "Stochastic Volatility Modeling. Chapman and Hall/CRC. Used for volatility-surface factor thinking and the distinction between local quotes and surface dynamics."),
    ]
    fig, ax = _page(pdf, "References I", page_no)
    y = 0.875
    for title, body in references:
        y = _draw_subhead(ax, title, 0.08, y)
        y = _draw_wrapped(ax, body, 0.08, y, 88, size=8.3, leading=0.016)
    _save(pdf, fig)
    page_no += 1

    references_2 = [
        ("Avellaneda and Stoikov (2008)", "High-frequency trading in a limit order book. Quantitative Finance 8(3), 217-224. The quote overlay borrows the reservation-price intuition, with the caveat that this project adapts it heuristically to vol-space factors."),
        ("Gatheral (2006)", "The Volatility Surface: A Practitioner's Guide. Wiley. Background for smile/surface diagnostics and the habit of thinking in no-arbitrage surface terms."),
        ("Brigo and Mercurio (2006)", "Interest Rate Models: Theory and Practice. Springer. Background for rate derivatives, annuities, and swaption pricing conventions."),
        ("Glasserman (2004)", "Monte Carlo Methods in Financial Engineering. Springer. Background for deterministic simulation design, reproducibility, and path-level diagnostics."),
    ]
    fig, ax = _page(pdf, "References II", page_no)
    y = 0.875
    for title, body in references_2:
        y = _draw_subhead(ax, title, 0.08, y)
        y = _draw_wrapped(ax, body, 0.08, y, 88, size=8.3, leading=0.016)
    y = _draw_subhead(ax, "Build Note", 0.08, y)
    _draw_wrapped(
        ax,
        "The sandbox used for this build did not provide XeLaTeX or dvisvgm. The repository "
        "still writes full LaTeX sources and a TikZ architecture diagram, while the delivered "
        "PDF is rendered through a deterministic Matplotlib fallback with the same palette.",
        0.08,
        y,
        88,
        size=8.3,
        leading=0.016,
    )
    _save(pdf, fig)


def _fallback_pdf(settings: Settings) -> None:
    apply_style()
    path = generate_cube(settings, steps=settings.market_maker.days)
    load_or_run_market_results(settings, path)
    generate_all_figures(settings, path)
    summary = pd.read_csv(TABLE_DIR / "results_summary.csv")
    arb = pd.read_csv(TABLE_DIR / "arb_violation_counts.csv")
    pca = pd.read_csv(TABLE_DIR / "pca_recovery.csv")
    output = REPORT_DIR / "report.pdf"
    pdf_metadata = {
        "Title": TITLE,
        "Author": "Imran Hakim",
        "Creator": "cxvega build_pdf_report.py",
        "CreationDate": datetime(2026, 5, 22, tzinfo=UTC),
        "ModDate": datetime(2026, 5, 22, tzinfo=UTC),
    }
    with PdfPages(output, metadata=pdf_metadata) as pdf:
        _cover(pdf)
        _body_pages(pdf, summary, arb, pca)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PDF report.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--fallback-matplotlib",
        action="store_true",
        help="Use the emergency Matplotlib renderer instead of HTML/CSS PDF output.",
    )
    args = parser.parse_args()
    settings = load_settings(args.config)
    ensure_dirs()
    _write_latex_sources(settings)
    if args.fallback_matplotlib:
        _fallback_pdf(settings)
        renderer = "matplotlib"
    else:
        html = _render_report_html(settings)
        renderer = _write_pdf_weasyprint(html)
    size = (REPORT_DIR / "report.pdf").stat().st_size
    print(f"renderer: {renderer}")
    print(f"wrote {REPORT_DIR / 'report.pdf'} ({size} bytes)")


if __name__ == "__main__":
    main()
