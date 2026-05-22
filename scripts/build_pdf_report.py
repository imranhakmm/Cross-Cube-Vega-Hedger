from __future__ import annotations

import argparse
import shutil
import subprocess
import textwrap
from datetime import UTC, date, datetime
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
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


def _write_latex_sources(settings: Settings) -> None:
    report_tex = r"""
\documentclass[11pt]{article}
\usepackage{style/cxvega}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\title{Cross-Cube Vega Hedging: A Swaption Market-Making Lab}
\author{Imran Hakim}
\date{22 May 2026}
\begin{document}
\maketitle
\begin{abstract}
Single-tenor vega is a lossy risk report for a swaption market-maker. This
lab simulates a factor-moving swaption cube, calibrates SABR slices, and
compares delta-only, node, bucketed, and factor-neutral hedges.
\end{abstract}
\section{Core thesis}
Vega is booked per point, but the cube moves in level, slope, curvature, and
skew factors. Hedges that zero a local number can still leak factor P\&L.
\section{Model}
The simulator uses Hagan SABR slices with fixed beta and OU dynamics for ATM
log-vol, rho, and log-nu. Calibration enforces the ATM constraint and then adds
cube smoothness and arbitrage penalties.
\section{Results}
See the generated figures and tables in this repository. The Python fallback
report expands this source into the full research note when a TeX toolchain is
not available.
\bibliographystyle{plain}
\bibliography{references}
\end{document}
"""
    references = r"""
@book{andersenpiterbarg2010,
  title={Interest Rate Modeling},
  author={Andersen, Leif B. G. and Piterbarg, Vladimir V.},
  year={2010},
  publisher={Atlantic Financial Press}
}
@article{hagan2002,
  title={Managing Smile Risk},
  author={Hagan, Patrick S. and Kumar, Deep and Lesniewski, Andrew S. and Woodward, Diana E.},
  journal={Wilmott Magazine},
  year={2002}
}
@book{rebonato2002,
  title={Modern Pricing of Interest-Rate Derivatives},
  author={Rebonato, Riccardo},
  year={2002},
  publisher={Princeton University Press}
}
@book{bergomi2016,
  title={Stochastic Volatility Modeling},
  author={Bergomi, Lorenzo},
  year={2016},
  publisher={Chapman and Hall/CRC}
}
@article{avellanedastoikov2008,
  title={High-frequency trading in a limit order book},
  author={Avellaneda, Marco and Stoikov, Sasha},
  journal={Quantitative Finance},
  year={2008}
}
@book{gatheral2006,
  title={The Volatility Surface},
  author={Gatheral, Jim},
  year={2006},
  publisher={Wiley}
}
@article{bartlett2006,
  title={Hedging under SABR Model},
  author={Bartlett, Bruce},
  journal={Wilmott Magazine},
  year={2006}
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


def _new_page(title: str, page_no: int) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.08, 0.965, "Cross-Cube Vega Hedging", fontsize=8, color=PALETTE.graphite)
    ax.text(0.92, 0.965, str(page_no), fontsize=8, ha="right", color=PALETTE.graphite)
    ax.plot([0.08, 0.92], [0.948, 0.948], color=PALETTE.silver, linewidth=0.8)
    ax.text(0.08, 0.91, title, fontsize=18, color=PALETTE.charcoal, weight="bold")
    ax.text(0.08, 0.035, "Imran Hakim - version 0.1", fontsize=8, color=PALETTE.graphite)
    return fig, ax


def _paragraphs(ax: plt.Axes, paragraphs: list[str], y_start: float = 0.86) -> None:
    y = y_start
    for paragraph in paragraphs:
        for line in textwrap.wrap(paragraph, width=92):
            ax.text(0.08, y, line, fontsize=10.5, color=PALETTE.charcoal, va="top")
            y -= 0.026
        y -= 0.024


def _image_page(pdf: PdfPages, title: str, image_path: Path, page_no: int, caption: str) -> None:
    fig, ax = _new_page(title, page_no)
    image = mpimg.imread(image_path)
    ax.imshow(image, extent=[0.08, 0.92, 0.23, 0.82], aspect="auto")
    ax.text(0.08, 0.18, caption, fontsize=9.5, color=PALETTE.graphite, style="italic")
    pdf.savefig(fig)
    plt.close(fig)


def _table_text(df: pd.DataFrame, columns: list[str]) -> list[str]:
    subset = df[columns].copy()
    for column in subset.columns:
        if column == "strategy":
            continue
        subset[column] = subset[column].map(
            lambda value: f"{float(value) / 1.0e6:,.2f}"
            if abs(float(value)) > 1.0e3
            else f"{float(value):,.2f}"
        )
    return subset.to_string(index=False).splitlines()


def _fallback_pdf(settings: Settings) -> None:
    apply_style()
    path = generate_cube(settings, steps=settings.market_maker.days)
    load_or_run_market_results(settings, path)
    generate_all_figures(settings, path)
    summary = pd.read_csv("outputs/tables/results_summary.csv")
    arb = pd.read_csv("outputs/tables/arb_violation_counts.csv")
    output = REPORT_DIR / "report.pdf"
    pdf_metadata = {
        "Title": "Cross-Cube Vega Hedging: A Swaption Market-Making Lab",
        "Author": "Imran Hakim",
        "Creator": "cxvega build_pdf_report.py",
        "CreationDate": datetime(2026, 5, 22, tzinfo=UTC),
        "ModDate": datetime(2026, 5, 22, tzinfo=UTC),
    }
    with PdfPages(output, metadata=pdf_metadata) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_facecolor(PALETTE.charcoal)
        fig.patch.set_facecolor(PALETTE.charcoal)
        ax.axis("off")
        ax.text(
            0.08,
            0.72,
            "Cross-Cube Vega Hedging",
            fontsize=34,
            color=PALETTE.silver,
            family="serif",
        )
        ax.text(
            0.08,
            0.66,
            "A Swaption Market-Making Lab",
            fontsize=21,
            color=PALETTE.off_white,
            family="serif",
        )
        ax.plot([0.08, 0.72], [0.61, 0.61], color=PALETTE.silver, linewidth=1.2)
        ax.text(0.08, 0.54, "Imran Hakim", fontsize=12, color=PALETTE.off_white)
        ax.text(0.08, 0.50, "Independent research - produced as a portfolio artefact", fontsize=10, color=PALETTE.silver)
        ax.text(0.08, 0.46, date(2026, 5, 22).strftime("%d %B %Y"), fontsize=10, color=PALETTE.silver)
        cube_x = [0.72, 0.88, 0.88, 0.72, 0.72, 0.78, 0.94, 0.94, 0.88, 0.94, 0.78, 0.72]
        cube_y = [0.17, 0.17, 0.33, 0.33, 0.17, 0.23, 0.23, 0.39, 0.33, 0.39, 0.39, 0.33]
        ax.plot(cube_x, cube_y, color=PALETTE.silver, linewidth=0.9)
        pdf.savefig(fig)
        plt.close(fig)

        pages = [
            (
                "Abstract",
                [
                    "A swaption market-maker sees vega by cube point, but the realised cube does not move one point at a time. It moves through correlated factors: ATM level, expiry slope, curvature, and skew-wing dynamics.",
                    "This lab builds a simulated cube, calibrates SABR smiles, runs a client-flow market-maker, and compares four hedging policies. The factor-neutral policy is not magic: it pays more hedge cost and relies on liquid proxy instruments. It does, however, make the residual cross-cube P&L smaller and easier to explain.",
                ],
            ),
            (
                "Executive Summary",
                [
                    "The headline result is mechanical and useful: hedging raw node vega can leave the book long one cube factor and short another. Bucketed hedges reduce the leak. Factor-neutral hedges reduce it further, especially in the left tail.",
                    "The experiment uses no real market data. Parameter choices are set to published stylised facts and kept explicit in YAML so the reader can change them and rebuild the note.",
                ],
            ),
            (
                "Introduction",
                [
                    "Node vega is operationally convenient. Traders quote, risk systems aggregate, and hedgers choose liquid points. But the reporting basis is not the risk basis.",
                    "A client trade in a 2y into 10y payer does not expose the book only to the 2y x 10y mark. It loads on the vol level, on the front-back slope, on the hump in intermediate expiries, and on smile dynamics. This is where naive hedges leak P&L.",
                ],
            ),
            (
                "Architecture",
                [
                    "The project is deliberately modular. The simulator creates the true cube. Calibration recovers per-slice and constrained SABR parameters. Analytics recover factors. The market-maker loop creates inventory, the hedger trades proxy instruments, and attribution reconciles the realised P&L.",
                ],
            ),
            (
                "Model: True Cube",
                [
                    "ATM log-vol is modelled as mu(T,tau) plus three smooth loadings times OU factors. The loadings are level, log-expiry slope, and a humped curvature mode.",
                    "SABR beta is fixed. Rho follows a bounded OU process in [-0.95, 0.0]. Log-nu follows OU dynamics. Both receive shocks correlated with the level factor so skew steepens when volatility rises.",
                ],
            ),
            (
                "Stylised Facts",
                [
                    "The front of the normal-vol cube is set around 60-90bp and the back is anchored near 80bp. The exact values are not calibrated to a live market; they are a plausible laboratory scale.",
                    "The report cites Andersen-Piterbarg, Hagan et al., Rebonato, and Bergomi for interest-rate volatility modelling conventions, SABR smile dynamics, and volatility-surface factor behaviour.",
                ],
            ),
            (
                "Calibration",
                [
                    "Per-slice calibration follows market practice: beta fixed, ATM vol matched exactly, and rho and nu fitted to the smile. The joint fit adds smoothness and static-arbitrage penalties.",
                    "When the joint fit sacrifices a small amount of local residual to reduce cube-level roughness, that is not hidden. The residual heatmaps and violation count table expose the trade-off.",
                ],
            ),
            (
                "Factor Recovery",
                [
                    "PCA is run on simulated ATM log-vol changes. The first three components are aligned by cosine similarity with the true loadings. This avoids penalising the nearly parallel level mode for having a large constant component.",
                ],
            ),
            (
                "Hedging Strategies",
                [
                    "Delta-only removes underlying swap delta and lets vega run. Single-tenor vega-flat offsets tenor-level raw vega into canonical liquid nodes. Bucketed vega-flat controls expiry buckets. Factor-neutral solves a small quadratic hedge in level, slope, curvature, and skew exposure.",
                ],
            ),
            (
                "Market-Maker Loop",
                [
                    "Client trades arrive as a Poisson process. Cube point, side, strike offset, and notional are sampled. The client crosses the quote by construction, which is a simplification and gives the market-maker a clean spread-capture signal.",
                ],
            ),
            (
                "Quoting Overlay",
                [
                    "The reservation vol follows an Avellaneda-Stoikov-style inventory charge in factor space. This is a heuristic adaptation, not a full HJB derivation for a swaption cube. The value is practical: widen where the next trade loads the already painful factor.",
                ],
            ),
            (
                "Results",
                [
                    "The factor-neutral hedge generally tightens the terminal distribution and reduces cross-vega residuals. Bucketed hedging captures much of the improvement with less instrumentation.",
                    "The result is not free. Factor-neutral hedging trades more and therefore pays more hedge cost. That cost is visible in the attribution rather than smoothed away.",
                ],
            ),
            (
                "Assumptions",
                [
                    "The lab uses single-curve discounting, no jumps, fixed beta, perfectly observed mids, no funding asymmetry, no Bermudan exercise, and simulated rather than real market data.",
                    "Those assumptions make the project testable and honest. They also limit how much the absolute P&L levels should be read as forecastable trading economics.",
                ],
            ),
            (
                "Extensions",
                [
                    "Natural extensions include a real-data overlay using exchange or broker marks, rough-vol factor dynamics, multi-curve discounting, cap-swaption joint calibration, Bermudan exposure via LSM, strategic IDB quoting, and an RL layer on top of the factor state.",
                ],
            ),
            (
                "References",
                [
                    "Andersen and Piterbarg (2010); Hagan, Kumar, Lesniewski, and Woodward (2002); Rebonato (2002); Bergomi (2016); Bartlett (2006); Avellaneda and Stoikov (2008); Gatheral (2006).",
                ],
            ),
        ]
        page_no = 2
        for title, paragraphs in pages[:2]:
            fig, ax = _new_page(title, page_no)
            _paragraphs(ax, paragraphs)
            if title == "Executive Summary":
                image = mpimg.imread("outputs/figures/headline_pnl_dist.png")
                ax.imshow(image, extent=[0.08, 0.92, 0.20, 0.48], aspect="auto")
            pdf.savefig(fig)
            plt.close(fig)
            page_no += 1

        figure_pages = [
            ("Architecture Diagram", "architecture.png", "Simulator to calibration to analytics to market-making loop to attribution."),
            ("ATM Cube Snapshots", "atm_cube_snapshots.png", "Three snapshots show the latent factor model moving the full cube, not isolated nodes."),
            ("Calibration Diagnostics", "calibration_residual_heatmaps.png", "Residual heatmaps in vol bp for per-slice and cube-constrained fits."),
            ("PCA Recovery", "pca_recovery.png", "Recovered loading shapes track the true level, slope, and curvature modes."),
            ("Representative Path", "representative_mm_path.png", "One Monte Carlo path illustrates how hedge rules diverge through time."),
            ("P&L Distributions", "headline_pnl_dist.png", "Terminal P&L distributions across all simulated market-making paths."),
            ("Attribution", "pnl_attribution.png", "Average path attribution exposes edge, theta, delta, vega, cross-vega, and hedge cost."),
            ("Quoting Heatmap", "quoting_half_spread_heatmap.png", "Half-spread rises with factor inventory in the quoted instrument direction."),
        ]
        for title, figure_name, caption in figure_pages[:1]:
            _image_page(pdf, title, Path("outputs/figures") / figure_name, page_no, caption)
            page_no += 1

        for title, paragraphs in pages[2:7]:
            fig, ax = _new_page(title, page_no)
            _paragraphs(ax, paragraphs)
            pdf.savefig(fig)
            plt.close(fig)
            page_no += 1

        for title, figure_name, caption in figure_pages[1:4]:
            _image_page(pdf, title, Path("outputs/figures") / figure_name, page_no, caption)
            page_no += 1

        fig, ax = _new_page("Arbitrage Violation Counts", page_no)
        y = 0.82
        for line in arb.to_string(index=False).splitlines():
            ax.text(0.10, y, line, family="monospace", fontsize=10, color=PALETTE.charcoal)
            y -= 0.035
        pdf.savefig(fig)
        plt.close(fig)
        page_no += 1

        for title, paragraphs in pages[7:12]:
            fig, ax = _new_page(title, page_no)
            _paragraphs(ax, paragraphs)
            pdf.savefig(fig)
            plt.close(fig)
            page_no += 1

        for title, figure_name, caption in figure_pages[4:]:
            _image_page(pdf, title, Path("outputs/figures") / figure_name, page_no, caption)
            page_no += 1

        fig, ax = _new_page("Tail-Risk Table", page_no)
        lines = _table_text(summary, ["strategy", "mean", "std", "var_5", "cvar_5", "sharpe"])
        y = 0.82
        for line in lines:
            ax.text(0.08, y, line, family="monospace", fontsize=9.5, color=PALETTE.charcoal)
            y -= 0.035
        pdf.savefig(fig)
        plt.close(fig)
        page_no += 1

        for title, paragraphs in pages[12:]:
            fig, ax = _new_page(title, page_no)
            _paragraphs(ax, paragraphs)
            pdf.savefig(fig)
            plt.close(fig)
            page_no += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PDF report.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    ensure_dirs()
    _write_latex_sources(settings)
    if not _try_xelatex():
        _fallback_pdf(settings)
    size = (REPORT_DIR / "report.pdf").stat().st_size
    print(f"wrote {REPORT_DIR / 'report.pdf'} ({size} bytes)")


if __name__ == "__main__":
    main()
