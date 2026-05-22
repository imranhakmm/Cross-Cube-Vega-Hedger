"""Figure and table generation for the report and static site."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from numpy.typing import NDArray

from cxvega.analytics import recover_pca_factors
from cxvega.arb_checks import check_cube_static_arbitrage
from cxvega.calibration import (
    apply_observation_noise,
    calibrate_cube_joint,
    calibrate_cube_per_slice,
)
from cxvega.config import Settings
from cxvega.market_maker import MarketMakerResult, factor_covariance, run_market_maker_simulation
from cxvega.plotting import PALETTE, apply_style, strategy_colours
from cxvega.quoting import spread_heatmap
from cxvega.rng import rng_for
from cxvega.simulator import CubePath, simulate_cube_path


def ensure_dirs() -> None:
    """Create all generated-output directories."""

    for folder in [
        Path("outputs/figures"),
        Path("outputs/tables"),
        Path("outputs/simulations"),
        Path("docs/report/figures"),
        Path("docs/site/assets"),
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def save_cube_path(path: CubePath, destination: Path) -> None:
    """Persist the simulated cube path as a compact NPZ archive."""

    np.savez_compressed(
        destination,
        times=path.times,
        expiries=path.expiries,
        tenors=path.tenors,
        strike_offsets_bps=path.strike_offsets_bps,
        forwards=path.forwards,
        annuities=path.annuities,
        strikes=path.strikes,
        loadings=path.loadings,
        factors=path.factors,
        atm_vol=path.atm_vol,
        alpha=path.alpha,
        rho=path.rho,
        nu=path.nu,
        vols=path.vols,
        beta=np.array([path.beta], dtype=float),
    )


def generate_cube(settings: Settings, steps: int | None = None) -> CubePath:
    """Simulate and save the cube path used by downstream tasks."""

    ensure_dirs()
    path = simulate_cube_path(settings, n_steps=steps)
    save_cube_path(path, Path("outputs/simulations/cube_path.npz"))
    return path


def generate_market_results(settings: Settings, path: CubePath) -> MarketMakerResult:
    """Run the market-maker simulation and save tabular artefacts."""

    ensure_dirs()
    result = run_market_maker_simulation(settings, path)
    result.path_pnl.to_csv("outputs/simulations/path_pnl.csv", index=False)
    result.representative_path.to_csv("outputs/simulations/representative_path.csv", index=False)
    result.attribution.to_csv("outputs/tables/attribution.csv", index=False)
    result.summary.to_csv("outputs/tables/results_summary.csv", index=False)
    np.save("outputs/simulations/factor_covariance.npy", result.factor_covariance)
    return result


def load_or_run_market_results(settings: Settings, path: CubePath) -> MarketMakerResult:
    """Load saved market-maker outputs, or regenerate them when absent."""

    required = [
        Path("outputs/simulations/path_pnl.csv"),
        Path("outputs/simulations/representative_path.csv"),
        Path("outputs/tables/attribution.csv"),
        Path("outputs/tables/results_summary.csv"),
    ]
    if not all(file.exists() for file in required):
        return generate_market_results(settings, path)
    return MarketMakerResult(
        path_pnl=pd.read_csv("outputs/simulations/path_pnl.csv"),
        daily_pnl=pd.DataFrame(),
        attribution=pd.read_csv("outputs/tables/attribution.csv"),
        representative_path=pd.read_csv("outputs/simulations/representative_path.csv"),
        summary=pd.read_csv("outputs/tables/results_summary.csv"),
        factor_covariance=np.load("outputs/simulations/factor_covariance.npy")
        if Path("outputs/simulations/factor_covariance.npy").exists()
        else factor_covariance(settings),
    )


def _save_figure(fig: Figure, name: str) -> Path:
    out = Path("outputs/figures") / name
    report_out = Path("docs/report/figures") / name
    fig.savefig(out)
    fig.savefig(report_out)
    plt.close(fig)
    return out


def architecture_figure() -> Path:
    """Draw the project architecture diagram used by the report and site."""

    apply_style()
    fig, ax = plt.subplots(figsize=(11, 3.4), constrained_layout=True)
    ax.axis("off")
    labels = [
        "Cube\nsimulator",
        "SABR\ncalibration",
        "PCA + factor\nanalytics",
        "Market-maker\nloop",
        "P&L\nattribution",
    ]
    xs = np.linspace(0.08, 0.92, len(labels))
    for index, (x_pos, label) in enumerate(zip(xs, labels, strict=True)):
        rect = FancyBboxPatch(
            (x_pos - 0.075, 0.38),
            0.15,
            0.28,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            facecolor="#eeeeee",
            edgecolor=PALETTE.graphite,
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x_pos, 0.52, label, ha="center", va="center", color=PALETTE.charcoal)
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(xs[index + 1] - 0.085, 0.52),
                xytext=(x_pos + 0.085, 0.52),
                arrowprops={"arrowstyle": "->", "color": PALETTE.oxblood, "lw": 1.4},
            )
    ax.text(
        0.50,
        0.18,
        "One seed rebuilds every path, table, figure, HTML page, and PDF report.",
        ha="center",
        va="center",
        color=PALETTE.graphite,
    )
    out = _save_figure(fig, "architecture.png")
    fig_svg, ax_svg = plt.subplots(figsize=(11, 3.4), constrained_layout=True)
    ax_svg.axis("off")
    for index, (x_pos, label) in enumerate(zip(xs, labels, strict=True)):
        rect = FancyBboxPatch(
            (x_pos - 0.075, 0.38),
            0.15,
            0.28,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            facecolor="#eeeeee",
            edgecolor=PALETTE.graphite,
            linewidth=1.2,
        )
        ax_svg.add_patch(rect)
        ax_svg.text(x_pos, 0.52, label, ha="center", va="center", color=PALETTE.charcoal)
        if index < len(labels) - 1:
            ax_svg.annotate(
                "",
                xy=(xs[index + 1] - 0.085, 0.52),
                xytext=(x_pos + 0.085, 0.52),
                arrowprops={"arrowstyle": "->", "color": PALETTE.oxblood, "lw": 1.4},
            )
    fig_svg.savefig("docs/architecture.svg", metadata={"Date": None})
    plt.close(fig_svg)
    return out


def plot_atm_cube_snapshots(path: CubePath) -> Path:
    """Plot three simulated ATM vol snapshots as heatmap small multiples."""

    apply_style()
    indices = [0, path.atm_vol.shape[0] // 2, path.atm_vol.shape[0] - 1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    vmin = float(np.min(path.atm_vol[indices]) * 100.0)
    vmax = float(np.max(path.atm_vol[indices]) * 100.0)
    for ax, idx in zip(axes, indices, strict=True):
        image = ax.imshow(path.atm_vol[idx] * 100.0, cmap="Greys", vmin=vmin, vmax=vmax)
        ax.set_title(f"t = {path.times[idx]:.2f}y")
        ax.set_xticks(range(path.n_tenors), path.tenor_labels)
        ax.set_yticks(range(path.n_expiries), path.expiry_labels)
        ax.set_xlabel("Swap tenor")
        ax.set_ylabel("Option expiry")
    fig.colorbar(image, ax=axes, label="ATM Black vol (%)", fraction=0.025)
    fig.suptitle("Simulated ATM Swaption Vol Cube")
    return _save_figure(fig, "atm_cube_snapshots.png")


def calibration_diagnostics(path: CubePath, settings: Settings) -> tuple[Path, pd.DataFrame]:
    """Generate calibration residual heatmaps and arbitrage count table."""

    apply_style()
    true_market = path.vols[min(5, path.vols.shape[0] - 1)]
    market = apply_observation_noise(
        true_market,
        rng_for(settings.seed, "calibration-observation-noise"),
        settings.observation_noise.calibration_vol_log_sigma,
    )
    per = calibrate_cube_per_slice(path.forwards, path.strikes, path.expiries, market, path.beta)
    joint = calibrate_cube_joint(
        path.forwards,
        path.annuities,
        path.strikes,
        path.expiries,
        market,
        path.beta,
        smoothness_weight=settings.calibration.joint_smoothness_weight,
        butterfly_weight=settings.calibration.joint_butterfly_weight,
        calendar_weight=settings.calibration.joint_calendar_weight,
        maxiter=55,
    )
    arb_tol = 1.0e-4
    per_report = check_cube_static_arbitrage(
        path.forwards, path.annuities, path.strikes, path.expiries, per.vols, tol=arb_tol
    )
    joint_report = check_cube_static_arbitrage(
        path.forwards, path.annuities, path.strikes, path.expiries, joint.vols, tol=arb_tol
    )
    table = pd.DataFrame(
        [
            {"fit": "Per-slice", "violations": per_report.count, **per_report.by_kind()},
            {"fit": "Cube-constrained", "violations": joint_report.count, **joint_report.by_kind()},
        ]
    ).fillna(0)
    for column in table.columns:
        if column != "fit":
            table[column] = table[column].astype(int)
    table.to_csv("outputs/tables/arb_violation_counts.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, title, residuals in [
        (axes[0], "Per-slice residual RMS", per.residuals),
        (axes[1], "Cube-constrained residual RMS", joint.residuals),
    ]:
        heat = np.sqrt(np.mean(residuals**2, axis=2)) * 1.0e4
        image = ax.imshow(heat, cmap="Greys")
        ax.set_title(title)
        ax.set_xticks(range(path.n_tenors), path.tenor_labels)
        ax.set_yticks(range(path.n_expiries), path.expiry_labels)
        ax.set_xlabel("Swap tenor")
        ax.set_ylabel("Option expiry")
        fig.colorbar(image, ax=ax, label="vol bp")
    fig.savefig("outputs/figures/calibration_residuals.png")
    fig.savefig("docs/report/figures/calibration_residuals.png")
    return _save_figure(fig, "calibration_residual_heatmaps.png"), table


def pca_figure(path: CubePath) -> Path:
    """Plot PCA explained variance and recovered factor-loading shapes."""

    apply_style()
    recovery = recover_pca_factors(path)
    x = np.arange(path.n_expiries)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(range(1, 4), recovery.explained_variance[:3] * 100.0, color=PALETTE.oxblood)
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("PCA Variance")
    colours = [PALETTE.graphite, PALETTE.pine, PALETTE.oxblood]
    for k, label in enumerate(["Level", "Slope", "Curvature"]):
        true_shape = np.mean(recovery.true_loadings[:, :, k], axis=1)
        recovered_shape = np.mean(recovery.recovered_loadings[:, :, k], axis=1)
        true_shape = true_shape / max(float(np.linalg.norm(true_shape)), 1.0e-12)
        recovered_shape = recovered_shape / max(float(np.linalg.norm(recovered_shape)), 1.0e-12)
        axes[1].plot(
            x,
            recovered_shape,
            color=colours[k],
            linewidth=1.5,
            alpha=0.58,
        )
        axes[1].plot(
            x,
            true_shape,
            color=colours[k],
            linestyle="--",
            label=label,
            linewidth=1.9,
            alpha=0.95,
        )
    axes[1].set_xticks(x, path.expiry_labels)
    axes[1].set_title("True vs Recovered Loading Shapes")
    axes[1].set_xlabel("Option expiry")
    axes[1].legend()
    pd.DataFrame(
        {
            "factor": ["level", "slope", "curvature"],
            "loading_similarity": recovery.loading_similarity,
            "explained_variance": recovery.explained_variance[:3],
        }
    ).to_csv("outputs/tables/pca_recovery.csv", index=False)
    return _save_figure(fig, "pca_recovery.png")


def representative_path_figure(result: MarketMakerResult) -> Path:
    """Plot cumulative P&L for one representative market-making path."""

    apply_style()
    colours = strategy_colours()
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for strategy, group in result.representative_path.groupby("strategy"):
        ax.plot(
            group["day"],
            group["cumulative_pnl"] / 1.0e6,
            label=strategy,
            color=colours[strategy],
        )
    ax.axhline(0.0, color=PALETTE.silver, linewidth=0.8)
    ax.set_title("Representative Market-Maker Path")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Cumulative P&L ($mm)")
    ax.legend(ncols=2)
    return _save_figure(fig, "representative_mm_path.png")


def headline_pnl_distribution_figure(result: MarketMakerResult) -> Path:
    """Plot same-axis terminal P&L distributions for all strategies."""

    apply_style()
    colours = strategy_colours()
    strategies = list(colours)
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    all_values = result.path_pnl["pnl"].to_numpy(dtype=float) / 1.0e6
    x_min = float(np.floor(np.min(all_values) / 25.0) * 25.0)
    x_max = float(np.ceil(np.max(all_values) / 25.0) * 25.0)
    bins = np.linspace(x_min, x_max, 58).tolist()
    for strategy in strategies:
        values = result.path_pnl.loc[
            result.path_pnl["strategy"] == strategy,
            "pnl",
        ].to_numpy(dtype=float) / 1.0e6
        ax.hist(
            values,
            bins=bins,
            density=True,
            color=colours[strategy],
            alpha=0.34,
            label=strategy,
        )
        ax.axvline(
            float(np.mean(values)),
            color=colours[strategy],
            linestyle="--",
            linewidth=1.35,
        )
    ax.axvline(0.0, color=PALETTE.silver, linewidth=0.9)
    ax.set_xlim(x_min, x_max)
    ax.set_title("Terminal P&L Distribution: Same-Axis Overlay")
    ax.set_xlabel("Terminal P&L ($mm)")
    ax.set_ylabel("Density")
    ax.legend(ncols=2)
    return _save_figure(fig, "headline_pnl_dist.png")


def pnl_distribution_small_multiples_figure(result: MarketMakerResult) -> Path:
    """Plot side-by-side terminal P&L distributions for all strategies."""

    apply_style()
    colours = strategy_colours()
    strategies = list(colours)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True, constrained_layout=True)
    all_values = result.path_pnl["pnl"].to_numpy(dtype=float) / 1.0e6
    x_min = float(np.floor(np.min(all_values) / 25.0) * 25.0)
    x_max = float(np.ceil(np.max(all_values) / 25.0) * 25.0)
    for ax, strategy in zip(axes, strategies, strict=True):
        values = result.path_pnl.loc[
            result.path_pnl["strategy"] == strategy,
            "pnl",
        ].to_numpy(dtype=float)
        ax.hist(values / 1.0e6, bins=34, color=colours[strategy], alpha=0.86)
        ax.axvline(np.mean(values) / 1.0e6, color=PALETTE.charcoal, linewidth=1.2)
        ax.set_xlim(x_min, x_max)
        ax.set_title(strategy)
        ax.set_xlabel("$mm")
    axes[0].set_ylabel("Path count")
    fig.suptitle("Terminal P&L Distribution Across Monte Carlo Paths")
    return _save_figure(fig, "pnl_dist_small_multiples.png")


def attribution_figure(result: MarketMakerResult) -> Path:
    """Plot average path-level P&L attribution as stacked bars."""

    apply_style()
    components = ["edge", "theta", "delta", "vega_first_order", "vega_cross", "hedge_cost"]
    labels = ["Edge", "Theta", "Delta", "Vega 1st", "Cross-vega", "Hedge cost"]
    colours = [
        PALETTE.pine,
        PALETTE.silver,
        "#8c8c8c",
        PALETTE.steel,
        PALETTE.oxblood,
        PALETTE.graphite,
    ]
    table = result.attribution.set_index("strategy").loc[list(strategy_colours())]
    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    positive_bottom = np.zeros(len(table), dtype=float)
    negative_bottom = np.zeros(len(table), dtype=float)
    for component, label, colour in zip(components, labels, colours, strict=True):
        values = cast(NDArray[np.float64], table[component].to_numpy(dtype=float) / 1.0e6)
        bottom = np.where(values >= 0.0, positive_bottom, negative_bottom)
        ax.bar(table.index, values, bottom=bottom, label=label, color=colour)
        positive_bottom += np.where(values >= 0.0, values, 0.0)
        negative_bottom += np.where(values < 0.0, values, 0.0)
    ax.axhline(0.0, color=PALETTE.charcoal, linewidth=0.8)
    ax.set_ylabel("Average path P&L ($mm)")
    ax.set_title("P&L Attribution by Strategy")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(ncols=3)
    return _save_figure(fig, "pnl_attribution.png")


def quoting_heatmap_figure(settings: Settings, cov: NDArray[np.float64]) -> Path:
    """Plot quoted half-spread versus level and skew inventory exposure."""

    apply_style()
    grid = np.linspace(-3.0e8, 3.0e8, 61)
    loading = np.array([1.0, 0.0, 0.0, 0.35], dtype=float)
    heat = spread_heatmap(
        settings.market_maker.base_half_spread_vol_bps,
        cov,
        loading,
        settings.market_maker.risk_aversion,
        grid,
    )
    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    image = ax.imshow(
        heat,
        cmap="Greys",
        origin="lower",
        extent=(
            float(grid[0] / 1.0e6),
            float(grid[-1] / 1.0e6),
            float(grid[0] / 1.0e6),
            float(grid[-1] / 1.0e6),
        ),
        aspect="auto",
    )
    ax.set_xlabel("Skew factor exposure ($mm vega units)")
    ax.set_ylabel("Level factor exposure ($mm vega units)")
    ax.set_title("Inventory-Inflated Quoting Half-Spread")
    fig.colorbar(image, ax=ax, label="half-spread (vol bp)")
    fig.savefig("outputs/figures/quoting_heatmap.png")
    fig.savefig("docs/report/figures/quoting_heatmap.png")
    return _save_figure(fig, "quoting_half_spread_heatmap.png")


def generate_all_figures(settings: Settings, path: CubePath | None = None) -> list[Path]:
    """Generate every figure and table consumed by the report and static site."""

    ensure_dirs()
    cube_path = path or generate_cube(settings)
    result = load_or_run_market_results(settings, cube_path)
    outputs: list[Path] = [
        architecture_figure(),
        plot_atm_cube_snapshots(cube_path),
        calibration_diagnostics(cube_path, settings)[0],
        pca_figure(cube_path),
        representative_path_figure(result),
        headline_pnl_distribution_figure(result),
        pnl_distribution_small_multiples_figure(result),
        attribution_figure(result),
        quoting_heatmap_figure(settings, result.factor_covariance),
    ]
    return outputs
