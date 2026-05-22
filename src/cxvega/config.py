"""Typed configuration loading for the Cross-Cube Vega lab."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field


class CalendarConfig(BaseModel):
    """Trading calendar controls for deterministic simulations."""

    model_config = ConfigDict(frozen=True)

    steps_per_year: int = 252
    simulation_years: float = 5.0
    mm_days: int = 252
    n_paths: int = 1000


class CubeConfig(BaseModel):
    """Swaption cube grid and SABR conventions."""

    model_config = ConfigDict(frozen=True)

    expiries_years: list[float]
    expiry_labels: list[str]
    tenors_years: list[float]
    tenor_labels: list[str]
    strike_offsets_bps: list[float]
    beta: float = 0.5


class RatesConfig(BaseModel):
    """Simple single-curve rate environment."""

    model_config = ConfigDict(frozen=True)

    flat_discount_rate: float = 0.035
    forward_level: float = 0.0325
    forward_slope: float = 0.002
    forward_curvature: float = -0.001


class SimulatorConfig(BaseModel):
    """Parameters for the latent cube dynamics."""

    model_config = ConfigDict(frozen=True)

    ou_kappa: list[float]
    ou_vol: list[float]
    ou_corr: list[list[float]]
    rho_kappa: float
    rho_vol: float
    rho_mean: float
    nu_kappa: float
    nu_vol: float
    nu_mean: float
    rho_level_corr: float
    nu_level_corr: float


class CalibrationConfig(BaseModel):
    """Weights for the cube-constrained SABR calibration objective."""

    model_config = ConfigDict(frozen=True)

    joint_smoothness_weight: float = 0.035
    joint_butterfly_weight: float = 0.50
    joint_calendar_weight: float = 0.25


class MarketMakerConfig(BaseModel):
    """Controls for the market-making experiment."""

    model_config = ConfigDict(frozen=True)

    n_paths: int = 1000
    days: int = 252
    daily_arrival_rate: float = 7.0
    notional_mean: float = 8.0e7
    notional_sigma: float = 0.65
    base_half_spread_vol_bps: float = 0.75
    hedge_cost_vol_bps: float = 0.35
    risk_aversion: float = 0.85
    shock_scale: float = 0.85
    liquid_expiry_indices: list[int] = Field(default_factory=list)
    liquid_tenor_indices: list[int] = Field(default_factory=list)


class ReportConfig(BaseModel):
    """Output labelling for generated reports."""

    model_config = ConfigDict(frozen=True)

    run_label: str = "baseline"


class Settings(BaseModel):
    """Top-level immutable project settings."""

    model_config = ConfigDict(frozen=True)

    seed: int
    calendar: CalendarConfig
    cube: CubeConfig
    rates: RatesConfig
    simulator: SimulatorConfig
    calibration: CalibrationConfig
    market_maker: MarketMakerConfig
    report: ReportConfig


T = TypeVar("T", bound=dict[str, Any])


def _deep_update(base: T, overrides: dict[str, Any]) -> T:
    """Return a recursive merge of two dictionaries."""

    merged: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_update(cast(dict[str, Any], current), value)
        else:
            merged[key] = value
    return cast(T, merged)


def load_settings(
    path: str | Path = "configs/default.yaml",
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load project settings from YAML and optional recursive overrides."""

    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration at {cfg_path} must be a mapping.")
    data = cast(dict[str, Any], raw)
    if overrides:
        data = _deep_update(data, overrides)
    return Settings.model_validate(data)
