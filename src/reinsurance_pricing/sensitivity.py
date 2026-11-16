"""Sensitivity testing and layer comparisons.

Every scenario is re-simulated with the same random seed (common random
numbers), so differences between scenarios reflect the change in assumption
rather than simulation noise.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .config import ModelConfig
from .pricing import price_layer
from .simulation import run_simulation

# (section, field, label) for each assumption that gets shocked
SENSITIVITY_DRIVERS: list[tuple[str, str, str]] = [
    ("portfolio", "claim_frequency", "Claim frequency (λ)"),
    ("portfolio", "severity_mean", "Severity mean"),
    ("portfolio", "severity_cv", "Severity CV (tail heaviness)"),
    ("pricing", "cost_of_capital", "Cost of capital"),
    ("treaty", "retention", "Retention"),
]


@dataclass(frozen=True)
class SensitivityRow:
    driver: str
    shock: float
    assumption_value: float
    expected_layer_loss: float
    risk_adjusted_premium: float
    upfront_premium: float
    change_vs_base: float  # % change in risk-adjusted premium


def run_sensitivities(config: ModelConfig, shock: float = 0.20) -> list[SensitivityRow]:
    """Shock each driver up and down by ``shock`` (relative), one at a time."""
    base = price_layer(run_simulation(config), config)
    rows: list[SensitivityRow] = []
    for section, fld, label in SENSITIVITY_DRIVERS:
        base_value = getattr(getattr(config, section), fld)
        for direction in (-1, 1):
            value = base_value * (1 + direction * shock)
            cfg = config.with_updates(**{section: {fld: value}})
            res = price_layer(run_simulation(cfg), cfg)
            rows.append(
                SensitivityRow(
                    driver=label,
                    shock=direction * shock,
                    assumption_value=value,
                    expected_layer_loss=res.expected_layer_loss,
                    risk_adjusted_premium=res.risk_adjusted_premium,
                    upfront_premium=res.upfront_premium,
                    change_vs_base=res.risk_adjusted_premium / base.risk_adjusted_premium - 1,
                )
            )
    return rows


@dataclass(frozen=True)
class LayerRow:
    layer: str
    retention: float
    limit: float
    expected_layer_loss: float
    prob_layer_hit: float
    risk_adjusted_premium: float
    upfront_premium: float
    rate_on_line: float
    expected_loss_ratio: float


def price_tower(config: ModelConfig, layers: list[tuple[float, float]]) -> list[LayerRow]:
    """Price a programme of stacked layers, given as (retention, limit) pairs."""
    rows = []
    for retention, limit in layers:
        cfg = config.with_updates(treaty={"retention": retention, "limit": limit})
        res = price_layer(run_simulation(cfg), cfg)
        rows.append(
            LayerRow(
                layer=f"£{limit / 1e6:g}m xs £{retention / 1e6:g}m",
                retention=retention,
                limit=limit,
                expected_layer_loss=res.expected_layer_loss,
                prob_layer_hit=res.prob_layer_hit,
                risk_adjusted_premium=res.risk_adjusted_premium,
                upfront_premium=res.upfront_premium,
                rate_on_line=res.rate_on_line,
                expected_loss_ratio=res.expected_loss_ratio,
            )
        )
    return rows


def write_csv(rows: list, path: str | Path) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].__dataclass_fields__)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: getattr(row, k) for k in fieldnames})
