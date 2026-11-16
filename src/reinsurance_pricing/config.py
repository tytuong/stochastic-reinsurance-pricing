"""Model inputs, grouped into small immutable dataclasses.

Keeping the assumptions in one place (and loading them from a TOML file)
means every run is reproducible and every number in the output can be traced
back to a stated assumption.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class PortfolioAssumptions:
    """Frequency-severity assumptions for the underlying (ground-up) portfolio.

    Attributes
    ----------
    claim_frequency:
        Expected number of large losses per year (Poisson mean, lambda).
    severity_mean:
        Mean individual loss size, in GBP.
    severity_cv:
        Coefficient of variation of individual loss size (std / mean).
        Higher values give a heavier right tail.
    """

    claim_frequency: float = 25.0
    severity_mean: float = 600_000.0
    severity_cv: float = 2.5

    def __post_init__(self) -> None:
        if self.claim_frequency <= 0:
            raise ValueError("claim_frequency must be positive")
        if self.severity_mean <= 0:
            raise ValueError("severity_mean must be positive")
        if self.severity_cv <= 0:
            raise ValueError("severity_cv must be positive")

    @property
    def lognormal_sigma(self) -> float:
        """sigma such that a Lognormal(mu, sigma) has the stated CV."""
        return math.sqrt(math.log(1.0 + self.severity_cv**2))

    @property
    def lognormal_mu(self) -> float:
        """mu such that a Lognormal(mu, sigma) has the stated mean."""
        return math.log(self.severity_mean) - 0.5 * self.lognormal_sigma**2


@dataclass(frozen=True)
class TreatyTerms:
    """Per-occurrence Excess of Loss treaty: ``limit`` xs ``retention``.

    Attributes
    ----------
    retention:
        Cedant's retention (the layer attaches above this amount).
    limit:
        Maximum recovery from any single loss.
    reinstatements:
        Number of reinstatements of the limit. ``None`` means unlimited, i.e.
        no annual aggregate limit. With ``k`` reinstatements the annual
        aggregate limit is ``(k + 1) * limit``.
    reinstatement_rate:
        Reinstatement premium as a proportion of the upfront premium, charged
        pro rata to the amount of limit reinstated (1.0 = "at 100%").
    """

    retention: float = 5_000_000.0
    limit: float = 5_000_000.0
    reinstatements: int | None = 1
    reinstatement_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.retention < 0:
            raise ValueError("retention must be non-negative")
        if self.limit <= 0:
            raise ValueError("limit must be positive")
        if self.reinstatements is not None and self.reinstatements < 0:
            raise ValueError("reinstatements must be >= 0 or None")
        if self.reinstatement_rate < 0:
            raise ValueError("reinstatement_rate must be non-negative")

    @property
    def exhaustion_point(self) -> float:
        return self.retention + self.limit

    @property
    def aggregate_limit(self) -> float:
        if self.reinstatements is None:
            return math.inf
        return (self.reinstatements + 1) * self.limit

    def describe(self) -> str:
        reinst = "unlimited free reinstatements" if self.reinstatements is None else (
            f"{self.reinstatements} reinstatement(s) @ {self.reinstatement_rate:.0%}"
        )
        return f"GBP {self.limit / 1e6:,.1f}m xs GBP {self.retention / 1e6:,.1f}m ({reinst})"


@dataclass(frozen=True)
class PricingAssumptions:
    """Loadings applied on top of the expected layer loss.

    Attributes
    ----------
    cost_of_capital:
        Annual return required on the capital the reinsurer holds against the
        layer (e.g. 0.10 = 10%).
    capital_quantile:
        Confidence level for the capital requirement (0.995 = 1-in-200 years,
        consistent with Solvency II).
    expense_ratio:
        Reinsurer's internal expenses as a proportion of gross premium.
    brokerage:
        Broker commission as a proportion of gross premium.
    """

    cost_of_capital: float = 0.10
    capital_quantile: float = 0.995
    expense_ratio: float = 0.05
    brokerage: float = 0.10

    def __post_init__(self) -> None:
        if not 0 < self.capital_quantile < 1:
            raise ValueError("capital_quantile must lie in (0, 1)")
        if self.expense_ratio + self.brokerage >= 1:
            raise ValueError("expense_ratio + brokerage must be below 100%")


@dataclass(frozen=True)
class SimulationSettings:
    n_years: int = 100_000
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.n_years < 2:
            raise ValueError("n_years must be at least 2")


@dataclass(frozen=True)
class ModelConfig:
    portfolio: PortfolioAssumptions = field(default_factory=PortfolioAssumptions)
    treaty: TreatyTerms = field(default_factory=TreatyTerms)
    pricing: PricingAssumptions = field(default_factory=PricingAssumptions)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)

    def with_updates(self, **sections: dict) -> "ModelConfig":
        """Return a copy with some fields changed, e.g.
        ``cfg.with_updates(portfolio={"severity_cv": 3.0})``."""
        updated = {name: replace(getattr(self, name), **changes) for name, changes in sections.items()}
        return replace(self, **updated)


def load_config(path: str | Path) -> ModelConfig:
    """Load a :class:`ModelConfig` from a TOML file.

    Any section or key left out of the file falls back to the default.
    In TOML, unlimited reinstatements are written as ``reinstatements = -1``.
    """
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    treaty = dict(raw.get("treaty", {}))
    if treaty.get("reinstatements", None) == -1:
        treaty["reinstatements"] = None

    return ModelConfig(
        portfolio=PortfolioAssumptions(**raw.get("portfolio", {})),
        treaty=TreatyTerms(**treaty),
        pricing=PricingAssumptions(**raw.get("pricing", {})),
        simulation=SimulationSettings(**raw.get("simulation", {})),
    )
