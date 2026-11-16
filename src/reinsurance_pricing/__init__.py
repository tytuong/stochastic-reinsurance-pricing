"""Stochastic pricing of an Excess of Loss reinsurance treaty.

Quick start
-----------
>>> from reinsurance_pricing import ModelConfig, run_simulation, price_layer
>>> config = ModelConfig()
>>> result = price_layer(run_simulation(config), config)
>>> round(result.rate_on_line, 2)  # doctest: +SKIP
"""

from .analytics import analytic_moments
from .config import (
    ModelConfig,
    PortfolioAssumptions,
    PricingAssumptions,
    SimulationSettings,
    TreatyTerms,
    load_config,
)
from .pricing import PricingResult, format_summary, price_layer
from .simulation import SimulationResult, run_simulation, simulate_losses

__all__ = [
    "ModelConfig",
    "PortfolioAssumptions",
    "PricingAssumptions",
    "PricingResult",
    "SimulationResult",
    "SimulationSettings",
    "TreatyTerms",
    "analytic_moments",
    "format_summary",
    "load_config",
    "price_layer",
    "run_simulation",
    "simulate_losses",
]

__version__ = "1.0.0"
