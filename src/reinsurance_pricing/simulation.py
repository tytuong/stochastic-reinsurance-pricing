"""Monte Carlo engine: simulate annual losses to the portfolio and to the layer.

Collective risk model
---------------------
For each simulated year ``j``:

    N_j  ~ Poisson(lambda)                      number of large losses
    X_ij ~ Lognormal(mu, sigma), i = 1..N_j     individual loss sizes
    S_j  = sum_i X_ij                           gross (ground-up) annual loss

Each loss is then passed through the per-occurrence Excess of Loss layer

    Y_ij = min(max(X_ij - R, 0), L)             recovery from one loss

and the annual recoveries are capped at the aggregate limit (if any)

    Z_j  = min(sum_i Y_ij, AAL)                 layer annual loss

The whole calculation is vectorised: all claims across all years are drawn in
one NumPy call and summed back to their year with ``np.bincount``. 100,000
years (about 2.5 million individual losses in the base case) run in well
under a second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig, PortfolioAssumptions, TreatyTerms


@dataclass(frozen=True)
class SimulationResult:
    """Simulated losses. Every annual array has one entry per simulated year."""

    claim_counts: np.ndarray        # N_j
    severities: np.ndarray          # every individual loss X_ij, ordered by year
    gross_annual: np.ndarray        # S_j
    largest_loss: np.ndarray        # max_i X_ij (0 in a year with no losses)
    layer_hits: np.ndarray          # number of losses that reach the layer in year j
    layer_annual_uncapped: np.ndarray  # sum_i Y_ij before any aggregate limit
    layer_annual: np.ndarray        # Z_j, after the aggregate limit

    @property
    def n_years(self) -> int:
        return self.gross_annual.size


def layer_loss_per_claim(severities: np.ndarray, treaty: TreatyTerms) -> np.ndarray:
    """Recovery from each individual loss: ``min(max(X - R, 0), L)``."""
    return np.clip(severities - treaty.retention, 0.0, treaty.limit)


def simulate_losses(
    portfolio: PortfolioAssumptions,
    treaty: TreatyTerms,
    n_years: int,
    seed: int | None = None,
) -> SimulationResult:
    """Simulate ``n_years`` independent years of losses.

    Using the same ``seed`` gives identical results, which is also what makes
    the sensitivity analysis stable (common random numbers).
    """
    rng = np.random.default_rng(seed)

    # 1. Frequency: number of losses in each year.
    counts = rng.poisson(portfolio.claim_frequency, size=n_years)

    # 2. Severity: draw every loss for every year in a single call.
    severities = rng.lognormal(
        mean=portfolio.lognormal_mu,
        sigma=portfolio.lognormal_sigma,
        size=int(counts.sum()),
    )

    # 3. Map each loss back to the year it belongs to.
    year_of_claim = np.repeat(np.arange(n_years), counts)

    gross_annual = np.bincount(year_of_claim, weights=severities, minlength=n_years)

    # 4. Apply the per-occurrence layer to each loss, then aggregate by year.
    per_claim_recovery = layer_loss_per_claim(severities, treaty)
    layer_uncapped = np.bincount(year_of_claim, weights=per_claim_recovery, minlength=n_years)
    layer_hits = np.bincount(
        year_of_claim, weights=(severities > treaty.retention).astype(float), minlength=n_years
    ).astype(np.int64)

    # 5. Annual aggregate limit from the reinstatement provisions.
    layer_annual = np.minimum(layer_uncapped, treaty.aggregate_limit)

    # 6. Largest single loss per year (used for the occurrence exceedance curve).
    largest = np.zeros(n_years)
    has_claims = counts > 0
    if severities.size:
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        largest[has_claims] = np.maximum.reduceat(severities, starts[has_claims])

    return SimulationResult(
        claim_counts=counts,
        severities=severities,
        gross_annual=gross_annual,
        largest_loss=largest,
        layer_hits=layer_hits,
        layer_annual_uncapped=layer_uncapped,
        layer_annual=layer_annual,
    )


def run_simulation(config: ModelConfig) -> SimulationResult:
    """Convenience wrapper that reads everything from a :class:`ModelConfig`."""
    return simulate_losses(
        config.portfolio,
        config.treaty,
        n_years=config.simulation.n_years,
        seed=config.simulation.seed,
    )
