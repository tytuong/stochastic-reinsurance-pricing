"""Closed-form results used to validate the Monte Carlo engine.

For a compound Poisson model with Lognormal severities, the first two moments
of the layer loss can be written down exactly (when there is no annual
aggregate limit). If the simulation is correct, its estimates must agree with
these values to within Monte Carlo error. This is the main check that the
model is doing what it claims to.

Key formulas (Phi = standard normal CDF, u = R + L)
---------------------------------------------------
Limited expected value of a Lognormal(mu, sigma):

    E[min(X, d)]   = exp(mu + sigma^2/2) * Phi((ln d - mu - sigma^2) / sigma)
                     + d * (1 - Phi((ln d - mu) / sigma))

    E[min(X, d)^2] = exp(2mu + 2sigma^2) * Phi((ln d - mu - 2sigma^2) / sigma)
                     + d^2 * (1 - Phi((ln d - mu) / sigma))

Per-claim layer loss Y = min(X, u) - min(X, R):

    E[Y]   = E[min(X, u)] - E[min(X, R)]
    E[Y^2] = E[min(X, u)^2] - E[min(X, R)^2] - 2R * E[Y]

Compound Poisson annual layer loss Z = Y_1 + ... + Y_N:

    E[Z]   = lambda * E[Y]
    Var[Z] = lambda * E[Y^2]
    P(Z=0) = exp(-lambda * P(X > R))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

from .config import PortfolioAssumptions, TreatyTerms


def lognormal_lev(d: float, mu: float, sigma: float) -> float:
    """Limited expected value E[min(X, d)] for X ~ Lognormal(mu, sigma)."""
    if d <= 0:
        return 0.0
    if math.isinf(d):
        return math.exp(mu + 0.5 * sigma**2)
    ln_d = math.log(d)
    return (
        math.exp(mu + 0.5 * sigma**2) * norm.cdf((ln_d - mu - sigma**2) / sigma)
        + d * norm.sf((ln_d - mu) / sigma)
    )


def lognormal_lev2(d: float, mu: float, sigma: float) -> float:
    """Second limited moment E[min(X, d)^2] for X ~ Lognormal(mu, sigma)."""
    if d <= 0:
        return 0.0
    if math.isinf(d):
        return math.exp(2 * mu + 2 * sigma**2)
    ln_d = math.log(d)
    return (
        math.exp(2 * mu + 2 * sigma**2) * norm.cdf((ln_d - mu - 2 * sigma**2) / sigma)
        + d**2 * norm.sf((ln_d - mu) / sigma)
    )


def lognormal_survival(x: float, mu: float, sigma: float) -> float:
    """P(X > x)."""
    if x <= 0:
        return 1.0
    return float(norm.sf((math.log(x) - mu) / sigma))


@dataclass(frozen=True)
class AnalyticMoments:
    expected_layer_loss: float      # E[Z]
    std_layer_loss: float           # sqrt(Var[Z])
    prob_layer_hit: float           # P(Z > 0)
    expected_layer_hits: float      # E[number of losses hitting the layer]
    expected_gross_loss: float      # E[S]
    std_gross_loss: float           # sqrt(Var[S])


def analytic_moments(portfolio: PortfolioAssumptions, treaty: TreatyTerms) -> AnalyticMoments:
    """Exact moments of the layer and gross annual loss (ignores any aggregate limit)."""
    lam = portfolio.claim_frequency
    mu, sigma = portfolio.lognormal_mu, portfolio.lognormal_sigma
    r, u = treaty.retention, treaty.exhaustion_point

    ey = lognormal_lev(u, mu, sigma) - lognormal_lev(r, mu, sigma)
    ey2 = lognormal_lev2(u, mu, sigma) - lognormal_lev2(r, mu, sigma) - 2 * r * ey
    p_exceed = lognormal_survival(r, mu, sigma)

    ex2 = math.exp(2 * mu + 2 * sigma**2)

    return AnalyticMoments(
        expected_layer_loss=float(lam * ey),
        std_layer_loss=float(math.sqrt(lam * ey2)),
        prob_layer_hit=1.0 - math.exp(-lam * p_exceed),
        expected_layer_hits=lam * p_exceed,
        expected_gross_loss=lam * portfolio.severity_mean,
        std_gross_loss=math.sqrt(lam * ex2),
    )
