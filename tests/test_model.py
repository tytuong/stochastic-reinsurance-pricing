import math

import numpy as np
import pytest
from scipy import integrate, stats

from reinsurance_pricing import (
    ModelConfig,
    PortfolioAssumptions,
    TreatyTerms,
    analytic_moments,
    load_config,
    price_layer,
    run_simulation,
    simulate_losses,
)
from reinsurance_pricing.analytics import lognormal_lev, lognormal_lev2
from reinsurance_pricing.simulation import layer_loss_per_claim

BASE = ModelConfig()
UNLIMITED = BASE.with_updates(treaty={"reinstatements": None})


@pytest.fixture(scope="module")
def base_sim():
    return run_simulation(UNLIMITED)


# --------------------------------------------------------------------------- config


def test_lognormal_parameters_reproduce_mean_and_cv():
    p = PortfolioAssumptions(claim_frequency=10, severity_mean=750_000, severity_cv=1.8)
    dist = stats.lognorm(s=p.lognormal_sigma, scale=math.exp(p.lognormal_mu))
    assert dist.mean() == pytest.approx(750_000)
    assert dist.std() / dist.mean() == pytest.approx(1.8)


def test_load_config_matches_defaults():
    cfg = load_config("config/base_case.toml")
    assert cfg == BASE


@pytest.mark.parametrize(
    "kwargs",
    [{"claim_frequency": 0}, {"severity_mean": -1}, {"severity_cv": 0}],
)
def test_invalid_portfolio_rejected(kwargs):
    with pytest.raises(ValueError):
        PortfolioAssumptions(**kwargs)


def test_aggregate_limit_from_reinstatements():
    assert TreatyTerms(limit=5e6, reinstatements=1).aggregate_limit == 10e6
    assert TreatyTerms(limit=5e6, reinstatements=0).aggregate_limit == 5e6
    assert math.isinf(TreatyTerms(limit=5e6, reinstatements=None).aggregate_limit)


# ------------------------------------------------------------------- layer mechanics


def test_layer_loss_per_claim():
    t = TreatyTerms(retention=5e6, limit=5e6)
    x = np.array([1e6, 5e6, 6e6, 10e6, 25e6])
    np.testing.assert_allclose(layer_loss_per_claim(x, t), [0, 0, 1e6, 5e6, 5e6])


# ------------------------------------------------------------------------ simulation


def test_simulation_is_reproducible():
    a = run_simulation(BASE.with_updates(simulation={"n_years": 2_000}))
    b = run_simulation(BASE.with_updates(simulation={"n_years": 2_000}))
    np.testing.assert_array_equal(a.layer_annual, b.layer_annual)


def test_simulation_internal_consistency():
    t = TreatyTerms(retention=1e6, limit=2e6, reinstatements=1)
    sim = simulate_losses(PortfolioAssumptions(5, 500_000, 2.0), t, n_years=5_000, seed=7)

    assert sim.claim_counts.sum() == sim.severities.size
    assert np.isclose(sim.gross_annual.sum(), sim.severities.sum())
    assert np.all(sim.layer_annual <= t.aggregate_limit + 1e-6)
    assert np.all(sim.layer_annual <= sim.layer_annual_uncapped + 1e-6)
    assert np.all(sim.layer_annual_uncapped <= sim.gross_annual + 1e-6)

    # largest loss per year, checked the slow way
    offsets = np.concatenate(([0], np.cumsum(sim.claim_counts)))
    for j in range(200):
        year = sim.severities[offsets[j] : offsets[j + 1]]
        assert sim.largest_loss[j] == (year.max() if year.size else 0.0)
        assert sim.layer_hits[j] == int((year > t.retention).sum())


def test_frequency_and_severity_match_inputs(base_sim):
    p = UNLIMITED.portfolio
    assert base_sim.claim_counts.mean() == pytest.approx(p.claim_frequency, rel=0.01)
    assert base_sim.claim_counts.var() == pytest.approx(p.claim_frequency, rel=0.03)  # Poisson: var = mean
    assert base_sim.severities.mean() == pytest.approx(p.severity_mean, rel=0.03)


# --------------------------------------------------------------- closed-form checks


def test_limited_expected_value_against_numerical_integration():
    mu, sigma = 12.5, 1.4
    dist = stats.lognorm(s=sigma, scale=math.exp(mu))
    for d in (1e5, 5e5, 3e6):
        lev = integrate.quad(dist.sf, 0, d, limit=200)[0]                        # E[min(X,d)] = int S(x) dx
        lev2 = integrate.quad(lambda x: 2 * x * dist.sf(x), 0, d, limit=200)[0]  # E[min(X,d)^2]
        assert lognormal_lev(d, mu, sigma) == pytest.approx(lev, rel=1e-6)
        assert lognormal_lev2(d, mu, sigma) == pytest.approx(lev2, rel=1e-6)
    assert lognormal_lev(math.inf, mu, sigma) == pytest.approx(dist.mean())


def test_monte_carlo_agrees_with_closed_form(base_sim):
    exact = analytic_moments(UNLIMITED.portfolio, UNLIMITED.treaty)
    z = base_sim.layer_annual
    se = z.std(ddof=1) / math.sqrt(z.size)

    assert abs(z.mean() - exact.expected_layer_loss) < 4 * se
    assert z.std(ddof=1) == pytest.approx(exact.std_layer_loss, rel=0.03)
    assert (z > 0).mean() == pytest.approx(exact.prob_layer_hit, abs=0.006)
    assert base_sim.layer_hits.mean() == pytest.approx(exact.expected_layer_hits, rel=0.02)


# --------------------------------------------------------------------------- pricing


def test_premium_build_up_identities(base_sim):
    r = price_layer(base_sim, UNLIMITED)
    pr = UNLIMITED.pricing
    assert r.capital == pytest.approx(r.layer.var - r.expected_layer_loss)
    assert r.risk_load == pytest.approx(pr.cost_of_capital * r.capital)
    assert r.risk_adjusted_premium == pytest.approx(r.expected_layer_loss + r.risk_load)
    assert r.gross_technical_premium == pytest.approx(
        r.risk_adjusted_premium / (1 - pr.expense_ratio - pr.brokerage)
    )
    assert r.expected_reinstatement_factor == 1.0
    assert r.upfront_premium == pytest.approx(r.gross_technical_premium)
    assert r.layer.tvar >= r.layer.var >= r.expected_layer_loss


def test_paid_reinstatements_reduce_upfront_premium():
    cfg = BASE.with_updates(simulation={"n_years": 20_000})
    r = price_layer(run_simulation(cfg), cfg)
    assert r.expected_reinstatement_factor > 1.0
    assert r.upfront_premium < r.gross_technical_premium
    assert r.layer.var <= cfg.treaty.aggregate_limit


def test_higher_cost_of_capital_means_higher_premium():
    cfg = UNLIMITED.with_updates(simulation={"n_years": 10_000})
    sim = run_simulation(cfg)
    low = price_layer(sim, cfg.with_updates(pricing={"cost_of_capital": 0.06}))
    high = price_layer(sim, cfg.with_updates(pricing={"cost_of_capital": 0.14}))
    assert high.risk_adjusted_premium > low.risk_adjusted_premium
    assert high.expected_layer_loss == low.expected_layer_loss


def test_unreachable_layer_prices_to_zero():
    cfg = UNLIMITED.with_updates(
        treaty={"retention": 1e15, "limit": 1e6}, simulation={"n_years": 2_000}
    )
    r = price_layer(run_simulation(cfg), cfg)
    assert r.expected_layer_loss == 0
    assert r.upfront_premium == 0
