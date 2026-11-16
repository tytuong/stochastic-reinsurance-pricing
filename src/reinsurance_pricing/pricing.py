"""Turn simulated layer losses into a premium.

Building blocks
---------------
1. Pure premium (expected layer loss):      EL = E[Z]
2. Capital held against the layer:          C  = VaR_q(Z) - E[Z]
3. Cost-of-capital risk load:               RL = CoC * C
4. Risk-adjusted premium:                   RAP = EL + RL
5. Gross technical premium:                 GP = RAP / (1 - expenses - brokerage)
6. If reinstatements are paid, part of GP is collected later as
   reinstatement premium, so the upfront premium P solves

       P * (1 + c * E[min(Z_uncapped, k*L)] / L) = GP

   where c is the reinstatement rate, k the number of reinstatements and
   L the layer limit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .config import ModelConfig
from .simulation import SimulationResult


@dataclass(frozen=True)
class RiskMeasures:
    mean: float
    std: float
    var: float          # Value-at-Risk at the capital quantile
    tvar: float         # Tail Value-at-Risk (expected loss given loss >= VaR)
    quantile: float


def risk_measures(losses: np.ndarray, quantile: float) -> RiskMeasures:
    """Mean, standard deviation, VaR and TVaR of a sample of annual losses."""
    var = float(np.quantile(losses, quantile))
    tail = losses[losses >= var]
    return RiskMeasures(
        mean=float(losses.mean()),
        std=float(losses.std(ddof=1)),
        var=var,
        tvar=float(tail.mean()) if tail.size else var,
        quantile=quantile,
    )


@dataclass(frozen=True)
class PricingResult:
    # Treaty
    treaty: str
    limit: float
    retention: float

    # Loss statistics
    expected_gross_loss: float
    expected_layer_loss: float
    layer_loss_std_error: float
    layer_loss_ci_95: tuple[float, float]
    prob_layer_hit: float
    expected_layer_hits: float
    prob_limit_exhausted: float
    layer: RiskMeasures
    gross: RiskMeasures

    # Premium build-up
    capital: float
    risk_load: float
    risk_adjusted_premium: float
    gross_technical_premium: float
    expected_reinstatement_factor: float
    upfront_premium: float
    rate_on_line: float
    payback_years: float
    expected_loss_ratio: float

    def to_dict(self) -> dict:
        out = asdict(self)
        out["layer_loss_ci_95"] = list(self.layer_loss_ci_95)
        return out


def price_layer(sim: SimulationResult, config: ModelConfig) -> PricingResult:
    """Price the treaty in ``config`` from the simulated losses in ``sim``."""
    treaty, pricing = config.treaty, config.pricing
    z = sim.layer_annual
    n = sim.n_years

    layer_rm = risk_measures(z, pricing.capital_quantile)
    gross_rm = risk_measures(sim.gross_annual, pricing.capital_quantile)

    el = layer_rm.mean
    se = layer_rm.std / math.sqrt(n)

    capital = max(layer_rm.var - el, 0.0)
    risk_load = pricing.cost_of_capital * capital
    rap = el + risk_load
    gtp = rap / (1.0 - pricing.expense_ratio - pricing.brokerage)

    # Reinstatement premium: expected proportion of the limit that is reinstated.
    if treaty.reinstatements is None or treaty.reinstatements == 0 or treaty.reinstatement_rate == 0:
        reinst_factor = 1.0
    else:
        reinstated = np.minimum(sim.layer_annual_uncapped, treaty.reinstatements * treaty.limit)
        reinst_factor = 1.0 + treaty.reinstatement_rate * float(reinstated.mean()) / treaty.limit
    upfront = gtp / reinst_factor

    rol = upfront / treaty.limit

    return PricingResult(
        treaty=treaty.describe(),
        limit=treaty.limit,
        retention=treaty.retention,
        expected_gross_loss=gross_rm.mean,
        expected_layer_loss=el,
        layer_loss_std_error=se,
        layer_loss_ci_95=(el - 1.96 * se, el + 1.96 * se),
        prob_layer_hit=float((z > 0).mean()),
        expected_layer_hits=float(sim.layer_hits.mean()),
        prob_limit_exhausted=float((sim.largest_loss >= treaty.exhaustion_point).mean()),
        layer=layer_rm,
        gross=gross_rm,
        capital=capital,
        risk_load=risk_load,
        risk_adjusted_premium=rap,
        gross_technical_premium=gtp,
        expected_reinstatement_factor=reinst_factor,
        upfront_premium=upfront,
        rate_on_line=rol,
        payback_years=1.0 / rol if rol > 0 else math.inf,
        expected_loss_ratio=el / gtp if gtp > 0 else math.nan,
    )


def _gbp(x: float) -> str:
    return f"£{x:,.0f}"


def format_summary(result: PricingResult, config: ModelConfig) -> str:
    """Markdown summary table of a pricing run."""
    p, pr, s = config.portfolio, config.pricing, config.simulation
    q = f"{pr.capital_quantile:.1%}"
    lo, hi = result.layer_loss_ci_95
    lines = [
        f"## Pricing summary: {result.treaty}",
        "",
        f"Portfolio: Poisson(λ = {p.claim_frequency:g}) frequency, Lognormal severity "
        f"(mean {_gbp(p.severity_mean)}, CV {p.severity_cv:g}). "
        f"{s.n_years:,} simulated years, seed {s.seed}.",
        "",
        "### Loss statistics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Expected gross annual loss | {_gbp(result.expected_gross_loss)} |",
        f"| Expected annual layer loss (pure premium) | {_gbp(result.expected_layer_loss)} |",
        f"| Monte Carlo standard error | {_gbp(result.layer_loss_std_error)} |",
        f"| 95% confidence interval | {_gbp(lo)} to {_gbp(hi)} |",
        f"| Expected number of losses hitting the layer | {result.expected_layer_hits:.2f} |",
        f"| Probability the layer is hit in a year | {result.prob_layer_hit:.1%} |",
        f"| Probability a single loss exhausts the layer | {result.prob_limit_exhausted:.1%} |",
        f"| Standard deviation of layer loss | {_gbp(result.layer.std)} |",
        f"| {q} VaR of layer loss | {_gbp(result.layer.var)} |",
        f"| {q} TVaR of layer loss | {_gbp(result.layer.tvar)} |",
        "",
        "### Premium build-up",
        "",
        "| Step | Value |",
        "|---|---:|",
        f"| Expected layer loss | {_gbp(result.expected_layer_loss)} |",
        f"| Capital ({q} VaR − mean) | {_gbp(result.capital)} |",
        f"| Risk load ({pr.cost_of_capital:.0%} cost of capital) | {_gbp(result.risk_load)} |",
        f"| **Risk-adjusted premium** | **{_gbp(result.risk_adjusted_premium)}** |",
        f"| Gross technical premium (÷ (1 − {pr.expense_ratio:.0%} expenses − {pr.brokerage:.0%} brokerage)) "
        f"| {_gbp(result.gross_technical_premium)} |",
    ]
    if result.expected_reinstatement_factor != 1.0:
        lines.append(
            f"| Expected reinstatement factor | {result.expected_reinstatement_factor:.4f} |"
        )
    lines += [
        f"| **Upfront premium** | **{_gbp(result.upfront_premium)}** |",
        f"| Rate on line | {result.rate_on_line:.2%} |",
        f"| Payback period | {result.payback_years:.1f} years |",
        f"| Expected loss ratio | {result.expected_loss_ratio:.1%} |",
        "",
    ]
    return "\n".join(lines)
