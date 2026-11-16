"""Command-line entry point.

Examples
--------
    reinsurance-pricing                                   # base case
    reinsurance-pricing --config config/base_case.toml
    reinsurance-pricing --n-years 500000 --seed 1
    reinsurance-pricing --retention 10e6 --limit 10e6 --reinstatements -1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import plotting
from .analytics import analytic_moments
from .config import ModelConfig, load_config
from .pricing import format_summary, price_layer
from .sensitivity import price_tower, run_sensitivities, write_csv
from .simulation import run_simulation

DEFAULT_TOWER = [(2.5e6, 2.5e6), (5e6, 5e6), (10e6, 10e6), (20e6, 20e6)]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="reinsurance-pricing",
        description="Monte Carlo pricing of an Excess of Loss reinsurance layer.",
    )
    p.add_argument("--config", type=Path, help="TOML file of assumptions (defaults used if omitted)")
    p.add_argument("--n-years", type=int, help="number of simulated years")
    p.add_argument("--seed", type=int, help="random seed")
    p.add_argument("--retention", type=float, help="layer retention (GBP)")
    p.add_argument("--limit", type=float, help="layer limit (GBP)")
    p.add_argument("--reinstatements", type=int, help="number of reinstatements (-1 = unlimited)")
    p.add_argument("--output-dir", type=Path, default=Path("results"), help="where to write tables")
    p.add_argument("--figures-dir", type=Path, default=Path("figures"), help="where to write charts")
    p.add_argument("--no-plots", action="store_true", help="skip chart generation")
    p.add_argument("--no-sensitivity", action="store_true", help="skip sensitivity and layer tower runs")
    return p.parse_args(argv)


def _apply_overrides(config: ModelConfig, args: argparse.Namespace) -> ModelConfig:
    sim, treaty = {}, {}
    if args.n_years is not None:
        sim["n_years"] = args.n_years
    if args.seed is not None:
        sim["seed"] = args.seed
    if args.retention is not None:
        treaty["retention"] = args.retention
    if args.limit is not None:
        treaty["limit"] = args.limit
    if args.reinstatements is not None:
        treaty["reinstatements"] = None if args.reinstatements < 0 else args.reinstatements
    updates = {k: v for k, v in {"simulation": sim, "treaty": treaty}.items() if v}
    return config.with_updates(**updates) if updates else config


def _validation_table(sim, config) -> str:
    exact = analytic_moments(config.portfolio, config.treaty)
    z = sim.layer_annual_uncapped
    se = z.std(ddof=1) / len(z) ** 0.5
    rows = [
        ("Expected layer loss", z.mean(), exact.expected_layer_loss, "£{:,.0f}"),
        ("Std dev of layer loss", z.std(ddof=1), exact.std_layer_loss, "£{:,.0f}"),
        ("P(layer is hit)", (z > 0).mean(), exact.prob_layer_hit, "{:.2%}"),
        ("Expected losses hitting layer", sim.layer_hits.mean(), exact.expected_layer_hits, "{:.4f}"),
        ("Expected gross loss", sim.gross_annual.mean(), exact.expected_gross_loss, "£{:,.0f}"),
    ]
    out = [
        "### Validation against closed-form results (before aggregate limit)",
        "",
        "| Quantity | Simulated | Closed form | Difference |",
        "|---|---:|---:|---:|",
    ]
    for name, simv, exv, fmt in rows:
        out.append(f"| {name} | {fmt.format(simv)} | {fmt.format(exv)} | {simv / exv - 1:+.2%} |")
    z_score = (z.mean() - exact.expected_layer_loss) / se
    out += ["", f"Expected layer loss differs from the exact value by {z_score:+.2f} standard errors.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_config(args.config) if args.config else ModelConfig()
    config = _apply_overrides(config, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    sim = run_simulation(config)
    elapsed = time.perf_counter() - t0
    result = price_layer(sim, config)

    summary = format_summary(result, config) + "\n" + _validation_table(sim, config)
    summary += f"\nSimulated {sim.n_years:,} years ({sim.severities.size:,} individual losses) in {elapsed:.2f}s.\n"

    if not args.no_sensitivity:
        rows = run_sensitivities(config)
        write_csv(rows, args.output_dir / "sensitivity.csv")
        tower = price_tower(config, DEFAULT_TOWER)
        write_csv(tower, args.output_dir / "layer_tower.csv")
        summary += "\n### Layer tower\n\n| Layer | Expected loss | P(hit) | Risk-adjusted premium | Rate on line | Loss ratio |\n"
        summary += "|---|---:|---:|---:|---:|---:|\n"
        for r in tower:
            summary += (
                f"| {r.layer} | £{r.expected_layer_loss:,.0f} | {r.prob_layer_hit:.1%} | "
                f"£{r.risk_adjusted_premium:,.0f} | {r.rate_on_line:.1%} | {r.expected_loss_ratio:.1%} |\n"
            )
        summary += "\n### Sensitivity (±20% on one assumption at a time)\n\n"
        summary += "| Driver | Shock | Risk-adjusted premium | Change vs base |\n|---|---:|---:|---:|\n"
        for r in rows:
            summary += f"| {r.driver} | {r.shock:+.0%} | £{r.risk_adjusted_premium:,.0f} | {r.change_vs_base:+.1%} |\n"

    (args.output_dir / "pricing_summary.md").write_text(summary, encoding="utf-8")
    (args.output_dir / "pricing_summary.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    if not args.no_plots:
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        f = args.figures_dir
        plotting.plot_layer_mechanics(config, f / "01_layer_mechanics.png")
        plotting.plot_annual_loss_distributions(sim, result, f / "02_annual_loss_distributions.png")
        plotting.plot_convergence(sim, config, f / "03_convergence.png")
        plotting.plot_exceedance_curves(sim, config, f / "04_exceedance_curves.png")
        if not args.no_sensitivity:
            plotting.plot_tornado(rows, f / "05_sensitivity_tornado.png")

    print(summary)


if __name__ == "__main__":
    main()
