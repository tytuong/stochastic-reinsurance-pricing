"""Charts for the README and the results folder (matplotlib, static PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from .analytics import analytic_moments  # noqa: E402
from .config import ModelConfig  # noqa: E402
from .pricing import PricingResult  # noqa: E402
from .sensitivity import SensitivityRow  # noqa: E402
from .simulation import SimulationResult, layer_loss_per_claim  # noqa: E402

# Palette (validated categorical order; slots 1-3 are safe for colour-blind readers)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK,
    }
)

_millions = FuncFormatter(lambda x, _: f"£{x / 1e6:,.0f}m" if abs(x) >= 1e6 or x == 0 else f"£{x / 1e6:,.1f}m")


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_layer_mechanics(config: ModelConfig, path: Path) -> Path:
    """How individual losses are split between cedant, reinsurer and the excess."""
    t = config.treaty
    losses = np.array([1.5, 4.0, 6.5, 8.0, 10.0, 13.0]) * 1e6
    retained = np.minimum(losses, t.retention)
    ceded = layer_loss_per_claim(losses, t)
    above = losses - retained - ceded

    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(losses.size)[::-1]
    h = 0.6
    gap = 0.03e6  # thin surface-coloured gap between segments
    ax.barh(y, retained - gap, height=h, color=BLUE, label="Retained by cedant")
    ax.barh(y, np.where(ceded > 0, ceded - gap, 0), left=retained, height=h, color=ORANGE,
            label="Recovered from the layer")
    ax.barh(y, np.where(above > 0, above - gap, 0), left=retained + ceded, height=h, color=AQUA,
            label="Above the limit (back to cedant)")
    for yi, total in zip(y, losses):
        ax.text(total + 0.15e6, yi, f"£{total / 1e6:g}m loss", va="center", color=INK_2, fontsize=9)

    for x, lab in [(t.retention, "Retention"), (t.exhaustion_point, "Exhaustion point")]:
        ax.axvline(x, color=INK_2, lw=1, ls="--")
        ax.text(x, y.max() + 0.55, lab, ha="center", va="bottom", color=INK_2, fontsize=9)

    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(_millions)
    ax.set_xlim(0, losses.max() * 1.18)
    ax.set_ylim(-0.6, y.max() + 1.1)
    ax.set_xlabel("Size of individual loss")
    ax.set_title(f"How a {t.describe().split(' (')[0].replace('GBP ', '£')} layer splits each loss", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=9)
    return _save(fig, path)


def plot_annual_loss_distributions(sim: SimulationResult, result: PricingResult, path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    g = sim.gross_annual
    upper = np.quantile(g, 0.999)
    ax1.hist(g[g <= upper], bins=80, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    for x, lab in [(result.gross.mean, "Mean"), (result.gross.var, f"{result.gross.quantile:.1%} VaR")]:
        ax1.axvline(x, color=INK, lw=1, ls="--")
        ax1.text(x, ax1.get_ylim()[1] * 0.95, f" {lab}\n £{x / 1e6:,.1f}m", color=INK, fontsize=8.5, va="top")
    ax1.xaxis.set_major_formatter(_millions)
    ax1.set_xlabel("Gross annual loss")
    ax1.set_ylabel("Number of simulated years")
    ax1.set_title("Gross annual portfolio loss")

    z = sim.layer_annual
    pos = z[z > 0]
    ax2.hist(pos, bins=60, color=ORANGE, edgecolor=SURFACE, linewidth=0.4)
    ax2.xaxis.set_major_formatter(_millions)
    ax2.set_xlabel("Annual loss to the layer (years with a loss)")
    ax2.set_title("Annual loss to the layer")
    ax2.text(
        0.97, 0.95,
        f"{1 - result.prob_layer_hit:.1%} of years:\nno loss to the layer\n\n"
        f"Expected layer loss:\n£{result.expected_layer_loss / 1e6:,.3f}m",
        transform=ax2.transAxes, ha="right", va="top", color=INK, fontsize=9,
    )
    limit = result.limit
    ymax = ax2.get_ylim()[1]
    ax2.annotate("One loss uses\nthe full limit", xy=(limit * 0.98, ymax * 0.6), xytext=(limit * 0.5, ymax * 0.62),
                 color=INK_2, fontsize=8.5, ha="center", va="center",
                 arrowprops={"arrowstyle": "->", "color": INK_2, "lw": 0.8})
    if (z >= 2 * limit - 1e-6).any():
        ax2.annotate("Aggregate limit\nreached", xy=(2 * limit * 0.99, ymax * 0.08),
                     xytext=(1.6 * limit, ymax * 0.3), color=INK_2, fontsize=8.5, ha="center", va="center",
                     arrowprops={"arrowstyle": "->", "color": INK_2, "lw": 0.8})
    return _save(fig, path)


def plot_convergence(sim: SimulationResult, config: ModelConfig, path: Path) -> Path:
    """Running Monte Carlo estimate against the closed-form answer."""
    z = sim.layer_annual_uncapped
    n = np.arange(1, z.size + 1)
    running_mean = np.cumsum(z) / n
    running_sq = np.cumsum(z**2) / n
    running_sd = np.sqrt(np.maximum(running_sq - running_mean**2, 0))
    half_width = 1.96 * running_sd / np.sqrt(n)
    exact = analytic_moments(config.portfolio, config.treaty).expected_layer_loss

    start = 100
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(n[start:], (running_mean - half_width)[start:], (running_mean + half_width)[start:],
                    color=BLUE, alpha=0.15, linewidth=0, label="95% confidence band")
    ax.plot(n[start:], running_mean[start:], color=BLUE, lw=1.5, label="Monte Carlo estimate")
    ax.axhline(exact, color=ORANGE, lw=1.5, ls="--", label=f"Closed-form answer (£{exact:,.0f})")
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"£{x / 1e3:,.0f}k"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel("Number of simulated years (log scale)")
    ax.set_ylabel("Estimated expected layer loss")
    ax.set_title("Convergence of the simulated pure premium (before aggregate limit)")
    ax.legend(loc="upper right")
    return _save(fig, path)


def plot_exceedance_curves(sim: SimulationResult, config: ModelConfig, path: Path) -> Path:
    """Aggregate (AEP) and occurrence (OEP) exceedance curves by return period."""
    t = config.treaty
    n = sim.n_years
    rp = n / np.arange(1, n + 1)  # return period of the k-th largest year
    aep = np.sort(sim.gross_annual)[::-1]
    oep = np.sort(sim.largest_loss)[::-1]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    mask = rp >= 1.0
    ax.plot(rp[mask], aep[mask], color=BLUE, lw=1.8, label="AEP: total loss in the year")
    ax.plot(rp[mask], oep[mask], color=ORANGE, lw=1.8, label="OEP: largest single loss in the year")
    for y, lab in [(t.retention, "Retention"), (t.exhaustion_point, "Exhaustion point")]:
        ax.axhline(y, color=INK_2, lw=1, ls="--")
        ax.text(950, y, f"{lab} ", va="bottom", ha="right", color=INK_2, fontsize=9)
    for period in (10, 50, 200):
        ax.axvline(period, color=GRID, lw=1, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"1-in-{x:g}"))
    ax.yaxis.set_major_formatter(_millions)
    ax.set_ylim(0, np.quantile(sim.gross_annual, 0.999) * 1.05)
    ax.set_xlabel("Return period (years, log scale)")
    ax.set_ylabel("Loss")
    ax.set_title("Exceedance probability curves")
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_tornado(rows: list[SensitivityRow], path: Path) -> Path:
    drivers = list(dict.fromkeys(r.driver for r in rows))
    down = {r.driver: r.change_vs_base for r in rows if r.shock < 0}
    up = {r.driver: r.change_vs_base for r in rows if r.shock > 0}
    order = sorted(drivers, key=lambda d: abs(up[d] - down[d]))
    shock = abs(rows[0].shock)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(order))
    ax.barh(y - 0.19, [down[d] for d in order], height=0.36, color=BLUE, label=f"Assumption −{shock:.0%}")
    ax.barh(y + 0.19, [up[d] for d in order], height=0.36, color=ORANGE, label=f"Assumption +{shock:.0%}")
    for yi, d in zip(y, order):
        for val, off in [(down[d], -0.19), (up[d], 0.19)]:
            ha = "left" if val >= 0 else "right"
            ax.text(val + (0.004 if val >= 0 else -0.004), yi + off, f"{val:+.1%}", va="center", ha=ha,
                    color=INK_2, fontsize=8.5)
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(y, order)
    ax.grid(axis="y", visible=False)
    lim = max(abs(v) for v in list(up.values()) + list(down.values())) * 1.3
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Change in risk-adjusted premium vs base case")
    ax.set_title("Sensitivity of the premium to each assumption")
    ax.legend(loc="lower right")
    return _save(fig, path)
