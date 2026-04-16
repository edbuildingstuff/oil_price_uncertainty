"""Figure generation for OPU and SVAR results."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from opu.config import FIGURES_DIR, OPU_DIR, SVAR_DIR, SVAR_H

SHOCK_NAMES = ["Flow Supply", "Flow Demand", "Speculative Demand", "Residual", "Uncertainty Demand"]
VAR_NAMES = ["Oil Production", "REA", "Real Oil Price", "Inventories", "OPU"]

OPU_EVENTS = {
    1973.83: "Arab Oil Embargo",
    1979.0: "Iranian Revolution",
    1980.75: "Iran-Iraq War",
    1985.92: "Saudi Output Surge",
    1990.58: "Gulf War",
    1997.5: "Asian Financial Crisis",
    2001.75: "9/11",
    2003.17: "Iraq War",
    2008.5: "GFC",
    2011.0: "Arab Spring",
    2014.5: "OPEC Collapse",
    2020.17: "COVID-19",
    2022.17: "Russia-Ukraine",
    2026.17: "Hormuz Closure",
}


def _to_decimal_years(dates):
    """Convert a dates array to decimal years.

    Handles three formats:
    - Objects with .year and .month attributes (e.g. pandas Timestamps)
    - numpy datetime64 arrays
    - Arrays already containing floats (returned as-is)
    """
    if hasattr(dates[0], "year"):
        # pandas Timestamps or similar
        return np.array([d.year + d.month / 12 for d in dates])
    try:
        import pandas as pd
        ts = pd.DatetimeIndex(dates)
        return ts.year.values + ts.month.values / 12
    except Exception:
        # Last resort: treat as already numeric
        return dates.astype(float)


def plot_opu_events():
    """Figure 1: OPU time series with event annotations."""
    data = np.load(OPU_DIR / "opu_baseline.npz", allow_pickle=True)
    opu = data["opu"]
    dates = data["dates"]
    t = _to_decimal_years(dates)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, opu, "k-", linewidth=0.8)
    ax.set_ylabel("Oil Price Uncertainty")
    ax.set_xlabel("Year")

    for year, label in OPU_EVENTS.items():
        if t[0] <= year <= t[-1]:
            ax.axvline(year, color="gray", linestyle="--", alpha=0.4, linewidth=0.5)
            ax.text(year, ax.get_ylim()[1] * 0.95, label, rotation=90,
                    fontsize=6, va="top", ha="right")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_opu_events.pdf")
    fig.savefig(FIGURES_DIR / "fig1_opu_events.png", dpi=150)
    plt.close(fig)
    print("Saved fig1_opu_events")


def plot_opu_comparison():
    """Figure 2: Baseline vs AR-only vs no-predictor OPU."""
    baseline = np.load(OPU_DIR / "opu_baseline.npz", allow_pickle=True)
    ar = np.load(OPU_DIR / "opu_ar.npz", allow_pickle=True)
    np_data = np.load(OPU_DIR / "opu_np.npz", allow_pickle=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    n = min(len(baseline["opu"]), len(ar["opu"]), len(np_data["opu"]))
    t = np.arange(n)
    ax.plot(t, baseline["opu"][:n], "k-", label="Baseline", linewidth=0.8)
    ax.plot(t, ar["opu"][:n], "b--", label="AR only", linewidth=0.6)
    ax.plot(t, np_data["opu"][:n], "r:", label="No predictor", linewidth=0.6)
    ax.legend()
    ax.set_ylabel("OPU")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_opu_comparison.pdf")
    plt.close(fig)
    print("Saved fig2_opu_comparison")


def plot_irfs():
    """Figure 5+: Structural IRFs with credible bands."""
    data = np.load(SVAR_DIR / "results.npz")
    med = data["irf_median"]
    lo = data["irf_lower"]
    hi = data["irf_upper"]
    n = med.shape[0]
    H = med.shape[2] - 1
    horizons = np.arange(H + 1)

    fig, axes = plt.subplots(n, n, figsize=(16, 14))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.fill_between(horizons, lo[i, j], hi[i, j], alpha=0.2, color="blue")
            ax.plot(horizons, med[i, j], "b-", linewidth=1)
            ax.axhline(0, color="k", linewidth=0.5)
            if i == 0:
                ax.set_title(SHOCK_NAMES[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(VAR_NAMES[i], fontsize=8)
            ax.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_irfs.pdf")
    plt.close(fig)
    print("Saved fig5_irfs")


def plot_fevds():
    """Figure 6+: Forecast error variance decompositions."""
    data = np.load(SVAR_DIR / "results.npz")
    fevd = data["fevd_median"]
    n = fevd.shape[0]
    H = fevd.shape[2] - 1
    horizons = np.arange(H + 1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(1, n, figsize=(16, 3.5))
    for i in range(n):
        ax = axes[i]
        bottom = np.zeros(H + 1)
        for j in range(n):
            ax.fill_between(horizons, bottom, bottom + fevd[i, j], alpha=0.7,
                          color=colors[j], label=SHOCK_NAMES[j] if i == 0 else "")
            bottom += fevd[i, j]
        ax.set_title(VAR_NAMES[i], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, H)
    axes[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_fevds.pdf")
    plt.close(fig)
    print("Saved fig6_fevds")


def generate_figures(which: str = "all"):
    """Generate all figures or a subset."""
    if which in ("all", "opu"):
        plot_opu_events()
        plot_opu_comparison()
    if which in ("all", "svar"):
        plot_irfs()
        plot_fevds()
    if which in ("all", "comparison"):
        plot_opu_events()
    print(f"Figures saved to {FIGURES_DIR}")
