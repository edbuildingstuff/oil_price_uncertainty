"""Bayesian SVAR estimation. Ports main_1.m through main_6.m."""
import numpy as np
from pathlib import Path
from opu.config import (
    SVAR_P, SVAR_H, SVAR_ROTATIONS, SVAR_TARGET_DRAWS,
    SVAR_ROOT_BOUND, SVAR_ALPHA, SVAR_ELASTICITY_BOUND,
    SVAR_USE_ELASTICITY_BOUNDS, SVAR_N_WORKERS, SEED_SVAR,
    OPU_DIR, SVAR_DIR, REFERENCE_DIR,
)
from opu.transforms import deseasonalize


def load_svar_data() -> dict:
    """Construct the 5-variable SVAR dataset.

    Variables: (1) oil production growth, (2) REA, (3) log real oil price,
    (4) inventory change, (5) OPU.
    """
    from opu.data import load_raw_data, SAMPLE_START, SAMPLE_END
    import pandas as pd

    raw = load_raw_data()
    start = pd.Timestamp(SAMPLE_START)
    end = pd.Timestamp(SAMPLE_END)
    dates = pd.date_range(start, end, freq="MS")

    def _align(df):
        # Drop duplicates before reindexing (e.g. production series has dupes)
        df = df.drop_duplicates(subset="date", keep="first").sort_values("date")
        return df.set_index("date").reindex(dates).interpolate()["value"].values

    # World oil production (growth rate = diff(log()))
    prod = _align(raw["production"])
    dprod = np.diff(np.log(prod))

    # REA (Kilian index, level)
    rea = _align(raw["IGREA"])[1:]  # trim to match dprod

    # Real oil price: RAC / CPI, rescaled
    rac = _align(raw["rac"])
    cpi = _align(raw["CPIAUCSL"])
    rpoil = rac / cpi
    # Normalize: Nov 2008 = $61.65
    # Find index for Nov 2008
    nov2008 = np.argmin(np.abs(dates - pd.Timestamp("2008-11-01")))
    rpoil = 49.1 * rpoil / rpoil[nov2008]
    lrpoil = np.log(rpoil)[1:]  # trim to match dprod

    # Inventory change
    stocks = _align(raw["stocks"])
    dstocks = np.diff(stocks)

    # OPU (load from artifacts)
    opu_data = np.load(OPU_DIR / "opu_baseline.npz")
    opu = opu_data["opu"]

    # Align all to common sample (OPU loses first 24 obs)
    p = SVAR_P
    n_vars = min(len(dprod), len(rea), len(lrpoil), len(dstocks))
    offset = n_vars - len(opu)

    y = np.column_stack([
        dprod[offset:offset + len(opu)],
        rea[offset:offset + len(opu)],
        lrpoil[offset:offset + len(opu)],
        dstocks[offset:offset + len(opu)],
        opu,
    ])

    # Deseasonalize first 4 variables
    for i in range(4):
        y[:, i] = deseasonalize(y[:, i])

    # Load Q.txt for use-elasticity prior.
    # Q.txt holds world oil production (thousands of barrels/day).
    # The formula converts to a scaled production level used in the
    # use-elasticity bound: (10000 * Q * 30 / 1000).
    Q = np.loadtxt(REFERENCE_DIR / "Q.txt")
    # Q.txt is a single column; handle both 1D (ndim==1) and 2D (ndim==2)
    # cases defensively so this port survives future data updates.
    if Q.ndim == 1:
        # Single-column file: Q is already the production series
        Q_1 = (10000 * Q[1:]) * 30 / 1000
    else:
        # Multi-column file: take the first column (index 0) as in original spec
        Q_1 = (10000 * Q[1:, 0]) * 30 / 1000
    # Trim Q_1 to at most len(y) if Q extends further into the sample than y.
    # Q.txt is a static reference file (original paper ends 2018); when the
    # replication sample extends beyond Q, Q_1 will be shorter than y -- that
    # is acceptable because Q_1 is used only to derive a scalar prior bound,
    # not as an observation-by-observation aligned series.
    Q_1 = Q_1[:len(y)]

    return {
        "y": y,
        "dates": dates[-len(y):],
        "Q_1": Q_1,
    }
