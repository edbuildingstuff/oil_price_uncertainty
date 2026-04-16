"""Generate figures and CSV exports for the March 2026 OPU vintage.

Reads from artifacts_mar2026/ (isolated from the running SVAR).
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEW_ARTIFACTS = PROJECT_ROOT / "artifacts_mar2026"
sys.path.insert(0, str(PROJECT_ROOT))

# Patch config so plotting module picks up the March 2026 paths.
import opu.config as cfg
cfg.OPU_DIR = NEW_ARTIFACTS / "opu"
cfg.FIGURES_DIR = NEW_ARTIFACTS / "figures"
cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Import plotting AFTER patch so its module-level imports resolve to new paths.
import opu.plotting as plotting
plotting.OPU_DIR = cfg.OPU_DIR
plotting.FIGURES_DIR = cfg.FIGURES_DIR


def main():
    import numpy as np
    import pandas as pd

    print("=== March 2026 figures + CSV export ===")
    print(f"Reading from: {cfg.OPU_DIR}")
    print(f"Writing to:   {cfg.FIGURES_DIR}")
    print()

    plotting.plot_opu_events()
    plotting.plot_opu_comparison()

    # CSV exports
    baseline = np.load(cfg.OPU_DIR / "opu_baseline.npz", allow_pickle=True)
    ar = np.load(cfg.OPU_DIR / "opu_ar.npz", allow_pickle=True)
    np_data = np.load(cfg.OPU_DIR / "opu_np.npz", allow_pickle=True)

    dates = pd.to_datetime(baseline["dates"])
    opu_b = baseline["opu"]
    opu_ar = ar["opu"]
    opu_np = np_data["opu"]

    # opu_main.csv: baseline + ar aligned to baseline dates (AR has same T as baseline)
    # AR variant typically has the same length as baseline; align tail-wise if not.
    ar_aligned = opu_ar[-len(opu_b):] if len(opu_ar) >= len(opu_b) else np.pad(
        opu_ar, (len(opu_b) - len(opu_ar), 0), constant_values=np.nan
    )
    df_main = pd.DataFrame({
        "date": dates.strftime("%Y-%m"),
        "opu_baseline": opu_b,
        "opu_ar": ar_aligned,
    })
    main_path = cfg.OPU_DIR / "opu_main.csv"
    df_main.to_csv(main_path, index=False)
    print(f"Wrote {main_path} ({len(df_main)} rows)")

    # opu_no_predictors.csv: NP variant has its own length (longer than baseline).
    # Align to a monthly date range that starts 1973-01 + len(baseline_offset) and
    # ends at the last baseline date.
    np_end = dates.max()
    np_start = np_end - pd.DateOffset(months=len(opu_np) - 1)
    np_dates = pd.date_range(np_start, np_end, freq="MS")
    df_np = pd.DataFrame({
        "date": np_dates.strftime("%Y-%m"),
        "opu_np": opu_np,
    })
    np_path = cfg.OPU_DIR / "opu_no_predictors.csv"
    df_np.to_csv(np_path, index=False)
    print(f"Wrote {np_path} ({len(df_np)} rows)")

    print()
    print("=== Complete ===")


if __name__ == "__main__":
    main()
