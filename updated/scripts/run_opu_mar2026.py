"""Isolated OPU update to March 2026, written to artifacts_mar2026/.

Runs alongside an ongoing SVAR process without touching shared state:
- New raw cache at artifacts_mar2026/raw/  (original artifacts/raw/ untouched)
- New OPU output at artifacts_mar2026/opu/  (original artifacts/opu/*.npz untouched)
- config.py on disk is never modified, so SVAR spawned workers see unchanged config
- Uses only 3 SV workers to leave cores free for the running SVAR
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEW_ARTIFACTS = PROJECT_ROOT / "artifacts_mar2026"
sys.path.insert(0, str(PROJECT_ROOT))

# Patch opu.config IN MEMORY (disk file untouched). These patches run at
# import time so that spawned SV workers re-importing this module also see
# the same patched values before build_opu is called.
import opu.config as cfg
cfg.RAW_DIR = NEW_ARTIFACTS / "raw"
cfg.OPU_DIR = NEW_ARTIFACTS / "opu"
cfg.SVAR_DIR = NEW_ARTIFACTS / "svar"
cfg.FIGURES_DIR = NEW_ARTIFACTS / "figures"
cfg.SAMPLE_END_YEAR = 2026
cfg.SAMPLE_END_MONTH = 3
for d in [cfg.RAW_DIR, cfg.OPU_DIR, cfg.SVAR_DIR, cfg.FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Patch opu.data module-level names (evaluated at module load from opu.config,
# so re-patching cfg after import doesn't auto-propagate).
import opu.data as data_mod
data_mod.RAW_DIR = cfg.RAW_DIR
data_mod.SAMPLE_END_YEAR = 2026
data_mod.SAMPLE_END_MONTH = 3
data_mod.SAMPLE_START = f"{cfg.SAMPLE_START_YEAR}-{cfg.SAMPLE_START_MONTH:02d}-01"
data_mod.SAMPLE_END = "2026-03-01"


def main(skip_fetch: bool = False):
    print("=== Isolated OPU update ===")
    print(f"Sample:    {data_mod.SAMPLE_START} -> {data_mod.SAMPLE_END}")
    print(f"Artifacts: {NEW_ARTIFACTS}")
    print()

    if not skip_fetch:
        data_mod.fetch_all(force=True)
    else:
        print("Skipping fetch (data already cached).")

    from opu.uncertainty import build_opu
    build_opu(workers=3)

    baseline = cfg.OPU_DIR / "opu_baseline.npz"
    print()
    print("=== Complete ===")
    print(f"New OPU written to: {baseline}")
    print("Original OPU (Dec 2025) at artifacts/opu/opu_baseline.npz is unchanged.")


if __name__ == "__main__":
    skip = "--skip-fetch" in sys.argv
    main(skip_fetch=skip)
