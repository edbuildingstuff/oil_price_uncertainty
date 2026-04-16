from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RAW_DIR = ARTIFACTS_DIR / "raw"
OPU_DIR = ARTIFACTS_DIR / "opu"
SVAR_DIR = ARTIFACTS_DIR / "svar"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
REFERENCE_DIR = PROJECT_ROOT / "reference"

for d in [RAW_DIR, OPU_DIR, SVAR_DIR, FIGURES_DIR, REFERENCE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- API Keys ---
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
EIA_API_KEY = os.environ.get("EIA_API_KEY", "")

# --- Sample ---
SAMPLE_START_YEAR = 1973
SAMPLE_START_MONTH = 1
SAMPLE_END_YEAR = 2025
SAMPLE_END_MONTH = 12

# --- OPU parameters ---
PY = 24  # own lags for oil price equation
PZ = 24  # predictor lags
SV_BURN = 50_000
SV_DRAWS = 50_000
SV_THIN = 10
OPU_HORIZON = 12
TSTAT_THRESHOLD = 2.575  # 99% significance for predictor selection

# --- SVAR parameters ---
SVAR_P = 24  # VAR lag order
SVAR_H = 17  # IRF horizon
SVAR_ROTATIONS = 20_000  # rotations per posterior draw
SVAR_TARGET_DRAWS = 100  # accepted draws to collect
SVAR_ROOT_BOUND = 0.99101  # stationarity eigenvalue bound
SVAR_ALPHA = 0.32  # credible set width (68%)
SVAR_ELASTICITY_BOUND = 0.04
SVAR_USE_ELASTICITY_BOUNDS = (-0.8, 0.0)
SVAR_N_WORKERS = 8

# --- RNG seeds ---
SEED_SV_Y = 0
SEED_SV_F = 1000
SEED_SVAR = 42
