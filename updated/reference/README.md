# Reference Data

Reference outputs from the original MATLAB/R codebase (`Codes_CNT_2022_JAE/`) used for
validation of the Python reimplementation.

## Files

| File | Source | Description |
|------|--------|-------------|
| `Q.txt` | `utilities/Q.txt` | Quarterly consumption data used in SVAR elasticity bounds |
| `Oilyt_baseline.txt` | `OilUncertaintyConstruction_forPublication/` | Oil price forecast errors (y-type, baseline) |
| `Oilft_baseline.txt` | `OilUncertaintyConstruction_forPublication/` | Oil price forecast factors (f-type, baseline) |
| `Oilsvymeans_baseline.txt` | `OilUncertaintyConstruction_forPublication/` | Stochastic volatility means for y (from R stochvol) |
| `Oilsvfmeans_baseline.txt` | `OilUncertaintyConstruction_forPublication/` | Stochastic volatility means for f (from R stochvol) |
| `ferrorsoil_baseline.mat` | `OilUncertaintyConstruction_forPublication/` | MATLAB workspace with forecast errors, betas, and model data |
| `opu_baseline.mat` | `OilUncertaintyConstruction_forPublication/` | Oil price uncertainty index (521 x 1, 1975.17 - 2018.50) |
| `OilMaster(RACPrice).xlsx` | `OilUncertaintyConstruction_forPublication/` | Master data spreadsheet with RAC oil prices |
| `IRFS_main_REA.mat` | Top-level | SVAR impulse response functions (5-variable system, 25 IRF matrices) |

## Usage

These files are loaded by validation routines to compare Python outputs against the
original published results. They should not be modified.
