# Oil Price Uncertainty — Updated Replication

Python replication and extension of Cross, Nguyen & Tran (2022), *"The Role of Precautionary and Speculative Demand in the Global Market for Crude Oil"*, with the sample extended from 1973M1–2018M6 to 1973M1–2025M12.

The original code (MATLAB + R) lives under `../Codes_CNT_2022_JAE/`. This directory is a from-scratch Python port that preserves the original methodology while using current data vintages. See `docs/decisions.md` for the full record of design choices.

## What this produces

1. **OPU index** — monthly oil-price uncertainty index following Jurado, Ludvigson & Ng (2015), built from a dynamic factor model of 131 macro/financial predictors with Kim-Shephard-Chib (1998) stochastic-volatility forecast errors.
2. **SVAR estimates** — 5-variable structural VAR (oil production growth, real economic activity, real oil price, inventory change, OPU) identified via the Inoue-Kilian (2013, 2018) sign + elasticity + dynamic-sign + narrative restrictions.
3. **Figures** — OPU time series with annotated events, impulse responses, forecast-error variance decompositions.

## Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) package manager
- FRED and EIA API keys (see next section — both are free)
- ~8 GB RAM, ≥ 8 CPU cores recommended (the SVAR stage is compute-bound)

## API key registration

The `fetch` stage pulls raw series from two public data providers. Each requires a free API key.

### 1. FRED (Federal Reserve Economic Data, St. Louis Fed)

Used for 131 macro/financial predictors, CPI, and the real-oil-price deflator.

1. Go to https://fredaccount.stlouisfed.org/apikeys and create an account (email + password).
2. After confirming your email, sign in and open https://fredaccount.stlouisfed.org/apikeys.
3. Click **Request API Key**, provide a one-line description (e.g. "oil price uncertainty replication"), and accept the terms.
4. Copy the 32-character alphanumeric key that appears.

Rate limits: 120 requests per 60-second window. The `fetch` stage stays well under this with parquet caching.

### 2. EIA (U.S. Energy Information Administration)

Used for world oil production, refiner acquisition cost, and petroleum stocks.

1. Go to https://www.eia.gov/opendata/register.php.
2. Fill in name, email, and intended use (any one-line description is fine).
3. The API key is emailed to you within a minute — check spam folder if not delivered.

Rate limits: 5,000 requests per hour. The `fetch` stage makes fewer than 20 requests per full refresh.

### 3. Wire keys into the project

```bash
cp .env.example .env
# then edit .env and paste the two keys:
#   FRED_API_KEY=<your 32-char key>
#   EIA_API_KEY=<your EIA key>
```

`.env` is gitignored — keys never touch the repository. `opu/config.py` loads them via `python-dotenv` at startup.

## Setup

```bash
uv sync
```

`uv sync` creates the virtual environment (`.venv/`) and installs pinned dependencies from `uv.lock`. This must succeed before any `run.py` command will work.

## Running the full pipeline

Four stages, in order. All produce artifacts under `artifacts/`.

```bash
uv run python -u run.py fetch                      # 1. download raw data from FRED / EIA / JLN
uv run python -u run.py opu                        # 2. build OPU index (parallel + Numba JIT, ~1 min)
uv run python -u run.py svar --workers 8           # 3. SVAR rotation search (long — days)
uv run python -u run.py figures                    # 4. render all figures
```

The `-u` flag forces unbuffered stdout so progress lines appear immediately on Windows. Set `PYTHONUNBUFFERED=1` in your environment to avoid typing `-u` each time.

### Stage-by-stage detail

**1. `fetch`** — Pulls FRED series (131 macro predictors + real oil price components), EIA world oil production, and Jurado-Ludvigson-Ng macro-uncertainty series. Cached on disk; re-runs skip already-downloaded series unless `--force` is passed.

**2. `opu`** — Runs the full OPU construction: predictor selection (t-stat > 2.575), VAR forecast errors, 20 independent SV chains (100k iterations each), and the JLN uncertainty recursion. Chains run in parallel across workers. Defaults to all CPUs; cap with `--workers N`.

**3. `svar`** — Bayesian SVAR with Normal-Inverse-Wishart posterior, filtered by cascaded sign / elasticity / dynamic-sign / narrative restrictions. Target is 100 accepted draws (`SVAR_TARGET_DRAWS` in `opu/config.py`). Checkpoints after every accepted draw; resume with `--resume`.

  Safe to Ctrl-C anytime. Check progress from another shell:

  ```bash
  uv run python -c "import numpy as np; c=np.load('artifacts/svar/checkpoint.npz', allow_pickle=True); print('accepted:', len(c['accepted']), '/ attempts:', int(c['total_attempts']))"
  ```

**4. `figures`** — Reads `artifacts/opu/` and `artifacts/svar/` and writes PNGs to `artifacts/figures/`. Regenerate a subset with `--which {all,opu,svar,comparison}`.

## Validation

```bash
uv run python -u run.py validate
```

Runs the full 59-test suite under `tests/`. One test (`test_opu_against_reference`) is marked `xfail` — it documents an expected level-divergence from the original paper's outputs due to data-vintage shifts and the removal of `PNGASINDEXM` from FRED. Shape correlation with the reference is ~0.82; see **Known caveats** below.

## Project layout

```
opu/                        core library
├── config.py               paths, seeds, sample window, all tunable parameters
├── data.py                 raw data loading + alignment
├── transforms.py           detrending, deseasonalization, standardization
├── factors.py              dynamic factor extraction for OPU predictors
├── forecast_errors.py      VAR(1) baseline + AR-only + no-predictor variants
├── sv.py                   Kim-Shephard-Chib (1998) SV sampler (Numba-JIT'd)
├── uncertainty.py          JLN recursion; build_opu pipeline
├── svar.py                 SVAR posterior + rotation search main loop
├── identification.py       sign / elasticity / dynamic-sign checks
├── narrative.py            narrative-restriction episodes (1979, 1985/86, 1990; optional 2014-16, 2020)
├── results.py              IRFs, FEVDs, historical decompositions from accepted draws
└── plotting.py             figure generation

tests/                      pytest test suite (59 tests)
reference/                  original paper's .mat / .txt / .xlsx outputs for validation
docs/decisions.md           decision log (D1-D9): scope, methodology, architecture, performance
artifacts/
├── raw/                    fetched data (gitignored)
├── opu/                    opu_baseline.npz, opu_ar.npz, opu_np.npz
├── svar/                   accepted_draws.npz, checkpoint.npz
└── figures/                PNGs
run.py                      CLI entry point
```

## Configuration

All knobs live in `opu/config.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `SAMPLE_START_YEAR/MONTH` | 1973 / 1 | Sample start |
| `SAMPLE_END_YEAR/MONTH` | 2025 / 12 | Sample end |
| `PY`, `PZ` | 24, 24 | Oil-equation lag order (own + predictors) |
| `SV_BURN`, `SV_DRAWS`, `SV_THIN` | 50k / 50k / 10 | SV Gibbs sampler settings |
| `OPU_HORIZON` | 12 | OPU forecast horizon (months) |
| `SVAR_P` | 24 | VAR lag order |
| `SVAR_H` | 17 | IRF horizon |
| `SVAR_ROTATIONS` | 20,000 | QR rotations tested per posterior draw |
| `SVAR_TARGET_DRAWS` | 100 | Accepted draws to collect |
| `SVAR_ROOT_BOUND` | 0.99101 | Stationarity eigenvalue bound |
| `SVAR_N_WORKERS` | 8 | SVAR rotation workers |
| `SEED_SV_Y`, `SEED_SV_F`, `SEED_SVAR` | 0, 1000, 42 | RNG seeds (seeds are index-keyed so parallelism is deterministic) |

## Methodology at a glance

For reviewers who want to verify methodological fidelity without reading every module:

**OPU index construction** (`opu/uncertainty.py`, following Jurado, Ludvigson & Ng 2015):

1. **Predictor panel** — 131 FRED-MD style macro/financial series plus EIA oil-market series. Transformations (log-differences, detrending, deseasonalization) follow `opu/transforms.py`, a direct port of the original `prepare_missing.m`.
2. **Factor extraction** — principal-components extraction with the Bai-Ng (2002) information criterion for factor count (`opu/factors.py`).
3. **Predictor selection** — each candidate predictor is tested in a VAR(1) with the oil-price series. Predictors with Newey-West HAC t-statistic > 2.575 (99% significance) are retained (`opu/forecast_errors.py`, `TSTAT_THRESHOLD` in config).
4. **Forecast errors** — one-step-ahead VAR residuals for oil price and each retained predictor.
5. **Stochastic volatility** — independent Kim-Shephard-Chib (1998) Gibbs samplers on each residual series, 50k burn + 50k draws, thin 10 (`opu/sv.py`, Numba-JIT'd).
6. **Uncertainty recursion** — JLN h-step recursion combining SV states, AR lag structure, and cross-series loadings (`compute_uf`, `compute_uy`). Headline OPU is the 12-month horizon.

**SVAR identification** (`opu/svar.py` + `opu/identification.py` + `opu/narrative.py`, following Kilian-Murphy 2014 / Inoue-Kilian 2013, 2018):

- **Variables:** (1) world oil production growth, (2) real economic activity (Kilian IGREA index), (3) log real oil price, (4) inventory change, (5) OPU.
- **Prior:** diffuse Normal-Inverse-Wishart (N₀ = 0, S₀ = 0, ν₀ = 0). VAR(24).
- **Posterior:** standard NIW conjugate draws with stationarity filter (max eigenvalue < 0.99101).
- **Identification — cascaded restrictions applied to each QR rotation:**
  1. Static sign restrictions on the impact matrix (oil supply, aggregate demand, oil-specific demand, precautionary/inventory demand, uncertainty).
  2. Use-elasticity bound (absolute value in [0, 0.04]; supply elasticity w.r.t. oil price in [−0.8, 0.0]).
  3. Dynamic sign restrictions over the IRF horizon (H = 17).
  4. Narrative sign restrictions: contributions of the supply shock at the 1979 Iranian revolution, 1985/86 OPEC collapse, and 1990 Gulf War episodes (extended robustness: 2014–16 OPEC price war and 2020 COVID crash).
- **Output:** 100 accepted posterior draws, each with B, Σ, and B₀⁻¹. Downstream IRFs, FEVDs, and historical decompositions are computed from these accepted draws with 68% credible bands.

**Narrative episodes used (exact month indices):**

| Episode | Months (inclusive) | Restriction |
|---|---|---|
| 1979 Iranian revolution | 1978M11 – 1979M12 | Supply shock raised real oil price |
| 1985/86 OPEC collapse | 1985M11 – 1986M07 | Supply shock lowered real oil price |
| 1990 Gulf War | 1990M08 – 1990M10 | Supply shock raised real oil price |
| 2014–16 OPEC price war (robustness) | 2014M11 – 2016M02 | Supply shock lowered real oil price |
| 2020 COVID crash (robustness) | 2020M03 – 2020M06 | Demand shock lowered real oil price |

## Reproducibility

- **Seeds.** All RNG is seeded from `opu/config.py`: `SEED_SV_Y`, `SEED_SV_F`, `SEED_SVAR`. The SV seeds are index-keyed per chain (`SEED_SV_Y + i`), so parallel execution is deterministic regardless of worker completion order.
- **Pinned dependencies.** `uv.lock` pins exact versions of every direct and transitive dependency. On another machine, `uv sync` reproduces the environment bit-for-bit.
- **Data vintages.** Raw data is cached to `artifacts/raw/` as parquet after first fetch. The cache is keyed by series ID and has a 30-day freshness check. To force a re-fetch from source, run `uv run python run.py fetch --force`.
- **Checkpoints.** SVAR state is written to `artifacts/svar/checkpoint.npz` after every accepted draw, enabling safe Ctrl-C and `--resume`. Final accepted draws are at `artifacts/svar/accepted_draws.npz`.
- **Validation suite.** `uv run python run.py validate` runs 59 tests, including regression tests against MATLAB reference outputs in `reference/` on the overlapping 1973M1–2018M6 window.

## Performance

The compute-bound stages are (1) OPU stochastic-volatility sampling and (2) SVAR rotation search. Both run in parallel across processes; the SV inner loop is additionally JIT-compiled with Numba. See `docs/decisions.md` entries **D8** (parallelism strategy) and **D9** (SV JIT) for the rationale, options considered, and measured speedups. End-to-end OPU construction: hours → ~1 minute. SVAR runtime: 2–7 days on 8 cores (dominated by narrative-restriction acceptance rate).

## Known caveats

- **Level divergence from the 2022 paper's OPU.** The replicated index has Pearson correlation 0.82 with the original over the overlapping 1975M2–2018M6 window but sits ~2.5× higher in level. Drivers: (a) the 2020 COVID oil-price crash inflates the baseline standard deviation used for z-scoring; (b) `PNGASINDEXM` was removed from FRED and is substituted with `PNGASUSUSDM` + `PNGASEUUSDM`; (c) current FRED fuel/gas series begin 1992 vs. the 2018-vintage availability; (d) cumulative FRED revisions. **Discuss in the paper — don't treat as a bug.** When cross-comparing with the original index, z-score each series to its own sample first.
- **`Q.txt` ends 2018-06.** The Kilian production series used to derive the use-elasticity prior bound is a static reference file. When `y` extends past 2018, `Q_1` is shorter than `y` — this is acceptable because `Q_1` is used only to derive a scalar prior bound, not as an observation-aligned series.
- **Narrative acceptance rate is unknown a priori.** If the checkpoint shows 0 accepted draws after ~6 hours of SVAR runtime, investigate the narrative-restriction parameters rather than waiting longer — the issue is calibration, not throughput.

## References

- Cross, J., Nguyen, B. H., & Tran, T. D. (2022). The Role of Precautionary and Speculative Demand in the Global Market for Crude Oil. *Journal of Applied Econometrics*.
- Inoue, A., & Kilian, L. (2013, 2018). Inference on impulse response functions in structural VAR models.
- Jurado, K., Ludvigson, S. C., & Ng, S. (2015). Measuring uncertainty. *American Economic Review*.
- Kilian, L., & Murphy, D. P. (2014). The role of inventories and speculative trading in the global market for crude oil. *Journal of Applied Econometrics*.
- Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: Likelihood inference and comparison with ARCH models. *Review of Economic Studies*.
