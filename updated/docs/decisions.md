# Decision Log

Running record of design options considered and choices made during the OPU replication update project. Each decision includes all options presented, the choice, and the rationale — intended for inclusion as supplementary material in the updated paper.

---

## D1: Implementation language

**Date:** 2026-04-16
**Context:** Original code is MATLAB + R. Neither is installed on the development machine. Python 3.12 and `uv` package manager are available.

**Options:**
- MATLAB + R (original stack)
- Python only
- Mixed Python + R

**Decision:** Python only.
**Rationale:** MATLAB license not available, R not installed. Python 3.12 with NumPy/SciPy/Numba provides equivalent numerical capabilities. The stochastic volatility sampler (originally R's `stochvol` package) will be ported as a custom Kim-Shephard-Chib (1998) Gibbs sampler in Python for full methodological parity.

---

## D2: Replication scope

**Date:** 2026-04-16
**Context:** The original paper has two main components: (1) OPU index construction and (2) SVAR analysis of oil market dynamics. The SVAR took the original authors ~1 month of compute with parallel MATLAB scripts.

**Options:**
- A — OPU only (skip SVAR). Fastest, ~hours.
- B — OPU + reduced SVAR (relaxed narrative restrictions). Moderate compute, hours to low days.
- C — Full faithful replication (all restrictions including narrative). Multi-day compute.

**Decision:** C — Full faithful replication.
**Rationale:** The paper's contribution depends on the interaction between the OPU index and the SVAR identification. Reducing the restriction set would undermine the comparison with original results. The development machine (AMD Ryzen 7 7800X3D, 8 cores / 16 threads) with multi-day compute tolerance makes this feasible.

---

## D3: Sample window

**Date:** 2026-04-16
**Context:** Original sample is 1973M1–2018M6. Data sources have varying end dates: EIA (Feb 2026), FRED (Feb 2026), OECD (Dec 2025), JLN uncertainty (Dec 2025).

**Options:**
- A — 1973M1 → Feb 2026. Push to latest available; requires forward-filling or interpolating JLN for Jan–Feb 2026.
- B — 1973M1 → Dec 2025. Common-coverage end date where every input has data.
- C — 1975M1 → Feb 2026. Trim early years to test robustness on a post-1970s-shock window.

**Decision:** B — 1973M1 → 2025M12.
**Rationale:** Maintains the original start date for comparability. Every input series has complete coverage through Dec 2025, avoiding interpolation artifacts. JLN's semi-annual release is the binding constraint (currently through Dec 2025). Option C is a different research question, not an update.

---

## D4: Narrative restrictions

**Date:** 2026-04-16
**Context:** The original uses three historical episodes (1979, 1985/86, 1990) to filter SVAR posterior draws via narrative sign restrictions. The extended sample (through 2025M12) contains major oil-market events not available to the original authors: 2014–16 OPEC collapse, 2020 COVID negative-price event, 2022 Russia invasion / OPEC+ cuts.

**Options:**
- A — Match original exactly (only 1979, 1985/86, 1990 restrictions). Most faithful.
- B — Extend with 1–2 well-established events (e.g., 2014–16, 2020). Stronger identification.
- C — Extend with all three post-2018 episodes. Strongest identification, most debatable bounds.

**Decision:** A for main results + B as robustness.
**Rationale:** The headline result should be "same model, more data" — any differences attributable purely to the sample extension. This avoids reviewer objections about cherry-picked new narratives. The robustness exercise with extended narratives (2014–16 and 2020 episodes) demonstrates identification stability and is reported separately. Narrative threshold calibration for robustness events will be drawn from published studies.

---

## D5: Project architecture

**Date:** 2026-04-16
**Context:** Need a structure that supports multi-day SVAR computation with checkpointing, validation against original outputs, and reproducible figure generation.

**Options:**
- Approach 1 — Script-per-stage (faithful MATLAB mirror). Easy to diff against original; lots of duplicated boilerplate.
- Approach 2 — Modular library + CLI stage runner. Testable, checkpointed, parallelism contained. Slightly harder to diff against original line-by-line.
- Approach 3 — Jupyter notebooks + shared lib. Inline visualization; painful for multi-day compute, poor git diffs.

**Decision:** Approach 2 — Modular library + CLI.
**Rationale:** The multi-day SVAR run requires resilient checkpointing (resume from last accepted draw) and process isolation (no notebook kernel to keep alive). Modular structure enables unit testing of each component (SV sampler, uncertainty recursion, restriction checks) independently. A methodology mapping document bridges the gap to the original MATLAB code for reviewers.

---

## D6: Stochastic volatility estimation

**Date:** 2026-04-16
**Context:** Original uses R's `stochvol` package (Kastner's implementation of the KSC Gibbs sampler). R is not installed.

**Options:**
- A — Port KSC (1998) Gibbs sampler from scratch in Python/NumPy. ~200 lines, full control, exact methodological parity.
- B — Use existing Python package (`pymc`, `numpyro`). Battle-tested MCMC but different sampler mechanics, heavyweight dependency.

**Decision:** A — Custom KSC port.
**Rationale:** The SV model is simple (univariate AR(1) log-volatility). KSC is well-documented with known mixture constants. Custom implementation gives exact methodological parity with the original R code and avoids a heavyweight probabilistic-programming dependency for a 200-line problem. Validation against original `stochvol` outputs on the overlapping sample ensures correctness.

---

## D7: Project directory

**Date:** 2026-04-16
**Context:** Need a new root directory separate from the original replication bundle (`Codes_CNT_2022_JAE/`).

**Options:**
- A — Sibling directory: `oil_price_uncertainty_updated/`
- B — Subdirectory: `oil_price_uncertainty/updated/`
- C — Custom name.

**Decision:** B — `oil_price_uncertainty/updated/`.
**Rationale:** The original replication bundle is already nested under `Codes_CNT_2022_JAE/`, providing clear separation. Keeping the updated code as a sibling subdirectory under the same research project root maintains logical grouping and simplifies navigation.
