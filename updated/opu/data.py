"""API clients for FRED, EIA, and OECD data sources."""
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from opu.config import (
    FRED_API_KEY, EIA_API_KEY, RAW_DIR,
    SAMPLE_START_YEAR, SAMPLE_START_MONTH,
    SAMPLE_END_YEAR, SAMPLE_END_MONTH,
)

SAMPLE_START = f"{SAMPLE_START_YEAR}-{SAMPLE_START_MONTH:02d}-01"
SAMPLE_END = f"{SAMPLE_END_YEAR}-{SAMPLE_END_MONTH:02d}-01"

FRED_SERIES = {
    "DEXUSAL": "AUD exchange rate",
    "EXCAUS": "CAD exchange rate",
    "CCUSSP02CLM650N": "CLP exchange rate",
    "DEXUSNZ": "NZD exchange rate",
    "EXSFUS": "ZAR exchange rate",
    "DEXNOUS": "NOK exchange rate (unused in model)",
    "IGREA": "Kilian REA",
    "M1SL": "US M1",
    "CPIAUCSL": "US CPI",
    "PNRGINDEXM": "WB energy index",
    "POILWTIUSDM": "WTI crude avg",
    "POILDUBUSDM": "Dubai crude avg",
    "PCOALAUUSDM": "Australian coal",
    "PNGASUSUSDM": "US natural gas",
    "PNGASEUUSDM": "EU natural gas",
    # NOTE: "PNGASINDEXM" (WB natural gas index) does not exist on FRED.
    # Individual gas prices (PNGASUSUSDM, PNGASEUUSDM) are available instead.
}

EIA_SERIES = {
    "rac": {
        "route": "petroleum/pri/rac2",
        "series_id": "PET.R1300____3.M",
        "description": "US crude imported acquisition cost",
    },
    "production": {
        "route": "petroleum/crd/crpdn",
        "series_id": "PET.MCRFPUS2.M",
        "description": "US field production of crude oil",
    },
    "stocks": {
        "route": "petroleum/sum/crdsnd",
        "series_id": "PET.MCESTUS1.M",
        "description": "US ending stocks excluding SPR of crude oil",
    },
}


def _cache_path(source: str, series_id: str) -> Path:
    safe_id = series_id.replace(".", "_").replace("/", "_")
    return RAW_DIR / f"{source}_{safe_id}.parquet"


def _is_fresh(path: Path, max_age_days: int = 30) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(days=max_age_days)


def fetch_fred_series(series_id: str, start: str, end: str) -> pd.DataFrame:
    """Fetch a single monthly series from the FRED API."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
        "frequency": "m",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["observations"]
    df = pd.DataFrame(data)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def fetch_eia_series(route: str, series_id: str, start: str, end: str) -> pd.DataFrame:
    """Fetch a single monthly series from the EIA API v2.

    The series_id (e.g. 'PET.R1300____3.M') is parsed to extract the
    inner series code (e.g. 'R1300____3') used to filter the response,
    since EIA v2 routes may return multiple series per endpoint.

    Handles pagination by following the offset until all records are retrieved.
    """
    # Extract the inner series code from the full series_id (e.g. PET.R1300____3.M -> R1300____3)
    parts = series_id.split(".")
    series_code = parts[1] if len(parts) >= 2 else series_id

    url = f"https://api.eia.gov/v2/{route}/data/"
    all_records = []
    offset = 0
    page_size = 5000

    while True:
        params = {
            "api_key": EIA_API_KEY,
            "frequency": "monthly",
            "data[0]": "value",
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": page_size,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        records = body["response"]["data"]
        all_records.extend(records)

        # Check if there are more pages
        total = int(body["response"].get("total", len(records)))
        if offset + len(records) >= total or len(records) == 0:
            break
        offset += len(records)

    df = pd.DataFrame(all_records)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])

    # Filter to the specific series if the response contains a 'series' column
    if "series" in df.columns:
        df = df[df["series"] == series_code]

    df["date"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]].dropna(subset=["value"]).sort_values("date").reset_index(drop=True)


def fetch_oecd_ip() -> pd.DataFrame:
    """Fetch OECD+6NME industrial production from OECD Data Explorer."""
    cache = _cache_path("oecd", "ip_oecd6nme")
    if cache.exists():
        return pd.read_parquet(cache)
    raise NotImplementedError(
        "OECD Data Explorer endpoint TBD -- resolve exact dataflow during implementation. "
        "Place manual CSV at artifacts/raw/oecd_ip_oecd6nme.parquet as fallback."
    )


def fetch_all(force: bool = False):
    """Fetch all FRED and EIA series, caching results as parquet files."""
    print("Fetching FRED series...")
    for sid, desc in FRED_SERIES.items():
        cache = _cache_path("fred", sid)
        if not force and _is_fresh(cache):
            print(f"  {sid}: cached")
            continue
        print(f"  {sid}: fetching ({desc})")
        try:
            df = fetch_fred_series(sid, SAMPLE_START, SAMPLE_END)
            df.to_parquet(cache, index=False)
            print(f"    -> {len(df)} observations")
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(0.6)

    print("Fetching EIA series...")
    eia_start = f"{SAMPLE_START_YEAR}-{SAMPLE_START_MONTH:02d}"
    eia_end = f"{SAMPLE_END_YEAR}-{SAMPLE_END_MONTH:02d}"
    for name, spec in EIA_SERIES.items():
        cache = _cache_path("eia", name)
        if not force and _is_fresh(cache):
            print(f"  {name}: cached")
            continue
        print(f"  {name}: fetching ({spec['description']})")
        try:
            df = fetch_eia_series(spec["route"], spec["series_id"], eia_start, eia_end)
            df.to_parquet(cache, index=False)
            print(f"    -> {len(df)} observations")
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(0.6)

    print("Fetch complete.")


def load_raw_data() -> dict[str, pd.DataFrame]:
    """Load all cached raw data into a dict keyed by series name."""
    result = {}
    for sid in FRED_SERIES:
        path = _cache_path("fred", sid)
        if path.exists():
            result[sid] = pd.read_parquet(path)
    for name in EIA_SERIES:
        path = _cache_path("eia", name)
        if path.exists():
            result[name] = pd.read_parquet(path)
    return result


def build_opu_dataset() -> dict:
    """Assemble the full dataset for OPU construction from cached raw data.

    Returns dict with keys: yt (oil prices), xt (predictors), dates.
    All series aligned to common monthly index, transformed, z-scored.
    """
    from opu.transforms import prepare_missing, zscore
    from opu.factors import factors_em

    raw = load_raw_data()
    # Build date index
    start = pd.Timestamp(f"{SAMPLE_START_YEAR}-{SAMPLE_START_MONTH:02d}-01")
    end = pd.Timestamp(f"{SAMPLE_END_YEAR}-{SAMPLE_END_MONTH:02d}-01")
    dates = pd.date_range(start, end, freq="MS")

    def _align(df, dates):
        # Drop duplicate dates (some EIA series have duplicates) and sort
        df = df.drop_duplicates(subset="date", keep="first").sort_values("date")
        df = df.set_index("date").reindex(dates).interpolate(method="linear")
        # Back/forward fill edges that linear interpolation won't cover
        df = df.bfill().ffill()
        return df["value"].values

    def _safe_zscore(x):
        """Z-score with NaN / constant-series protection."""
        x = np.asarray(x, dtype=float)
        if np.all(np.isnan(x)):
            return np.zeros_like(x)
        mask = ~np.isnan(x)
        m = np.mean(x[mask])
        s = np.std(x[mask], ddof=0)
        if s < 1e-12:
            return np.zeros_like(x)
        out = (x - m) / s
        # Replace any remaining NaNs with zero (post-standardization)
        out[np.isnan(out)] = 0.0
        return out

    # Oil prices (Y sheet equivalent) -- use RAC, energy index, crude avg
    rac = _align(raw.get("rac", raw.get("CPIAUCSL")), dates)  # fallback
    energy_idx = _align(raw["PNRGINDEXM"], dates)
    crude_avg = _align(raw["POILWTIUSDM"], dates)

    # Transformation codes matching original: RAC=5, energy=5, crude_avg=1
    yt_rac = prepare_missing(rac, tcode=5)
    yt_energy = prepare_missing(energy_idx, tcode=5)
    yt_crude = prepare_missing(crude_avg, tcode=1)
    yt_raw = np.column_stack([yt_rac, yt_energy, yt_crude])

    # Remove first NaN row from log-diff
    valid = ~np.any(np.isnan(yt_raw), axis=1)
    first_valid = int(np.argmax(valid))
    yt_raw = yt_raw[first_valid:]
    dates_valid = dates[first_valid:]

    yt = np.column_stack([_safe_zscore(yt_raw[:, i]) for i in range(yt_raw.shape[1])])

    # Exchange rates
    fx_ids = ["DEXUSAL", "EXCAUS", "CCUSSP02CLM650N", "DEXUSNZ", "EXSFUS"]
    fx = np.column_stack([
        _safe_zscore(prepare_missing(_align(raw[sid], dates)[first_valid:], tcode=5))
        if sid in raw else np.zeros(len(dates_valid))
        for sid in fx_ids
    ])

    # Other predictors
    rea = _safe_zscore(prepare_missing(_align(raw["IGREA"], dates)[first_valid:], tcode=1))
    prod_df = raw.get("production", pd.DataFrame({"date": dates, "value": np.zeros(len(dates))}))
    stocks_df = raw.get("stocks", pd.DataFrame({"date": dates, "value": np.zeros(len(dates))}))
    prod = _safe_zscore(prepare_missing(_align(prod_df, dates)[first_valid:], tcode=5))
    stocks = _safe_zscore(prepare_missing(_align(stocks_df, dates)[first_valid:], tcode=5))
    m1 = _safe_zscore(prepare_missing(_align(raw["M1SL"], dates)[first_valid:], tcode=5))
    cpi = _safe_zscore(prepare_missing(_align(raw["CPIAUCSL"], dates)[first_valid:], tcode=5))

    # Fuel group factors
    coal = _align(raw["PCOALAUUSDM"], dates)[first_valid:]
    gas_us = _align(raw["PNGASUSUSDM"], dates)[first_valid:]
    gas_eu = _align(raw["PNGASEUUSDM"], dates)[first_valid:]
    fuel = np.column_stack([
        _safe_zscore(prepare_missing(coal, tcode=5)),
        _safe_zscore(prepare_missing(gas_us, tcode=5)),
        _safe_zscore(prepare_missing(gas_eu, tcode=5)),
    ])
    # Remove NaN rows from fuel
    fuel_valid = ~np.any(np.isnan(fuel), axis=1)
    if not np.all(fuel_valid):
        fuel[~fuel_valid] = 0.0

    _, fhat, _, _, _ = factors_em(fuel, kmax=1, jj=2, demean=2)
    _, ghat, _, _, _ = factors_em(fuel ** 2, kmax=1, jj=2, demean=2)

    xt = np.column_stack([fx, rea, prod, stocks, m1, cpi, fhat, fhat ** 2, ghat])

    return {"yt": yt, "xt": xt, "dates": dates_valid}
