"""API clients for FRED, EIA, and OECD data sources."""
import time
from pathlib import Path
from datetime import datetime, timedelta

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
