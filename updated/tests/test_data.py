import pytest
import numpy as np
from pathlib import Path
from opu.data import fetch_fred_series, fetch_eia_series, fetch_all, load_raw_data


def test_fetch_fred_series_returns_dataframe():
    from opu.config import FRED_API_KEY
    if not FRED_API_KEY:
        pytest.skip("FRED_API_KEY not set")
    df = fetch_fred_series("CPIAUCSL", "1973-01-01", "1973-12-01")
    assert len(df) > 0
    assert "date" in df.columns
    assert "value" in df.columns


def test_fetch_eia_series_returns_dataframe():
    from opu.config import EIA_API_KEY
    if not EIA_API_KEY:
        pytest.skip("EIA_API_KEY not set")
    df = fetch_eia_series(
        route="petroleum/pri/rac2",
        series_id="PET.R1300____3.M",
        start="1974-01",
        end="1974-12",
    )
    assert len(df) > 0
    assert "date" in df.columns
    assert "value" in df.columns
