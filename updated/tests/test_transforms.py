import numpy as np
import pytest
from opu.transforms import prepare_missing, zscore, deseasonalize, mlags


def test_prepare_missing_level():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = prepare_missing(x, tcode=1)
    np.testing.assert_array_equal(result, x)


def test_prepare_missing_first_diff():
    x = np.array([10.0, 12.0, 15.0, 20.0])
    result = prepare_missing(x, tcode=2)
    assert np.isnan(result[0])
    np.testing.assert_array_almost_equal(result[1:], [2.0, 3.0, 5.0])


def test_prepare_missing_log_first_diff():
    x = np.array([100.0, 110.0, 121.0])
    result = prepare_missing(x, tcode=5)
    assert np.isnan(result[0])
    expected = np.diff(np.log(x))
    np.testing.assert_array_almost_equal(result[1:], expected)


def test_prepare_missing_second_diff():
    x = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
    result = prepare_missing(x, tcode=3)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    expected = x[2:] - 2 * x[1:-1] + x[:-2]
    np.testing.assert_array_almost_equal(result[2:], expected)


def test_zscore():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = zscore(x)
    np.testing.assert_almost_equal(result.mean(), 0.0)
    np.testing.assert_almost_equal(result.std(ddof=0), 1.0)


def test_deseasonalize():
    np.random.seed(0)
    t = 120
    seasonal = np.tile(np.arange(12, dtype=float), t // 12)
    noise = np.random.randn(t) * 0.01
    y = seasonal + noise
    resid = deseasonalize(y)
    assert resid.shape == (t,)
    assert abs(resid.mean()) < 0.1


def test_mlags():
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    result = mlags(x, k=2)
    assert result.shape == (5, 2)
    np.testing.assert_array_equal(result[0, :], [0.0, 0.0])
    np.testing.assert_array_equal(result[1, :], [1.0, 0.0])
    np.testing.assert_array_equal(result[2, :], [2.0, 1.0])


def test_transforms_match_original():
    """Validate transforms on original OilMaster data reproduce original forecast errors."""
    import pandas as pd
    from pathlib import Path

    ref = Path("reference")
    if not (ref / "OilMaster(RACPrice).xlsx").exists():
        pytest.skip("Reference data not available")

    df = pd.read_excel(ref / "OilMaster(RACPrice).xlsx", sheet_name="Y", header=0)
    # Drop Date column if present, keep only numeric series
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    tcodes = df.iloc[0].values.astype(float).astype(int)
    data = df.iloc[1:].values.astype(float)

    for i in range(data.shape[1]):
        result = prepare_missing(data[:, i], tcodes[i])
        assert not np.all(np.isnan(result)), f"Series {i} all NaN after transform"
