import numpy as np
import pytest
from opu.forecast_errors import newey_west, build_forecast_errors


def test_newey_west_ols_equivalence():
    np.random.seed(0)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    beta_true = np.array([1.0, 2.0])
    y = x @ beta_true + 0.1 * np.random.randn(n)
    result = newey_west(y, x, nlag=4)
    np.testing.assert_array_almost_equal(result["beta"], beta_true, decimal=1)
    assert result["resid"].shape == (n,)
    assert result["tstat"].shape == (2,)


def test_newey_west_tstat_shape():
    np.random.seed(1)
    n = 200
    k = 5
    x = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])
    y = np.random.randn(n)
    result = newey_west(y, x, nlag=4)
    assert result["tstat"].shape == (k,)
    assert result["beta"].shape == (k,)
