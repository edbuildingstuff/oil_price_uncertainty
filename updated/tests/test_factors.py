import numpy as np
from opu.factors import factors_em


def test_factors_em_basic():
    np.random.seed(42)
    T, N = 200, 5
    f_true = np.random.randn(T, 1)
    loadings = np.random.randn(N, 1)
    x = f_true @ loadings.T + 0.1 * np.random.randn(T, N)
    ehat, fhat, lamhat, ve2, x2 = factors_em(x, kmax=1, jj=2, demean=2)
    assert fhat.shape == (T, 1)
    assert lamhat.shape == (N, 1)
    corr = abs(np.corrcoef(f_true.ravel(), fhat.ravel())[0, 1])
    assert corr > 0.95


def test_factors_em_with_missing():
    np.random.seed(42)
    T, N = 200, 5
    f_true = np.random.randn(T, 1)
    loadings = np.random.randn(N, 1)
    x = f_true @ loadings.T + 0.1 * np.random.randn(T, N)
    x[10, 2] = np.nan
    x[50, 0] = np.nan
    ehat, fhat, lamhat, ve2, x2 = factors_em(x, kmax=1, jj=2, demean=2)
    assert not np.any(np.isnan(fhat))
    assert not np.any(np.isnan(x2))
