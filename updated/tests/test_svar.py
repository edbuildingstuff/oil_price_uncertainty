import numpy as np
import pytest
from opu.svar import load_svar_data, setup_posterior, draw_posterior


def test_load_svar_data_keys():
    result = load_svar_data()
    assert set(result.keys()) == {"y", "dates", "Q_1"}


def test_load_svar_data_y_shape():
    result = load_svar_data()
    y = result["y"]
    assert y.ndim == 2
    assert y.shape[1] == 5


def test_load_svar_data_y_length():
    result = load_svar_data()
    # OPU baseline has 611 observations; y must match
    assert result["y"].shape[0] == 611


def test_load_svar_data_no_nan():
    result = load_svar_data()
    assert not np.any(np.isnan(result["y"])), "y contains NaN values"


def test_load_svar_data_q1_aligned():
    result = load_svar_data()
    # Q.txt is a static reference file from the original paper sample (ends 2018).
    # Q_1 is trimmed to at most len(y); when the replication extends beyond 2018,
    # Q_1 will be shorter than y -- this is correct because Q_1 is a prior input,
    # not an observation-aligned series.
    assert len(result["Q_1"]) <= len(result["y"])
    assert len(result["Q_1"]) > 0


def test_load_svar_data_dates_aligned():
    result = load_svar_data()
    assert len(result["dates"]) == len(result["y"])


# ---------------------------------------------------------------------------
# Smoke tests for setup_posterior / draw_posterior (synthetic data only)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_var_data():
    """Return synthetic (y, p) for a fast, well-conditioned VAR.

    t=50, n=3, p=2 => T=48, k=1+n*p=7; well above k so X.T@X is full rank.
    """
    rng = np.random.default_rng(0)
    y = rng.standard_normal((50, 3))
    return y, 2


def test_setup_posterior_keys(small_var_data):
    y, p = small_var_data
    post = setup_posterior(y, p)
    expected_keys = {"nuT", "NT", "BbarT", "ST", "EvecB", "Ydep", "X", "T", "n", "p"}
    assert set(post.keys()) == expected_keys


def test_setup_posterior_shapes(small_var_data):
    y, p = small_var_data
    n = y.shape[1]
    post = setup_posterior(y, p)
    k = 1 + n * p  # number of regressors incl. constant
    assert post["BbarT"].shape == (k, n)
    assert post["ST"].shape == (n, n)
    assert post["NT"].shape == (k, k)


def test_draw_posterior_shapes(small_var_data):
    y, p = small_var_data
    n = y.shape[1]
    post = setup_posterior(y, p)
    rng = np.random.default_rng(42)
    B, Sigma, vecB = draw_posterior(post, rng)
    k = 1 + n * p
    assert B.shape == (n, k)
    assert Sigma.shape == (n, n)
    assert vecB.shape == (n * k,)


def test_draw_posterior_sigma_positive_definite(small_var_data):
    y, p = small_var_data
    post = setup_posterior(y, p)
    rng = np.random.default_rng(7)
    _, Sigma, _ = draw_posterior(post, rng)
    # np.linalg.cholesky raises LinAlgError if Sigma is not PSD
    np.linalg.cholesky(Sigma)


def test_draw_posterior_different_seeds_differ(small_var_data):
    y, p = small_var_data
    post = setup_posterior(y, p)
    B1, _, _ = draw_posterior(post, np.random.default_rng(1))
    B2, _, _ = draw_posterior(post, np.random.default_rng(2))
    assert not np.allclose(B1, B2)
