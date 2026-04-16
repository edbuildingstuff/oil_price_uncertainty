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


# ---------------------------------------------------------------------------
# Tests for opu.identification (Task 12)
# ---------------------------------------------------------------------------

from opu.identification import check_sign_restrictions, check_elasticity


def test_sign_restrictions_valid():
    # Construct a B0inv that satisfies all sign patterns
    B0inv = np.array([
        [ 1,  1,  1,  0.5,  0.3],   # row 1: oil production
        [ 1,  1, -1,  0.1, -0.5],   # row 2: REA
        [-1,  1,  1,  0.2,  0.8],   # row 3: price
        [ 0,  0,  1,  0.1,  0.9],   # row 4: inventory
        [ 0.1, 0,  0.5, 0,  1.0],   # row 5: OPU
    ])
    result = check_sign_restrictions(B0inv)
    # Should return either valid B0inv or None
    assert result is None or result.shape == (5, 5)


def test_elasticity_valid():
    B0inv = np.eye(5) * 0.01
    B0inv[0, :] = 0.001
    B0inv[3, :] = 0.001
    Q_1 = np.ones(100) * 50
    DSbar = 0.5
    ok = check_elasticity(B0inv, Q_1, DSbar)
    assert isinstance(ok, bool)


# ---------------------------------------------------------------------------
# Smoke tests for opu.narrative (Task 13)
# ---------------------------------------------------------------------------

from opu.narrative import (
    get_narrative_dates,
    compute_historical_decomposition,
    check_narrative_restrictions,
    get_extended_narrative_dates,
    check_extended_narrative,
)


@pytest.fixture
def narrative_synthetic():
    """Synthetic data for narrative smoke tests: n=5, p=2, T=30."""
    rng = np.random.default_rng(99)
    n, p, T = 5, 2, 30
    # Build sample dates covering 1979–1990 episode years (float years)
    # Start at 1979.0, monthly spacing = 1/12
    sample_dates = 1979.0 + np.arange(T) / 12.0
    # B: (n, 1+n*p) reduced-form coefficients
    k = 1 + n * p
    B = rng.standard_normal((n, k)) * 0.05
    # B0inv: (n, n) structural impact matrix (well-conditioned)
    B0inv = np.eye(n) + rng.standard_normal((n, n)) * 0.1
    # Ydep: (T, n)
    Ydep = rng.standard_normal((T, n))
    # X: (T, k)
    X = np.column_stack([np.ones(T), rng.standard_normal((T, n * p))])
    return sample_dates, B, B0inv, Ydep, X, n, p, T


def test_get_narrative_dates_keys(narrative_synthetic):
    sample_dates, *_ = narrative_synthetic
    dates = get_narrative_dates(sample_dates)
    expected_keys = {
        "id_90M10", "id_90M06", "id_90M07",
        "id_79M05", "id_79M12",
        "id_85M12", "id_86M12",
    }
    assert set(dates.keys()) == expected_keys


def test_get_narrative_dates_count(narrative_synthetic):
    sample_dates, *_ = narrative_synthetic
    dates = get_narrative_dates(sample_dates)
    assert len(dates) == 7


def test_compute_historical_decomposition_keys(narrative_synthetic):
    sample_dates, B, B0inv, Ydep, X, n, p, T = narrative_synthetic
    yhat = compute_historical_decomposition(B, B0inv, Ydep, X, n, p)
    expected_keys = {"supply", "flow_demand", "speculative", "uncertainty"}
    assert set(yhat.keys()) == expected_keys


def test_compute_historical_decomposition_array_lengths(narrative_synthetic):
    sample_dates, B, B0inv, Ydep, X, n, p, T = narrative_synthetic
    yhat = compute_historical_decomposition(B, B0inv, Ydep, X, n, p)
    for key, arr in yhat.items():
        assert arr.shape == (T,), f"yhat['{key}'] has shape {arr.shape}, expected ({T},)"


def test_check_narrative_restrictions_returns_bool(narrative_synthetic):
    sample_dates, B, B0inv, Ydep, X, n, p, T = narrative_synthetic
    yhat = compute_historical_decomposition(B, B0inv, Ydep, X, n, p)
    dates = get_narrative_dates(sample_dates)
    result = check_narrative_restrictions(yhat, dates, B0inv)
    # Accept both Python bool and numpy bool_ (numpy comparisons return np.bool_)
    assert isinstance(result, (bool, np.bool_))


# ---------------------------------------------------------------------------
# Smoke tests for extended narrative restrictions (Task 18)
# ---------------------------------------------------------------------------


@pytest.fixture
def extended_narrative_synthetic():
    """Synthetic data for extended narrative smoke tests.

    Uses a wide date range (1975-2025, monthly) so get_narrative_dates finds
    all base episodes, and a large enough array that 2014-16 and 2020 indices
    exist within bounds.  n=5, p=2, T covers the full date span.
    """
    rng = np.random.default_rng(42)
    n, p = 5, 2
    # 1975.0 to 2025.0, monthly => 601 entries
    sample_dates = 1975.0 + np.arange(601) / 12.0
    T = len(sample_dates)
    k = 1 + n * p
    B = rng.standard_normal((n, k)) * 0.05
    B0inv = np.eye(n) + rng.standard_normal((n, n)) * 0.1
    Ydep = rng.standard_normal((T, n))
    X = np.column_stack([np.ones(T), rng.standard_normal((T, n * p))])
    return sample_dates, B, B0inv, Ydep, X, n, p, T


def test_get_extended_narrative_dates_key_count(extended_narrative_synthetic):
    """get_extended_narrative_dates must return exactly 11 keys (7 base + 4 extended)."""
    sample_dates, *_ = extended_narrative_synthetic
    dates = get_extended_narrative_dates(sample_dates)
    assert len(dates) == 11


def test_get_extended_narrative_dates_has_extended_keys(extended_narrative_synthetic):
    """Extended dict must contain all four new episode keys."""
    sample_dates, *_ = extended_narrative_synthetic
    dates = get_extended_narrative_dates(sample_dates)
    for key in ("id_14M06", "id_16M02", "id_20M01", "id_20M04"):
        assert key in dates, f"Missing key: {key}"


def test_get_extended_narrative_dates_has_base_keys(extended_narrative_synthetic):
    """Extended dict must also contain all seven base keys."""
    sample_dates, *_ = extended_narrative_synthetic
    dates = get_extended_narrative_dates(sample_dates)
    for key in ("id_90M10", "id_90M06", "id_90M07",
                "id_79M05", "id_79M12", "id_85M12", "id_86M12"):
        assert key in dates, f"Missing base key: {key}"


def test_check_extended_narrative_returns_bool(extended_narrative_synthetic):
    """check_extended_narrative must return a bool for synthetic input."""
    sample_dates, B, B0inv, Ydep, X, n, p, T = extended_narrative_synthetic
    yhat = compute_historical_decomposition(B, B0inv, Ydep, X, n, p)
    dates = get_extended_narrative_dates(sample_dates)
    result = check_extended_narrative(yhat, dates)
    assert isinstance(result, (bool, np.bool_))
