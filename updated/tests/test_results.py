"""Smoke tests for opu.results: IRFs, FEVDs, compute_all_results."""
import numpy as np
import pytest
from opu.results import compute_irfs, irf_summary, compute_fevd, compute_all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_draws(rng, M: int, n: int, p: int) -> list:
    """Build M fake accepted draws with random B, Sigma, B0inv."""
    k = 1 + n * p
    draws = []
    for _ in range(M):
        B = rng.standard_normal((n, k)) * 0.05
        Sigma = np.eye(n)
        B0inv = np.eye(n) + rng.standard_normal((n, n)) * 0.1
        draws.append({"B": B, "Sigma": Sigma, "B0inv": B0inv})
    return draws


# ---------------------------------------------------------------------------
# Tests for compute_irfs
# ---------------------------------------------------------------------------

def test_compute_irfs_output_shape():
    rng = np.random.default_rng(0)
    n, p, H, M = 5, 2, 5, 3
    draws = _make_draws(rng, M, n, p)
    irfs = compute_irfs(draws, n, p, H=H)
    assert irfs.shape == (M, n, n, H + 1), (
        f"Expected ({M}, {n}, {n}, {H + 1}), got {irfs.shape}"
    )


def test_compute_irfs_returns_ndarray():
    rng = np.random.default_rng(1)
    n, p, H, M = 5, 2, 5, 3
    draws = _make_draws(rng, M, n, p)
    irfs = compute_irfs(draws, n, p, H=H)
    assert isinstance(irfs, np.ndarray)


def test_compute_irfs_different_draws_differ():
    rng = np.random.default_rng(2)
    n, p, H, M = 5, 2, 5, 3
    draws = _make_draws(rng, M, n, p)
    irfs = compute_irfs(draws, n, p, H=H)
    # IRFs for different draws should generally not be identical
    assert not np.allclose(irfs[0], irfs[1]) or not np.allclose(irfs[1], irfs[2])


# ---------------------------------------------------------------------------
# Tests for irf_summary
# ---------------------------------------------------------------------------

def test_irf_summary_output_shape():
    rng = np.random.default_rng(3)
    M, n, H = 10, 5, 5
    irfs = rng.standard_normal((M, n, n, H + 1))
    stats = irf_summary(irfs)
    for key in ("median", "lower", "upper"):
        assert stats[key].shape == (n, n, H + 1), (
            f"stats['{key}'] shape {stats[key].shape} != ({n}, {n}, {H + 1})"
        )


def test_irf_summary_keys():
    rng = np.random.default_rng(4)
    irfs = rng.standard_normal((5, 3, 3, 6))
    stats = irf_summary(irfs)
    assert set(stats.keys()) == {"median", "lower", "upper"}


def test_irf_summary_ordering():
    """lower <= median <= upper at every element."""
    rng = np.random.default_rng(5)
    irfs = rng.standard_normal((20, 5, 5, 6))
    stats = irf_summary(irfs)
    assert np.all(stats["lower"] <= stats["median"])
    assert np.all(stats["median"] <= stats["upper"])


# ---------------------------------------------------------------------------
# Tests for compute_fevd
# ---------------------------------------------------------------------------

def test_compute_fevd_output_shape():
    rng = np.random.default_rng(6)
    n, p, H, M = 5, 2, 5, 2
    draws = _make_draws(rng, M, n, p)
    fevd = compute_fevd(draws, n, p, H=H)
    assert fevd.shape == (M, n, n, H + 1), (
        f"Expected ({M}, {n}, {n}, {H + 1}), got {fevd.shape}"
    )


def test_compute_fevd_values_in_unit_interval():
    rng = np.random.default_rng(7)
    n, p, H, M = 5, 2, 5, 2
    draws = _make_draws(rng, M, n, p)
    fevd = compute_fevd(draws, n, p, H=H)
    assert np.all(fevd >= 0.0), "FEVD values must be >= 0"
    assert np.all(fevd <= 1.0 + 1e-9), "FEVD values must be <= 1"


def test_compute_fevd_rows_sum_to_one():
    """For each draw m, variable i, and horizon h, shares across shocks j sum to 1."""
    rng = np.random.default_rng(8)
    n, p, H, M = 5, 2, 5, 2
    draws = _make_draws(rng, M, n, p)
    fevd = compute_fevd(draws, n, p, H=H)
    # Sum over j (axis=2): fevd[m, i, :, h].sum() should == 1 for all m, i, h
    row_sums = fevd.sum(axis=2)  # shape (M, n, H+1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), (
        f"FEVD rows do not sum to 1; max deviation: {np.max(np.abs(row_sums - 1.0))}"
    )


# ---------------------------------------------------------------------------
# Integration smoke test for compute_all_results (uses tmp_path + monkeypatch)
# ---------------------------------------------------------------------------

def test_compute_all_results_writes_expected_keys(tmp_path, monkeypatch):
    """Fabricate accepted_draws.npz in a temp dir, patch SVAR_DIR, run compute_all_results."""
    rng = np.random.default_rng(42)
    n, p, M = 5, 2, 3
    k = 1 + n * p

    # Build fake accepted draws arrays as stored by the SVAR sampler
    vecB_arr = np.zeros((M, n * k))
    Sigma_arr = np.zeros((M, n * n))
    B0inv_arr = np.zeros((M, n * n))

    for i in range(M):
        B = rng.standard_normal((n, k)) * 0.05
        # vecB stored in Fortran order: B.T.flatten(order='F') -> reshape(1+n*p, n, order='F').T == B
        vecB_arr[i] = B.T.flatten(order="F")
        Sigma_arr[i] = np.eye(n).flatten()
        B0inv_arr[i] = (np.eye(n) + rng.standard_normal((n, n)) * 0.1).flatten()

    # Write fake accepted_draws.npz
    np.savez(
        tmp_path / "accepted_draws.npz",
        vecB=vecB_arr,
        Sigma=Sigma_arr,
        B0inv=B0inv_arr,
        n=np.array(n),
        p=np.array(p),
    )

    # Patch SVAR_DIR in both opu.config and opu.results
    import opu.config as cfg
    import opu.results as res

    monkeypatch.setattr(cfg, "SVAR_DIR", tmp_path)
    monkeypatch.setattr(res, "SVAR_DIR", tmp_path)

    compute_all_results()

    out_path = tmp_path / "results.npz"
    assert out_path.exists(), "results.npz was not created"

    data = np.load(out_path)
    expected_keys = {
        "irf_median", "irf_lower", "irf_upper",
        "fevd_median", "fevd_lower", "fevd_upper",
    }
    assert set(data.files) == expected_keys, (
        f"Missing keys: {expected_keys - set(data.files)}"
    )
