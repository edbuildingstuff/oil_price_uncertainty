import numpy as np
import pytest
from opu.sv import sv_sample, KSC_MEANS, KSC_VARS, KSC_WEIGHTS


def test_ksc_mixture_constants():
    assert len(KSC_MEANS) == 10
    assert len(KSC_VARS) == 10
    assert len(KSC_WEIGHTS) == 10
    np.testing.assert_almost_equal(sum(KSC_WEIGHTS), 1.0, decimal=5)


def test_sv_sample_shape():
    np.random.seed(42)
    T = 200
    y = np.random.randn(T) * np.exp(0.5 * np.sin(np.arange(T) / 20))
    result = sv_sample(y, draws=500, burnin=500, thin=1, seed=42)
    assert "mu" in result
    assert "phi" in result
    assert "sigma" in result
    assert "latent" in result
    assert result["latent"].shape == (T,)


def test_sv_sample_recovers_parameters():
    np.random.seed(0)
    T = 500
    mu_true, phi_true, sigma_true = -0.5, 0.95, 0.2
    h = np.zeros(T)
    h[0] = mu_true
    for t in range(1, T):
        h[t] = mu_true + phi_true * (h[t - 1] - mu_true) + sigma_true * np.random.randn()
    y = np.exp(h / 2) * np.random.randn(T)

    result = sv_sample(y, draws=5000, burnin=5000, thin=5, seed=0)
    assert abs(result["mu"] - mu_true) < 1.0
    assert abs(result["phi"] - phi_true) < 0.15
    assert abs(result["sigma"] - sigma_true) < 0.15


def test_sv_sample_against_reference():
    """Compare against original stochvol output on overlapping data."""
    from pathlib import Path
    ref_path = Path("reference/Oilsvymeans_baseline.txt")
    if not ref_path.exists():
        pytest.skip("Reference data not available")
    ref = np.loadtxt(ref_path)
    # ref format: row 0 = mu, row 1 = phi, row 2 = sigma, rows 3:-3 = latent, last 3 = Geweke
    ref_mu = ref[0]
    ref_phi = ref[1]
    ref_sigma = ref[2]

    vyt_path = Path("reference/Oilyt_baseline.txt")
    if not vyt_path.exists():
        pytest.skip("Reference forecast errors not available")
    vyt = np.loadtxt(vyt_path)

    result = sv_sample(vyt, draws=50000, burnin=50000, thin=10, seed=0)

    # Tier 3: within ~2 posterior SDs (rough)
    assert abs(result["mu"] - ref_mu) < 1.0
    assert abs(result["phi"] - ref_phi) < 0.1
    assert abs(result["sigma"] - ref_sigma) < 0.2
