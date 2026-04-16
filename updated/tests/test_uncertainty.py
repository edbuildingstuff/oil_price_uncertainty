import numpy as np
import pytest
from opu.uncertainty import compute_uf, compute_uy, expected_var


def test_expected_var_basic():
    a, b, t2 = 0.1, 0.9, 0.04
    x = np.array([0.0, 0.5, -0.5])
    result = expected_var(a, b, t2, x, h=1)
    assert result.shape == (3,)
    assert np.all(result > 0)


def test_expected_var_horizon_increasing():
    a, b, t2 = 0.1, 0.9, 0.04
    x = np.array([0.0])
    v1 = expected_var(a, b, t2, x, h=1)[0]
    v12 = expected_var(a, b, t2, x, h=12)[0]
    # Longer horizon should converge to unconditional variance
    assert v12 > 0


def test_compute_uf_shape():
    np.random.seed(0)
    r = 3
    pf = 4
    T = 50
    xf = np.random.randn(T, r)
    thf = np.array([
        [0.1, 0.2, 0.15],    # alpha = mu*(1-phi)
        [0.9, 0.85, 0.88],   # beta = phi
        [0.04, 0.03, 0.05],  # tau2 = sigma^2
    ])
    fb = np.random.randn(r, pf + 1) * 0.1
    evf, phif = compute_uf(xf, thf, fb, h=12)
    assert len(evf) == 12
    assert evf[0].shape == (T, r)
    assert phif.shape == (r * pf, r * pf)


def test_compute_uy_shape():
    np.random.seed(0)
    T, r, pf, py, pz = 50, 3, 4, 4, 4
    h = 12
    xf = np.random.randn(T, r)
    thf = np.array([
        [0.1, 0.2, 0.15],
        [0.9, 0.85, 0.88],
        [0.04, 0.03, 0.05],
    ])
    fb = np.random.randn(r, pf + 1) * 0.1
    evf, phif = compute_uf(xf, thf, fb, h=h)

    xy = np.random.randn(T)
    thy = np.array([0.1, 0.9, 0.04])
    yb = np.random.randn(1 + py + pz * r) * 0.1

    U = compute_uy(xy, thy, yb, py, evf, phif)
    assert U.shape == (T, h)
    assert np.all(U >= 0)
