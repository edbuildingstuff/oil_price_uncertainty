"""EM factor extraction. Port of factors_em.m (McCracken & Ng, 2017)."""
import numpy as np


def factors_em(
    x: np.ndarray, kmax: int = 1, jj: int = 2, demean: int = 2, maxit: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate factors via principal components with EM for missing values.

    Returns: (ehat, Fhat, lamhat, eigenvalues, x_filled)
    """
    T, N = x.shape
    x1 = np.isnan(x)

    # Initialize: fill missing with column means
    col_means = np.nanmean(x, axis=0)
    x2 = x.copy()
    for j in range(N):
        x2[x1[:, j], j] = col_means[j]

    x3, mut, sdt = _transform_data(x2, demean)

    if kmax != 99:
        icstar = _baing(x3, kmax, jj)
    else:
        icstar = 8

    chat, Fhat, lamhat, ve2 = _pc2(x3, icstar)
    chat0 = chat.copy()

    err = 999.0
    it = 0
    while err > 1e-6 and it < maxit:
        it += 1

        # Update missing values
        for t in range(T):
            for j in range(N):
                if x1[t, j]:
                    x2[t, j] = chat[t, j] * sdt[t, j] + mut[t, j]
                else:
                    x2[t, j] = x[t, j]

        x3, mut, sdt = _transform_data(x2, demean)

        if kmax != 99:
            icstar = _baing(x3, kmax, jj)
        else:
            icstar = 8

        chat, Fhat, lamhat, ve2 = _pc2(x3, icstar)

        diff = chat - chat0
        v1 = diff.ravel()
        v2 = chat0.ravel()
        err = (v1 @ v1) / (v2 @ v2) if (v2 @ v2) > 0 else 0.0
        chat0 = chat.copy()

    ehat = x - chat * sdt + mut
    return ehat, Fhat, lamhat, ve2, x2


def _transform_data(x2: np.ndarray, demean: int):
    T, N = x2.shape
    if demean == 0:
        mut = np.zeros((T, N))
        sdt = np.ones((T, N))
        x22 = x2.copy()
    elif demean == 1:
        mut = np.tile(np.mean(x2, axis=0), (T, 1))
        sdt = np.ones((T, N))
        x22 = x2 - mut
    elif demean == 2:
        mut = np.tile(np.mean(x2, axis=0), (T, 1))
        sdt = np.tile(np.std(x2, axis=0, ddof=0), (T, 1))
        sdt[sdt == 0] = 1.0
        x22 = (x2 - mut) / sdt
    elif demean == 3:
        mut = np.empty((T, N))
        for t in range(T):
            mut[t, :] = np.mean(x2[: t + 1, :], axis=0)
        sdt = np.tile(np.std(x2, axis=0, ddof=0), (T, 1))
        sdt[sdt == 0] = 1.0
        x22 = (x2 - mut) / sdt
    else:
        raise ValueError(f"Invalid demean code: {demean}")
    return x22, mut, sdt


def _pc2(X: np.ndarray, nfac: int):
    N = X.shape[1]
    U, S, Vt = np.linalg.svd(X.T @ X, full_matrices=False)
    lamhat = U[:, :nfac] * np.sqrt(N)
    fhat = X @ lamhat / N
    chat = fhat @ lamhat.T
    return chat, fhat, lamhat, S


def _baing(X: np.ndarray, kmax: int, jj: int) -> int:
    T, N = X.shape
    NT = N * T
    NT1 = N + T

    ii = np.arange(1, kmax + 1)
    if jj == 1:
        CT = np.log(NT / NT1) * ii * NT1 / NT
    elif jj == 2:
        CT = (NT1 / NT) * np.log(min(N, T)) * ii
    elif jj == 3:
        GCT = min(N, T)
        CT = ii * np.log(GCT) / GCT
    else:
        raise ValueError(f"Invalid jj: {jj}")

    if T < N:
        ev, eigval, _ = np.linalg.svd(X @ X.T)
        Fhat0 = ev * np.sqrt(T)
        Lambda0 = X.T @ Fhat0 / T
    else:
        ev, eigval, _ = np.linalg.svd(X.T @ X)
        Lambda0 = ev * np.sqrt(N)
        Fhat0 = X @ Lambda0 / N

    Sigma = np.zeros(kmax + 1)
    IC1 = np.zeros(kmax + 1)

    for i in range(kmax, 0, -1):
        Fhat = Fhat0[:, :i]
        lam = Lambda0[:, :i]
        chat = Fhat @ lam.T
        ehat = X - chat
        Sigma[i - 1] = np.mean(np.sum(ehat * ehat / T, axis=0))
        IC1[i - 1] = np.log(Sigma[i - 1]) + CT[i - 1]

    Sigma[kmax] = np.mean(np.sum(X * X / T, axis=0))
    IC1[kmax] = np.log(Sigma[kmax])

    ic1 = int(np.argmin(IC1))
    if ic1 >= kmax:
        ic1 = 0
    else:
        ic1 += 1

    return max(ic1, 1)
