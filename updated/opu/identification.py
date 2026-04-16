"""SVAR identification restrictions. Ports shuffle_mod_unc_nosup.m."""
import numpy as np
from opu.config import SVAR_ELASTICITY_BOUND, SVAR_USE_ELASTICITY_BOUNDS


def check_sign_restrictions(Atilde: np.ndarray) -> np.ndarray | None:
    """Check static sign restrictions and reorder columns.

    Returns reordered B0inv if valid, None otherwise.
    Tries Atilde first, then -Atilde.
    """
    for sign in [1, -1]:
        A = sign * Atilde
        result = _try_assign(A)
        if result is not None:
            return result
    return None


def _try_assign(A: np.ndarray) -> np.ndarray | None:
    n = A.shape[0]
    max_col5 = np.max(A[4, :])
    assignments = {}
    used_cols = set()

    for col in range(n):
        shock = _classify_column(A, col, max_col5)
        if shock is not None and shock not in assignments:
            assignments[shock] = col
            used_cols.add(col)

    # Need exactly: supply(100), flow_demand(1), speculative(10), uncertainty(1000)
    required = {100, 1, 10, 1000}
    if not required.issubset(assignments.keys()):
        return None

    # Remaining column is residual (0)
    residual_cols = set(range(n)) - used_cols
    if len(residual_cols) != 1:
        return None
    assignments[0] = residual_cols.pop()

    # Build B0inv: col 0=supply, 1=flow demand, 2=speculative, 3=residual, 4=uncertainty
    shock_to_col = {100: 0, 1: 1, 10: 2, 0: 3, 1000: 4}
    B0inv = np.zeros((n, n))
    for shock_code, src_col in assignments.items():
        B0inv[:, shock_to_col[shock_code]] = A[:, src_col]

    return B0inv


def _classify_column(A: np.ndarray, col: int, max_col5: float) -> int | None:
    """Classify a column of Atilde as a shock type.

    Returns shock code: 100=supply, 1=flow_demand, 10=speculative, 1000=uncertainty, None=unclassified.
    """
    a1, a2, a3, a4, a5 = A[0, col], A[1, col], A[2, col], A[3, col], A[4, col]

    # Supply: +,+,-,?,? with OPU impact < max
    if a1 > 0 and a2 > 0 and a3 < 0 and a5 < max_col5:
        return 100

    # Speculative demand: +,-,+,+,? with OPU < max
    if a1 > 0 and a2 < 0 and a3 > 0 and a4 > 0 and a5 < max_col5:
        return 10

    # Flow demand: +,+,+,?,?
    if a1 > 0 and a2 > 0 and a3 > 0:
        return 1

    # Uncertainty demand: ?,-,+,+,+ with OPU == max
    if a2 < 0 and a3 > 0 and a4 > 0 and a5 > 0 and a5 == max_col5:
        return 1000

    return None


def check_elasticity(B0inv: np.ndarray, Q_1: np.ndarray, DSbar: float) -> bool:
    """Check elasticity bounds (S2 restriction)."""
    # Supply response to speculative demand
    if B0inv[2, 2] == 0:
        return False
    eta_s1a = B0inv[0, 2] / B0inv[2, 2]

    # Supply response to uncertainty demand
    if B0inv[2, 4] == 0:
        return False
    eta_s1b = B0inv[0, 4] / B0inv[2, 4]

    # Supply response to flow demand
    if B0inv[2, 1] == 0:
        return False
    eta_s2 = B0inv[0, 1] / B0inv[2, 1]

    if eta_s1a >= SVAR_ELASTICITY_BOUND:
        return False
    if eta_s1b >= SVAR_ELASTICITY_BOUND:
        return False
    if eta_s2 >= SVAR_ELASTICITY_BOUND:
        return False

    # Oil-in-use elasticity
    if B0inv[2, 0] == 0:
        return False
    eta_use_t = ((Q_1 * B0inv[0, 0] - B0inv[3, 0]) / (Q_1 - DSbar)) / B0inv[2, 0]
    eta_use = np.mean(eta_use_t[:len(Q_1)])
    lo, hi = SVAR_USE_ELASTICITY_BOUNDS
    if not (lo < eta_use < hi):
        return False

    return True


def compute_irf(B: np.ndarray, B0inv: np.ndarray, n: int, p: int, H: int) -> np.ndarray:
    """Compute structural IRFs. Port of IRvar.m.

    Returns (n, n, H+1) array: ir[i, j, h] = response of var i to shock j at horizon h.
    """
    A = np.zeros((n * p, n * p))
    A[:n, :] = B[:, 1:]
    if p > 1:
        A[n:, : n * (p - 1)] = np.eye(n * (p - 1))

    J = np.zeros((n, n * p))
    J[:n, :n] = np.eye(n)

    ir = np.zeros((n, n, H + 1))
    for h in range(H + 1):
        ir[:, :, h] = J @ np.linalg.matrix_power(A, h) @ J.T @ B0inv
    return ir


def check_dynamic_sign(ir: np.ndarray) -> bool:
    """Check dynamic sign restrictions (S3): cumulative IRFs maintain sign for 12 months."""
    cum_ir = np.cumsum(ir, axis=2)

    # Supply shock (col 0): oil prod cumulative >=0, REA cumulative >=0, price cumulative <=0
    if not np.all(cum_ir[0, 0, :12] >= 0):
        return False
    if not np.all(cum_ir[1, 0, :12] >= 0):
        return False
    if not np.all(cum_ir[2, 0, :12] <= 0):
        return False

    # Speculative demand (col 2): inventory cumulative >=0, price cumulative >=0
    if not np.all(cum_ir[3, 2, :12] >= 0):
        return False
    if not np.all(cum_ir[2, 2, :12] >= 0):
        return False

    # Uncertainty demand (col 4): inventory cumulative >=0, price cumulative >=0,
    # REA cumulative <=0
    if not np.all(cum_ir[3, 4, :12] >= 0):
        return False
    if not np.all(cum_ir[2, 4, :12] >= 0):
        return False
    if not np.all(cum_ir[1, 4, :12] <= 0):
        return False

    return True
