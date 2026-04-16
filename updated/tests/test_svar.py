import numpy as np
import pytest
from opu.svar import load_svar_data


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
