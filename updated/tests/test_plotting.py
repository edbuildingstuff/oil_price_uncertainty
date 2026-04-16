"""Smoke tests for opu.plotting: plot_irfs and plot_fevds with synthetic data."""
import numpy as np
import pytest


def _make_results_npz(tmp_path, n=5, H=17):
    """Create a synthetic results.npz with the shapes plot_irfs / plot_fevds expect."""
    rng = np.random.default_rng(0)
    shape = (n, n, H + 1)

    irf_median = rng.standard_normal(shape)
    irf_lower = irf_median - np.abs(rng.standard_normal(shape)) * 0.1
    irf_upper = irf_median + np.abs(rng.standard_normal(shape)) * 0.1

    # FEVD: each row (variable i) shares across shocks must sum to 1
    raw = np.abs(rng.standard_normal(shape)) + 0.01
    fevd_median = raw / raw.sum(axis=1, keepdims=True)
    fevd_lower = fevd_median * 0.9
    fevd_upper = np.minimum(fevd_median * 1.1, 1.0)

    np.savez(
        tmp_path / "results.npz",
        irf_median=irf_median,
        irf_lower=irf_lower,
        irf_upper=irf_upper,
        fevd_median=fevd_median,
        fevd_lower=fevd_lower,
        fevd_upper=fevd_upper,
    )


def test_plot_irfs_creates_pdf(tmp_path, monkeypatch):
    """plot_irfs writes fig5_irfs.pdf to FIGURES_DIR."""
    import opu.plotting as plotting

    _make_results_npz(tmp_path)

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    monkeypatch.setattr(plotting, "SVAR_DIR", tmp_path)
    monkeypatch.setattr(plotting, "FIGURES_DIR", figures_dir)

    plotting.plot_irfs()

    assert (figures_dir / "fig5_irfs.pdf").exists(), "fig5_irfs.pdf was not created"


def test_plot_fevds_creates_pdf(tmp_path, monkeypatch):
    """plot_fevds writes fig6_fevds.pdf to FIGURES_DIR."""
    import opu.plotting as plotting

    _make_results_npz(tmp_path)

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    monkeypatch.setattr(plotting, "SVAR_DIR", tmp_path)
    monkeypatch.setattr(plotting, "FIGURES_DIR", figures_dir)

    plotting.plot_fevds()

    assert (figures_dir / "fig6_fevds.pdf").exists(), "fig6_fevds.pdf was not created"


def test_plot_irfs_grid_dimensions(tmp_path, monkeypatch):
    """plot_irfs produces a 5x5 grid figure (n=5 variables x n=5 shocks)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import opu.plotting as plotting

    _make_results_npz(tmp_path, n=5, H=17)

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    monkeypatch.setattr(plotting, "SVAR_DIR", tmp_path)
    monkeypatch.setattr(plotting, "FIGURES_DIR", figures_dir)

    # Capture figure before close
    created_figs = []
    orig_savefig = matplotlib.figure.Figure.savefig

    def patched_savefig(self, *args, **kwargs):
        created_figs.append(self)
        return orig_savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", patched_savefig)
    plotting.plot_irfs()

    assert len(created_figs) >= 1
    fig = created_figs[0]
    axes = fig.get_axes()
    assert len(axes) == 25, f"Expected 25 subplots (5x5), got {len(axes)}"


def test_plot_fevds_grid_dimensions(tmp_path, monkeypatch):
    """plot_fevds produces a 1x5 grid figure (1 row x n=5 variables)."""
    import matplotlib
    matplotlib.use("Agg")
    import opu.plotting as plotting

    _make_results_npz(tmp_path, n=5, H=17)

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    monkeypatch.setattr(plotting, "SVAR_DIR", tmp_path)
    monkeypatch.setattr(plotting, "FIGURES_DIR", figures_dir)

    created_figs = []
    orig_savefig = matplotlib.figure.Figure.savefig

    def patched_savefig(self, *args, **kwargs):
        created_figs.append(self)
        return orig_savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", patched_savefig)
    plotting.plot_fevds()

    assert len(created_figs) >= 1
    fig = created_figs[0]
    axes = fig.get_axes()
    assert len(axes) == 5, f"Expected 5 subplots (1x5), got {len(axes)}"


def test_plot_opu_events_creates_files(tmp_path, monkeypatch):
    """plot_opu_events writes fig1_opu_events.pdf and .png."""
    import opu.plotting as plotting

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    monkeypatch.setattr(plotting, "FIGURES_DIR", figures_dir)
    # OPU_DIR is left pointing to the real artifacts/opu — real data is available

    plotting.plot_opu_events()

    assert (figures_dir / "fig1_opu_events.pdf").exists()
    assert (figures_dir / "fig1_opu_events.png").exists()


def test_plot_opu_comparison_creates_file(tmp_path, monkeypatch):
    """plot_opu_comparison writes fig2_opu_comparison.pdf."""
    import opu.plotting as plotting

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    monkeypatch.setattr(plotting, "FIGURES_DIR", figures_dir)

    plotting.plot_opu_comparison()

    assert (figures_dir / "fig2_opu_comparison.pdf").exists()
