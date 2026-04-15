"""Tests for teacher-hyperparameter selection."""
import numpy as np
import pandas as pd
import pytest

from medal.selection import select_teacher_param


def _make_df(param_col, param_values, seeds, rng):
    """Build a synthetic SweepResults-style DataFrame."""
    rows = []
    for p in param_values:
        for seed in seeds:
            for split in ("Train", "Val", "Test"):
                # Lowest loss at param_values[2] = 30
                base = abs(p - 30) * 0.01 + 0.1
                rows.append({
                    param_col: p,
                    "seed": seed,
                    "split": split,
                    "recon_loss": base + rng.normal(0, 0.005),
                    "recon_mse": base + rng.normal(0, 0.005),
                    "distill_mse": base * 0.5,
                })
    return pd.DataFrame(rows)


class TestSelectTeacherParam:
    def test_selects_smallest_within_one_sem(self):
        rng = np.random.default_rng(42)
        df = _make_df("perplexity", [5, 15, 30, 50, 100], seeds=range(5), rng=rng)
        opt = select_teacher_param(df, param_col="perplexity")
        # Should select something near 30 (the true minimum), not the largest value
        assert opt in [5, 15, 30, 50, 100]
        assert opt <= 30

    def test_returns_scalar(self):
        rng = np.random.default_rng(0)
        df = _make_df("n_neighbors", [5, 10, 20], seeds=[0], rng=rng)
        opt = select_teacher_param(df, param_col="n_neighbors")
        assert np.isscalar(opt) or hasattr(opt, "item")

    def test_single_param_value(self):
        rng = np.random.default_rng(0)
        df = _make_df("perplexity", [30], seeds=[0, 1], rng=rng)
        opt = select_teacher_param(df, param_col="perplexity")
        assert opt == 30
