"""Tests for teacher embedding utilities."""
import numpy as np
import pytest

from medal.teacher import build_param_grid, get_teacher_embeddings


class TestBuildParamGrid:
    def test_tsne_default(self):
        grid = build_param_grid("tsne", n_components=2)
        assert len(grid) > 0
        for item in grid:
            assert "perplexity" in item
            assert item["n_components"] == 2

    def test_umap_default(self):
        grid = build_param_grid("umap", n_components=2)
        for item in grid:
            assert "n_neighbors" in item
            assert "min_dist" in item

    def test_custom_grid(self):
        custom = [{"perplexity": 5}, {"perplexity": 30}]
        grid = build_param_grid("tsne", custom_grid=custom)
        assert len(grid) == 2
        assert grid[0]["perplexity"] == 5

    def test_pca_no_param(self):
        grid = build_param_grid("pca", n_components=2)
        assert len(grid) == 1
        assert grid[0]["n_components"] == 2


class TestGetTeacherEmbeddings:
    def test_pca(self, small_X):
        Z = get_teacher_embeddings("pca", small_X, n_components=2)
        assert Z.shape == (200, 2)
        assert np.isfinite(Z).all()

    def test_tsne(self, small_X):
        Z = get_teacher_embeddings(
            "tsne", small_X[:100], n_components=2, perplexity=10,
            random_state=0,
        )
        assert Z.shape == (100, 2)
        assert np.isfinite(Z).all()
