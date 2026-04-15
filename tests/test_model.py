"""Tests for AutoEncoder and MEDAL model."""
import numpy as np
import pytest
import torch

from medal.model import AutoEncoder, MEDAL


class TestAutoEncoder:
    def test_forward_shape(self):
        model = AutoEncoder(input_dim=20, latent_dim=2, hidden_dims=(32, 16))
        x = torch.randn(8, 20)
        x_recon, z = model(x)
        assert x_recon.shape == (8, 20)
        assert z.shape == (8, 2)

    def test_different_activations(self):
        import torch.nn as nn
        for act in (nn.ReLU, nn.SELU, None):
            model = AutoEncoder(
                input_dim=10, latent_dim=2, hidden_dims=(16,),
                activation=act,
            )
            x = torch.randn(4, 10)
            x_recon, z = model(x)
            assert x_recon.shape == (4, 10)

    def test_batchnorm(self):
        model = AutoEncoder(
            input_dim=10, latent_dim=2, hidden_dims=(16, 8),
            use_batchnorm=True,
        )
        x = torch.randn(16, 10)
        x_recon, z = model(x)
        assert x_recon.shape == (16, 10)


class TestMEDAL:
    def test_fit_transform_shapes(self, small_X, small_Z):
        student = MEDAL(
            input_dim=20, latent_dim=2, hidden_dims=(32, 16),
            epochs=5, batch_size=64,
        )
        student.fit(small_X, small_Z, verbose=False)
        Z_pred = student.transform(small_X)
        assert Z_pred.shape == (200, 2)

    def test_reconstruct_shape(self, small_X, small_Z):
        student = MEDAL(
            input_dim=20, latent_dim=2, hidden_dims=(32,),
            epochs=3, batch_size=64,
        )
        student.fit(small_X, small_Z, verbose=False)
        X_recon = student.reconstruct(small_X)
        assert X_recon.shape == small_X.shape

    def test_fit_transform_convenience(self, small_X, small_Z):
        student = MEDAL(
            input_dim=20, latent_dim=2, hidden_dims=(32,),
            epochs=3, batch_size=64,
        )
        Z = student.fit_transform(small_X, small_Z, verbose=False)
        assert Z.shape == (200, 2)

    def test_no_pretrain_param(self):
        """Ensure pretrain/phase interface has been fully removed."""
        import inspect
        sig = inspect.signature(MEDAL.fit)
        assert "phase" not in sig.parameters
        assert "pretrained_path" not in sig.parameters
