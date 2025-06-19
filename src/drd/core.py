import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset
from src.drd.loss import get_loss_function
    
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10, hidden_dims=(128, 128), activation=nn.ReLU):
        super(AutoEncoder, self).__init__()

        # build encoder: input -> hidden_dims... -> latent
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            # encoder_layers.append(nn.BatchNorm1d(h))
            encoder_layers.append(activation())
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # build decoder: latent -> reversed(hidden_dims)... -> output
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            # decoder_layers.append(nn.BatchNorm1d(h))
            decoder_layers.append(activation())
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_identity()

    def _init_identity(self, eps=1e-3):
        # initialize each residual block close to identity
        for m in self.modules():
            if isinstance(m, nn.Linear):
                try:
                    nn.init.eye_(m.weight)
                except Exception:
                    # non-square weight: fallback to small random
                    nn.init.normal_(m.weight, mean=0.0, std=eps)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # produce latent and reconstruction
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class DRD(BaseEstimator, TransformerMixin):
    def __init__(self, input_dim, latent_dim=2, hidden_dims=(128, 64), activation="ReLU",
                 lambda_kl = 0, lambda_d = 10, lambda_reg=0, lr=1e-3, epochs=100, batch_size=32, 
                 device=None, clip_grad_norm=1.0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation = getattr(nn, activation) if isinstance(activation, str) else activation
        self.lambda_kl = lambda_kl
        self.lambda_d = lambda_d
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_grad_norm = clip_grad_norm

        self.model = AutoEncoder(self.input_dim, self.latent_dim, self.hidden_dims, activation = self.activation).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.criterion = get_loss_function(lambda_kl=self.lambda_kl, lambda_d=self.lambda_d)
        self.criterion = nn.MSELoss()

    def fit(self, X, y=None, teacher_Z=None, verbose=True):

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if teacher_Z is not None:
            teacher_Z = torch.tensor(teacher_Z, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X, teacher_Z) if teacher_Z is not None else TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            total_loss = 0
            for batch in loader:
                self.optimizer.zero_grad()
                x_batch = batch[0]
                teacher_z_batch = batch[1] if teacher_Z is not None else None

                x_batch_recon, student_z_batch = self.model(x_batch)

                recon = self.criterion(x_batch, x_batch_recon)

                if teacher_z_batch is not None:
                    distill_loss = self.criterion(student_z_batch, teacher_z_batch)
                    loss = recon + self.lambda_d * distill_loss
                else:
                    loss = recon

                loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()
            
            if verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(loader)}")
        
        return self