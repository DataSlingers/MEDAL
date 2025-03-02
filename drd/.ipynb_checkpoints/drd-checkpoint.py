import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset
from drd.loss import get_loss_function

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10, hidden_dims=(128, 64), activation=nn.ReLU):
        super(AutoEncoder, self).__init__()
        self.encoder = self._build_layers(input_dim, hidden_dims, latent_dim, activation, encode=True)
        self.decoder = self._build_layers(latent_dim, reversed(hidden_dims), input_dim, activation, encode=False)

    def _build_layers(self, in_dim, hidden_dims, out_dim, activation, encode):
        layers = []
        prev_dim = in_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, out_dim))
        # if not encode:
        #     layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class DRD(BaseEstimator, TransformerMixin):
    def __init__(self, input_dim, latent_dim=10, hidden_dims=(128, 64), activation="ReLU",
                 lambda_kl = 0, lambda_d = 0, lr=1e-3, epochs=50, batch_size=32, device=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation = getattr(nn, activation) if isinstance(activation, str) else activation
        self.lambda_kl = lambda_kl
        self.lambda_d = lambda_d
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._build_model()

    def _build_model(self):
        self.model = AutoEncoder(self.input_dim, self.latent_dim, self.hidden_dims, self.activation).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = get_loss_function(lambda_kl=self.lambda_kl, lambda_d=self.lambda_d)

    def fit(self, X, y=None, teacher_Z=None, verbose=False):

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
                if teacher_Z is not None:
                    x_batch, teacher_z_batch = batch
                else:
                    x_batch = batch[0]
                    teacher_z_batch = None

                x_batch_recon, student_z_batch = self.model(x_batch)

                loss = self.criterion(x_batch, x_batch_recon, student_z_batch, teacher_z_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(loader)}")

    def transform(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, z = self.model(X)
        return z.cpu().numpy()

    def inverse_transform(self, Z):
        self.model.eval()
        Z = torch.tensor(Z, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            X_recon = self.model.decoder(Z)
        return X_recon.cpu().numpy()
    
    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))
    
# if __name__ == "__main__":
