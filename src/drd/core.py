import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from ray import tune

from torch.utils.data import DataLoader, TensorDataset
from src.drd.loss import get_loss_function
    
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10, hidden_dims=(128, 128), activation=nn.ReLU, bottleneck_activation=None):
        super().__init__()

        encoder_layers = []
        prev_dim = input_dim
        for e_id, h in enumerate(hidden_dims):                
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(activation())
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        if bottleneck_activation is not None:
            encoder_layers.append(bottleneck_activation())
        print("encoder layers:", encoder_layers)
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(activation())
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        print("decoder layers:", decoder_layers)
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
                 bottleneck_activation = "ReLU",
                 lambda_kl = 0, lambda_d = 10, lambda_reg=0, lr=1e-3, epochs=100, batch_size=32, eta_min1 = 1e-16, T_max=1000, eta_min2=1e-16,
                 device=None, clip_grad_norm=1.0, warmup = 0, **kwargs):
        """
        DRD (Distillation of Representation Distillation) model for dimensionality reduction.
        Args:

            - update_model: "joint" | "sep_freeze" | "sep_shared" | "sep_opt"
        """
        print("inside drd init: ", device)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation = getattr(nn, activation) if isinstance(activation, str) else activation
        self.bottleneck_activation = getattr(nn, bottleneck_activation) if isinstance(bottleneck_activation, str) else None
        # self.lambda_kl = lambda_kl
        self.lambda_d = lambda_d
        # self.lambda_reg = lambda_reg
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_grad_norm = clip_grad_norm
        self.warmup_epochs = warmup
        self.eta_min1 = eta_min1
        self.eta_min2 = eta_min2
        self.T_max = T_max

        self.model = AutoEncoder(
            input_dim, 
            latent_dim, 
            hidden_dims, 
            activation = self.activation,
            bottleneck_activation=self.bottleneck_activation).to(self.device)

        self.opt_joint = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_joint,
            T_max = T_max,
            eta_min=eta_min1,    
        )
        self.scheduler2 = None
        # two optimizers if using separate-optimizers
        self.criterion = nn.MSELoss().to(self.device)

    def fit(self, X, teacher_Z=None, verbose=True):

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        if teacher_Z is not None:
            teacher_Z = torch.tensor(teacher_Z, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X, teacher_Z) if teacher_Z is not None else TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        report_interval = max(1, self.epochs // 100)

        for epoch in tqdm(range(self.epochs), disable=not verbose):
            epoch_distill_loss = 0.0
            epoch_recon_loss = 0.0
            num_batches = 0
            for batch in loader:
                x = batch[0].to(self.device, non_blocking=True)
                teacher_z= batch[1].to(self.device, non_blocking=True) 
                self.opt_joint.zero_grad()
                x_rec, z = self.model(x)
                recon_loss = self.criterion(x_rec, x)
                if teacher_z is not None:
                    if epoch < self.warmup_epochs:
                        lambda_d = self.lambda_d * (epoch / self.warmup_epochs)
                    else:
                        lambda_d = self.lambda_d
                    distill_loss = self.criterion(z, teacher_z)
                    
                    loss = recon_loss + lambda_d * distill_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.opt_joint.step()

                epoch_distill_loss += distill_loss.item()
                epoch_recon_loss += recon_loss.item()
                num_batches += 1

            if epoch<self.T_max: self.scheduler1.step()
            else: 
                if self.scheduler2 is None:
                    # at this moment, optimizer.param_groups[0]['lr'] is the end-of-sched1 LR
                    self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.opt_joint, 
                        T_max=self.epochs-epoch, 
                        eta_min=self.eta_min2)
                self.scheduler2.step()

            if (epoch + 1) % report_interval == 0 or epoch == self.epochs - 1:
                tune.report({'distill_loss': epoch_distill_loss / num_batches, 'recon_loss': epoch_recon_loss / num_batches})
        
        return self