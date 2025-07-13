import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset
from src.drd.loss import get_loss_function
    
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10, hidden_dims=(128, 128), activation=nn.ReLU, constrained=False):
        super().__init__()

        encoder_layers = []
        prev_dim = input_dim
        for e_id, h in enumerate(hidden_dims):                
            encoder_layers.append(nn.Linear(prev_dim, h))
            if activation is not None:
                encoder_layers.append(activation())
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        # encoder_layers.append(activation())
        print("encoder layers:", encoder_layers)
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            if activation is not None:
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
                 lambda_kl = 0, lambda_d = 10, lambda_reg=0, lr=1e-3, epochs=100, batch_size=32, eta_min1 = 1e-16, T_max=1000, eta_min2=1e-16,
                 device=None, clip_grad_norm=1.0, update_mode="joint", constrained=False, warmup = 0, **kwargs):
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
        self.lambda_kl = lambda_kl
        self.lambda_d = lambda_d
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_grad_norm = clip_grad_norm
        self.update_mode    = update_mode
        self.constrained = constrained
        self.warmup_epochs = warmup
        self.eta_min1 = eta_min1
        self.eta_min2 = eta_min2
        self.T_max = T_max

        self.model = AutoEncoder(self.input_dim, self.latent_dim, self.hidden_dims, activation = self.activation, constrained=self.constrained).to(self.device)

        self.opt_joint = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_joint,
            T_max = T_max,
            eta_min=eta_min1,    
        )
        self.scheduler2 = None
        # two optimizers if using separate-optimizers
        if self.update_mode == "sep_opt":
            self.opt_enc = optim.Adam(self.model.encoder.parameters(), lr=self.lr)
            self.opt_dec = optim.Adam(self.model.decoder.parameters(), lr=self.lr)
        else:
            self.opt_enc = self.opt_dec = None

        self.criterion = nn.MSELoss().to(self.device)

    def fit(self, X, teacher_Z=None, verbose=False):

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        if teacher_Z is not None:
            teacher_Z = torch.tensor(teacher_Z, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X, teacher_Z) if teacher_Z is not None else TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        for epoch in tqdm(range(self.epochs), disable=not verbose):
            total_loss = 0
            for batch in loader:
                x = batch[0].to(self.device, non_blocking=True)
                teacher_z= batch[1].to(self.device, non_blocking=True) if teacher_Z is not None else None
                # --- 1) Joint update: one forward, one backward, one step ---
                if self.update_mode == "joint":
                    self.opt_joint.zero_grad()
                    x_rec, z = self.model(x)
                    loss = self.criterion(x_rec, x)
                    if teacher_z is not None:
                        if epoch < self.warmup_epochs:
                            lambda_d = self.lambda_d * (epoch / self.warmup_epochs)
                        else:
                            lambda_d = self.lambda_d
                        loss = loss + lambda_d * self.criterion(z, teacher_z)
                        
                        print(f"distill loss {epoch}/{self.epochs}: {self.criterion(z, teacher_z)}")
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                    self.opt_joint.step()

                    batch_loss = loss

                # --- 2) Separate-freeze: freeze one subnetwork at a time ---
                elif self.update_mode == "sep_freeze":
                    # Decoder step (freeze encoder)
                    for p in self.model.encoder.parameters():
                        p.requires_grad = False
                    for p in self.model.decoder.parameters():
                        p.requires_grad = True

                    self.opt_joint.zero_grad()
                    with torch.no_grad():
                        z = self.model.encoder(x)
                    x_rec = self.model.decoder(z)
                    loss_dec = self.criterion(x_rec, x)
                    loss_dec.backward()
                    self.opt_joint.step()

                    # Encoder step (freeze decoder)
                    for p in self.model.encoder.parameters():
                        p.requires_grad = True
                    for p in self.model.decoder.parameters():
                        p.requires_grad = False

                    self.opt_joint.zero_grad()
                    x_rec, z = self.model(x)
                    loss_enc = self.criterion(x_rec, x)
                    if teacher_z is not None:
                        loss_enc = loss_enc + self.lambda_d * self.criterion(z, teacher_z)
                    loss_enc.backward()
                    self.opt_joint.step()

                    # unfreeze all for next iteration
                    for p in self.model.decoder.parameters():
                        p.requires_grad = True

                    batch_loss = loss_dec + loss_enc
                
                # 3) Separate-optimizers: two optimizers, two passes 
                elif self.update_mode == "sep_opt":
                    # Decoder update
                    with torch.no_grad():
                        z = self.model.encoder(x)
                    x_rec = self.model.decoder(z)
                    loss_dec = self.criterion(x_rec, x)
                    self.opt_dec.zero_grad()
                    loss_dec.backward()
                    self.opt_dec.step()

                    # Encoder update (with gradients through decoder)
                    x_rec, z = self.model(x)
                    loss_enc = self.criterion(x_rec, x)
                    if teacher_z is not None:
                        loss_enc = loss_enc + self.lambda_d * self.criterion(z, teacher_z)
                    self.opt_enc.zero_grad()
                    loss_enc.backward()
                    self.opt_enc.step()

                    batch_loss = loss_dec + loss_enc

                else:
                    raise ValueError(f"Unknown update_mode={self.update_mode}")

                total_loss += batch_loss.item()
            
            if epoch<self.T_max: self.scheduler1.step()
            else: 
                if self.scheduler2 is None:
                    print("Switching...., init=",self.opt_joint.param_groups[0]['lr'])
                    # at this moment, optimizer.param_groups[0]['lr'] is the end-of-sched1 LR
                    self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_joint, T_max=self.epochs-epoch, eta_min=self.eta_min2)
                self.scheduler2.step()

            if verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, [{self.update_mode}] Loss: {total_loss / len(loader)}")
        
        return self