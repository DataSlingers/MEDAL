import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from ray import tune
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pickle
    
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10, hidden_dims=(128, 128),
                 activation=nn.ReLU, bottleneck_activation=None, dropout_rate=0.1, use_batchnorm=False,
                 final_activation=None):
        super().__init__()
        self.ActivationCls = activation
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm

        # --- ENCODER BLOCK ---
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            if activation is not None:
                encoder_layers.append(activation())
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        if bottleneck_activation is not None:
            encoder_layers.append(bottleneck_activation())

        self.encoder = nn.Sequential(*encoder_layers)
        

        # --- DECODER BLOCK ---
        decoder_layers = []
        prev_dim = latent_dim
        decoder_layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm1d(hidden_dims[-1]))

        if activation is not None:
            decoder_layers.append(activation())
        
        if self.dropout_rate > 0:
            decoder_layers.append(nn.Dropout(self.dropout_rate))

        prev_dim = hidden_dims[-1]

        for h in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm: decoder_layers.append(nn.BatchNorm1d(h))
            if activation is not None:
                decoder_layers.append(activation())
            
            if self.dropout_rate > 0:
                decoder_layers.append(nn.Dropout(self.dropout_rate))
            
            prev_dim = h
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        if final_activation is not None:
            decoder_layers.append(final_activation())
        
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_identity()

    def _init_identity(self, eps=1e-3):
        # initialize dense nonlinear layers close to identity
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.use_batchnorm: 
                    # shallower networks, use this
                    nn.init.normal_(m.weight, mean=0.0, std=eps)
                else: 
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.encoder(x) 
        x_recon = self.decoder(z)
        return x_recon, z

class MEDAL(BaseEstimator, TransformerMixin):
    def __init__(self, input_dim, latent_dim=2, hidden_dims=(128, 64), activation="ReLU",   
                 bottleneck_activation = "ReLU", final_activation = None, criterion=nn.MSELoss,
                 lambda_d = 10, lr=1e-3, epochs=100, batch_size=32, eta_min = 1e-16, device=None, clip_grad_norm=1.0, warmup = 0, adamw_weight_decay = 1e-5, patience = 20, factor = 0.9, use_batchnorm=False, dropout_rate=0.1,
                 **kwargs):
        """
        DRD (Distillation of Representation Distillation) model for dimensionality reduction.
        Args:
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation = getattr(nn, activation) if isinstance(activation, str) else activation
        self.bottleneck_activation = getattr(nn, bottleneck_activation) if isinstance(bottleneck_activation, str) else None
        self.final_activation = getattr(nn, final_activation) if isinstance(final_activation, str) else final_activation
        self.lambda_d = lambda_d
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_grad_norm = clip_grad_norm
        self.warmup = warmup
        self.eta_min = eta_min
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

        self.model = AutoEncoder(
            input_dim, 
            latent_dim, 
            hidden_dims, 
            activation = self.activation,
            bottleneck_activation=self.bottleneck_activation,
            final_activation=self.final_activation,
            dropout_rate=self.dropout_rate,
            use_batchnorm=self.use_batchnorm).to(self.device)

        self.opt_joint = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=adamw_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_joint, "min", factor = factor, threshold=1e-4, patience=patience, min_lr = self.eta_min, eps=1e-15)
        self.criterion = criterion().to(self.device)

    def fit(self, X, teacher_Z=None, verbose=True,
            target_bands=None, stability_window=10,
            epsilon_distill=0.1, epsilon_recon=0.005,
            patience=3, return_on_stable=False,
            # checkpointing
            save_dir = None, prefix=None,
            print_tag=False
            ):
        '''
        target_bands:       list of (min, max) tuples for distill loss bands to target
        stability_window:   number of epochs to consider for stability
        patience:           number of consecutive stable checks to confirm stability
        return_on_stable:   if True, stop training once a stable band checkpoint is captured
        save_dir:           directory to save checkpoints
        prefix:             prefix for checkpoint filenames
        print_tag:          if True, print training progress to console
        '''

        def _report_or_print(metrics):
            if print_tag:
                print(metrics)
            else:
                tune.report(metrics)

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        if teacher_Z is not None:
            teacher_Z = torch.tensor(teacher_Z, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X, teacher_Z) if teacher_Z is not None else TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        report_interval = max(1, self.epochs // 1000)

        # --- history and state init ---
        distill_history, recon_history = [], []
        self.stable_counter = 0
        early_stopped = False
        target_bands_saved = np.zeros(len(target_bands)) if target_bands else None ### TEMPORARY
        self.stable_band_ = None  
        self.stable_epoch_ = None
        best_recon_in_band = {}            
        
         # --- main training loop ---
        for epoch in tqdm(range(self.epochs), disable=not verbose):
            epoch_distill_loss = 0.0
            epoch_recon_loss = 0.0
            num_batches = 0
            for batch in loader:
                x = batch[0].to(self.device, non_blocking=True)
                teacher_z= batch[1].to(self.device, non_blocking=True) if teacher_Z is not None else None
                self.opt_joint.zero_grad() 
                x_rec, z = self.model(x)
                recon_loss = self.criterion(x_rec, x) 
                
                if teacher_z is None:
                    loss = recon_loss
                    distill_loss = torch.tensor(0.0, device=self.device)
                else:
                    if epoch < self.warmup:
                        lambda_d = self.lambda_d * (epoch / self.warmup)
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

            avg_distill = epoch_distill_loss / num_batches
            avg_recon = epoch_recon_loss / num_batches

            distill_history.append(avg_distill)
            recon_history.append(avg_recon)

            self.scheduler.step(avg_distill)
            
            # Task: ensure stability and convergence
            if target_bands:
                early_stopped = self._check_stability_and_checkpoint(
                    epoch, avg_distill, avg_recon, distill_history, recon_history,
                    target_bands, stability_window, epsilon_distill, epsilon_recon, patience,
                    return_on_stable, save_dir, prefix, target_bands_saved, best_recon_in_band
                )
            
            # reporting losses
            if (epoch + 1) % report_interval == 0 or epoch == self.epochs - 1:
                _report_or_print({'distill_loss': avg_distill, 'recon_loss': avg_recon, 'lr': self.opt_joint.param_groups[0]['lr'], "stability": self.stable_counter})
            
            if early_stopped:
                break
            
        if save_dir is not None:
            base = Path(save_dir) / f"{prefix}_ckpts"
            base.mkdir(parents=True, exist_ok=True)
            ckpt_path = base / "final.pt"

            state = {"model": self._state_dict_cpu()}
            torch.save(state, ckpt_path)
            print(f"Saved model to {ckpt_path}")
        
        return self


    def _state_dict_cpu(self):
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
    
    def _check_stability_and_checkpoint(self, epoch, avg_distill, avg_recon, 
                                        distill_history, recon_history,
                                        target_bands, stability_window, 
                                        epsilon_distill, epsilon_recon, patience,
                                        return_on_stable, save_dir, prefix, 
                                        target_bands_saved, best_recon_in_band):
        """
        Checks for training stability within defined bands and saves checkpoints.
        Returns (stable_counter, early_stop_flag).
        """
        current_band = None
        
        # 1. Check which band the current distillation loss falls into
        for idx, (tau_min, tau_max) in enumerate(target_bands):
            if tau_min <= avg_distill <= tau_max:
                current_band = idx
                ###### TEMPORARY CHKPTS
                if not target_bands_saved[idx] and idx < len(target_bands) - 1:
                    base = Path(save_dir) / f"{prefix}_ckpts"
                    base.mkdir(parents=True, exist_ok=True)
                    ckpt_path = base / f"band{idx}.pt"
                    state = {"model": self._state_dict_cpu()}
                    torch.save(state, ckpt_path)
                    print(f"Saved model to {ckpt_path}, distill loss: {avg_distill:.6f}, epoch {epoch}")

                    target_bands_saved[idx] = 1
                ######
                break

        # 2. Perform stability check only if in a band and enough history exists
        early_stop_flag = False
        
        if current_band is not None and len(distill_history) >= stability_window:
            delta_dist = distill_history[-1] - distill_history[-stability_window]
            delta_recon = recon_history[-1] - recon_history[-stability_window]
            slope_dist = abs(delta_dist) / stability_window
            slope_recon = abs(delta_recon) / stability_window

            is_stable = (slope_dist < epsilon_distill) and (slope_recon < epsilon_recon)

            if is_stable:
                self.stable_counter += 1
            else:
                self.stable_counter = 0

            # 3. Handle confirmed stability (patience reached)
            if self.stable_counter >= patience:
                self.stable_band_ = current_band
                self.stable_epoch_ = epoch
                
                # Check if this is the best reconstruction loss found for this band
                prev = best_recon_in_band.get(current_band)
                if (prev is None) or (avg_recon < prev[0] - 1e-12):
                    best_recon_in_band[current_band] = (avg_recon, None)
                    
                    if return_on_stable and current_band == len(target_bands) - 1:
                        early_stop_flag = True # Early stop for the final band
        else:
            self.stable_counter = 0 # Reset if not in band or not enough history

        return early_stop_flag

class GlobalEmbeddingNormalizer:
    def __init__(self, mean_=None, scale_=None, eps=1e-8):
        self.mean_ = mean_
        self.scale_ = scale_
        self.eps = eps

    def fit(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        self.mean_ = Z.mean(axis=0, keepdims=True)
        Zc = Z - self.mean_
        self.scale_ = np.sqrt(np.mean(np.sum(Zc**2, axis=1)))
        self.scale_ = max(float(self.scale_), self.eps)
        return self

    def transform(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        return (Z - self.mean_) / self.scale_

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        return Z * self.scale_ + self.mean_
    
    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(
            mean_=np.asarray(d["mean"], dtype=np.float32),
            scale_=float(d["scale"]),
        )