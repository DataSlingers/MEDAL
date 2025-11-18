import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from ray import tune
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Dataset
from src.drd.loss import get_loss_function
import torch.nn.functional as F
    
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10, hidden_dims=(128, 128),
                 activation=nn.ReLU, bottleneck_activation=None):
        super().__init__()
        self.ActivationCls = activation

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
        print("encoder layers:", self.encoder)
        

        # --- DECODER BLOCK (unchanged) ---
        decoder_layers = []
        prev_dim = latent_dim
        decoder_layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        if activation is not None:
            decoder_layers.append(activation())
        prev_dim = hidden_dims[-1]

        for h in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, h))
            if activation is not None:
                decoder_layers.append(activation())
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        print("decoder layers:", decoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_identity()

    def _init_identity(self, eps=1e-3):
        # initialize dense Linear layers close to identity; SparseLinearFirst has its own init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                try:
                    nn.init.eye_(m.weight)
                except Exception:
                    nn.init.normal_(m.weight, mean=0.0, std=eps)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        If use_sparse_first=True:
            - x must be a torch sparse tensor (coalesced COO or CSR) on the right device.
        Else:
            - x is a dense float tensor as before.
        """
        z = self.encoder(x) 
        x_recon = self.decoder(z)
        return x_recon, z

class DRD(BaseEstimator, TransformerMixin):
    def __init__(self, input_dim, latent_dim=2, hidden_dims=(128, 64), activation="ReLU",   
                 bottleneck_activation = "ReLU",
                 lambda_d = 10, lr=1e-3, epochs=100, batch_size=32, eta_min1 = 1e-16, T_max=1000, eta_min2=1e-16, lr_restart = None,
                 device=None, clip_grad_norm=1.0, warmup = 0, adamw_weight_decay = 1e-5, new_scheduler = False,
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
        self.lambda_d = lambda_d
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_grad_norm = clip_grad_norm
        self.warmup = warmup
        self.eta_min1 = eta_min1
        self.eta_min2 = eta_min2
        self.T_max = T_max

        self.model = AutoEncoder(
            input_dim, 
            latent_dim, 
            hidden_dims, 
            activation = self.activation,
            bottleneck_activation=self.bottleneck_activation).to(self.device)

        self.opt_joint = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=adamw_weight_decay)
        self.scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_joint,
            T_max = T_max,
            eta_min=eta_min1,    
        )
        if new_scheduler == True:
            self.scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_joint, "min", factor = 0.9, threshold=1e-4, patience=20, min_lr = 1e-7)
        self.scheduler2 = None
        self.criterion = nn.MSELoss().to(self.device)
        self.lr_restart = lr_restart

    def fit(self, X, teacher_Z=None, verbose=True,
            phase=None, pretrained_path=None,
            target_bands=None, stability_window=10,
            epsilon_distill=0.1, epsilon_recon=0.005,
            patience=3, return_on_stable=False,
            # checkpointing
            save_dir = None, prefix=None,
            print_tag=False
            ):
        '''
        phase:              "pretrain" only reconstruction. Save weights to pretrained_path, 
                            "finetune" load weights from pretrained_path before starting, then train with distill loss.
        target_bands:       list of (min, max) tuples for distill loss bands to target
        stability_window:   number of epochs to consider for stability
        patience:           number of consecutive stable checks to confirm stability
        return_on_stable:   if True, stop training once a stable band checkpoint is captured
        save_dir:           directory to save checkpoints
        prefix:             prefix for checkpoint filenames
        print_tag:          if True, print training progress to console
        '''

        def _report_or_print(metrics):
            if print_tag or phase == "pretrain":
                print(metrics)
            else: 
                tune.report(metrics)

        if phase == "finetune" and pretrained_path:
            ref_state = torch.load(pretrained_path, map_location="cpu")["model"]
            self.model.load_state_dict(ref_state, strict=False)

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        if teacher_Z is not None:
            teacher_Z = torch.tensor(teacher_Z, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X, teacher_Z) if teacher_Z is not None else TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        report_interval = max(1, self.epochs // 1000)

        distill_history, recon_history = [], []
        stable_counter = 0
        early_stopped = False
        target_bands_saved = np.zeros(len(target_bands)) if target_bands else None ### TEMPORARY
        self.stable_band_ = None  
        self.stable_epoch_ = None
        best_recon_in_band = {}            

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
                
                if phase == "pretrain" or teacher_z is None: # during pretraining, no distill loss
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

            # checking if we need to switch scheduler
            if epoch <= self.T_max: self.scheduler1.step(avg_distill) # for pretraining, just set T_max = epochs
            else:
                if self.scheduler2 is None:
                    # at this moment, optimizer.param_groups[0]['lr'] is the end-of-sched1 LR
                    if self.lr_restart is not None:
                        self.opt_joint.param_groups[0]['lr'] = self.lr_restart
                    self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.opt_joint, 
                        T_max=self.epochs-epoch, 
                        eta_min=self.eta_min2)
                self.scheduler2.step()
            
            # Task: ensure stability and convergence
            if target_bands:
                distill_history.append(avg_distill)
                recon_history.append(avg_recon)
                # find which band (if any) we are in
                current_band = None
                for idx, (tau_min, tau_max) in enumerate(target_bands):
                    if tau_min <= avg_distill <= tau_max:
                        current_band = idx
                        ###### TEMPORARY CHKPTS
                        if idx < len(target_bands) - 1 and not target_bands_saved[idx]:
                            return_on_stable = False  # only return on stable for the last band
                            base = Path(save_dir) / f"{prefix}_ckpts"
                            base.mkdir(parents=True, exist_ok=True)
                            ckpt_path = base / f"band{idx}.pt"

                            state = {"model": _state_dict_cpu(self.model)}
                            torch.save(state, ckpt_path)
                            print(f"Saved model to {ckpt_path}, distill loss: {avg_distill:.6f}")
                            target_bands_saved[idx] = 1
                        elif idx == len(target_bands) - 1:
                            return_on_stable = True
                        ######
                        break
                # perform stability check only if in a band and enough history exists
                if current_band is not None and len(distill_history) >= stability_window:
                    delta_dist = distill_history[-1] - distill_history[-stability_window]
                    delta_recon = recon_history[-1] - recon_history[-stability_window]
                    slope_dist = abs(delta_dist) / stability_window
                    slope_recon = abs(delta_recon) / stability_window

                    if (slope_dist < epsilon_distill  and
                            slope_recon < epsilon_recon ):
                        stable_counter += 1
                    else:
                        stable_counter = 0

                    # when stable, save/replace ONE checkpoint for this band
                    if stable_counter >= patience:
                        # record band and epoch, optionally break early
                        self.stable_band_ = current_band
                        self.stable_epoch_ = epoch

                        prev = best_recon_in_band.get(current_band)
                        if (prev is None) or (avg_recon < prev[0] - 1e-12):
                            best_recon_in_band[current_band] = (avg_recon, None)

                            if return_on_stable:
                                _report_or_print({'distill_loss': avg_distill, 'recon_loss': avg_recon, 'lr': self.opt_joint.param_groups[0]['lr']})
                                break
                else:
                    stable_counter = 0  # reset if not in a band or not enough history

            # reporting losses
            if (epoch + 1) % report_interval == 0 or epoch == self.epochs - 1:
                _report_or_print({'distill_loss': avg_distill, 'recon_loss': avg_recon, 'lr': self.opt_joint.param_groups[0]['lr']})
            if early_stopped:
                break
            
        if save_dir is not None:
            base = Path(save_dir) / f"{prefix}_ckpts"
            base.mkdir(parents=True, exist_ok=True)
            ckpt_path = base / "final.pt"

            state = {"model": _state_dict_cpu(self.model)}
            torch.save(state, ckpt_path)
            print(f"Saved model to {ckpt_path}")
        
        if phase == "pretrain" and pretrained_path is not None:
            pretrain_ckpt_path = Path(pretrained_path) / (
                f"{prefix}_pretrain.pt" if prefix else "pretrain.pt"
            )
            pretrain_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": _state_dict_cpu(self.model)}, pretrain_ckpt_path)
            return pretrain_ckpt_path
        
        return self


def _state_dict_cpu(model: nn.Module):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}
