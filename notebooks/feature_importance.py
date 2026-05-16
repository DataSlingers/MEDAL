import torch
from captum.attr import IntegratedGradients
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np, pandas as pd
from src.medal.eval_utils import load_and_split, get_teacher_embeddings
from src.medal.core import AutoEncoder
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
# Set MEDAL_DATA_DIR to your data directory (containing embeddings/, tmp_results/, etc.)
PATH_PREFIX = os.environ.get('MEDAL_DATA_DIR', os.path.expanduser('~/drd_data'))

class MedalDistillScore(torch.nn.Module):
    """
    Returns a scalar score per sample: negative distill MSE to teacher embedding.
    """
    def __init__(self, medal_model, teacher_embed_tensor):
        super().__init__()
        self.model = medal_model
        self.teacher = teacher_embed_tensor  # shape (N, zdim) aligned with dataset order

    def forward(self, x, idx):
        z = self.model.encoder(x)            # (B, zdim)
        t = self.teacher[idx]                # (B, zdim)
        # score: higher is better alignment
        return -torch.sum((z - t) ** 2, dim=1)  # (B,)

def medal_feature_importance_ig(
    medal_model,
    X,                # torch.Tensor (N, D)
    teacher_Z,         # torch.Tensor (N, zdim)
    baseline="zeros",  # or "mean"
    batch_size=256,
    steps=64,
    device="cuda:0",
):
    medal_model.eval().to(device)
    X = X.to(device)
    teacher_Z = teacher_Z.to(device)

    if baseline == "zeros":
        base_x = torch.zeros(1, X.shape[1], device=device)
    elif baseline == "mean":
        base_x = X.mean(dim=0, keepdim=True)
    else:
        raise ValueError("baseline must be 'zeros' or 'mean'")

    score_model = MedalDistillScore(medal_model, teacher_Z).to(device)
    ig = IntegratedGradients(lambda x, idx: score_model(x, idx))  # idx passed as additional arg

    all_attr = []
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        xb = X[start:end]
        idxb = torch.arange(start, end, device=device)

        # Expand baseline to batch
        bb = base_x.expand_as(xb)

        attr = ig.attribute(
            inputs=xb,
            baselines=bb,
            additional_forward_args=(idxb,),
            n_steps=steps,
        )  # (B, D)
        all_attr.append(attr.detach())

    attr = torch.cat(all_attr, dim=0)  # (N, D)

    # Global feature importance: mean absolute attribution
    global_imp = attr.abs().mean(dim=0)  # (D,)
    return attr, global_imp

# loading AE
def load_trained_ae(ckpt_path, input_dim, hidden_dims, latent_dim=2, batchnorm=False):
    import sys, os
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    model = AutoEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=torch.nn.SELU,            
        bottleneck_activation=None,
        use_batchnorm = batchnorm
    )
    sys.stdout = old_stdout

    sd = torch.load(ckpt_path, map_location="cuda:0")

    # unwrap to the real state dict
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # remove DDP prefixes if any
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # load STRICTLY so we fail fast if something doesn’t match
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded params: {len(sd)} | missing={len(missing)} | unexpected={len(unexpected)}")
    if missing or unexpected:
        print("⚠️ Check architecture/keys. Example missing:", missing[:3], "unexpected:", unexpected[:3])

    model.eval()
    return model

X, X_test, y, y_test = load_and_split("hydra", test_size=0.2, seed=0, labels=True)
teacher_embed = np.load(Path(PATH_PREFIX) / 'embeddings/hydra_umap_36_0.1_0_train.npy')
student = load_trained_ae(Path(PATH_PREFIX) / f'tmp_results/chkpt/hydra/umap2_36_0.1_0_ckpts/final.pt',
                input_dim=500,
                hidden_dims=[309, 1792, 1792, 1792],
                latent_dim=2, batchnorm=True)
attr, global_imp = medal_feature_importance_ig(student, torch.from_numpy(X).float(), torch.from_numpy(teacher_embed))
attr, global_imp = attr.detach().float().cpu(), global_imp.detach().float().cpu()
np.save('attr_hydra_umap_36', attr)
np.save('global_imp_hydra_umap_36', global_imp)