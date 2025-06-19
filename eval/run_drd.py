# experiment.py
import numpy as np, pandas as pd
if not hasattr(np, "product"):
    np.product = np.prod
from pathlib import Path
from src.drd import DRD
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from umap.parametric_umap import ParametricUMAP
from sklearn.metrics import mean_squared_error
from keras.losses import MeanSquaredError
import os
import argparse, json
import torch, torch.nn as nn
import umap

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# compute all four losses
def compute_losses(model, X, teacher_z=None, device=None):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    # embeddings
    with torch.no_grad():
        x_recon, student_z = model(X_tensor)
    x_recon = x_recon.cpu().numpy()
    student_z = student_z.cpu().numpy()

    recon_mse = mean_squared_error(X, x_recon)
    if teacher_z is not None:
        distill_mse = mean_squared_error(teacher_z, student_z)
        return recon_mse, distill_mse
    return recon_mse, None

def run_simulation(var_name, var_values, param_dict):
    wine_data = load_wine()
    X_wine = wine_data.data
    X_wine = StandardScaler().fit_transform(X_wine)  # scale features

    all_results = []

    for seed in range(10):
        X_train, X_test = train_test_split(
            X_wine, test_size=0.5, random_state=seed
        )
        param_dict["input_dim"] = X_train.shape[1]

        # Suppose teacher_z comes from a small MLP teacher
        # D_T = 3
        # teacher_model = nn.Sequential(
        #     nn.Linear(param_dict["input_dim"], 256), nn.ReLU(),
        #     *[ layer for _ in range(D_T-1) for layer in (nn.Linear(256,256), nn.ReLU()) ],
        #     nn.Linear(256, 2)
        #     ).eval()
        # with torch.no_grad():
        #     teacher_z_train = teacher_model(torch.tensor(X_train, dtype=torch.float32)).numpy()
        #     teacher_z_test = teacher_model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        u = umap.UMAP(n_components=2, random_state = 42)
        teacher_z_train = u.fit_transform(X_train)
        teacher_z_test = u.transform(X_test)

        exp_factor = [4, 5, 6, 7, 8, 9, 10, 11, 12]
        for val in var_values:
            param_dict[var_name] = val
            for id, e in enumerate(exp_factor):
                param_dict["hidden_dims"] = tuple(np.exp2(exp_factor[:id+1]).astype(int).tolist())
                # DRD
                student = DRD(**param_dict)
                student.fit(X_train, teacher_Z=teacher_z_train, verbose=False)

                recon_mse_tr, distill_mse_tr = compute_losses(model = student.model, X = X_train, teacher_z = teacher_z_train)
                recon_mse_te, distill_mse_te = compute_losses(model = student.model, X = X_test, teacher_z = teacher_z_test)
                
                # save results
                all_results.append({
                    "seed": seed,
                    f"{var_name}": val,
                    "exp": e,
                    "distill_drd_train":    distill_mse_tr,
                    "recon_drd_train":      recon_mse_tr,
                    "distill_drd_test":     distill_mse_te,
                    "recon_drd_test":       recon_mse_te,
                })
    return all_results

# save
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DRD simulation on wine dataset")
    parser.add_argument("--var_name", type=str, default="lambda_d", help="Variable name to vary")
    parser.add_argument("--var_values", type=json.loads, help="Values for the variable")
    parser.add_argument("--latent_dim", type=int, default=2, help="Dimensionality of the latent space")
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[64,128,256,256], help="Hidden dimensions for the DRD model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lambda_d", type=float, default=1.0, help="Weight for the distillation loss")
    parser.add_argument("--lambda_reg", type=float, default=0.0, help="Weight for the regularization loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model on (e.g., 'cuda', 'cpu')")
    args = parser.parse_args()
    print(args.var_values)

    # construct param_dict
    param_dict = {}
    for arg in vars(args):
        if arg in ["var_name", "var_values"]:
            continue
        param_dict[arg] = getattr(args, arg)

    all_results = run_simulation(args.var_name, args.var_values, param_dict)
    Path("results").mkdir(exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(f"results/drd_losses_{args.var_name}_size&depth.csv", index=False)
