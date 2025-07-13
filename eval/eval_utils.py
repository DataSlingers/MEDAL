import torch
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, Isomap
import umap
from sklearn.decomposition import PCA
from src.drd import DRD
import pandas as pd
from pathlib import Path
import pickle

def load_and_split(dataset_name, test_size=0.5, seed=0, labels=False):
    """
    Returns:
      X_train, X_test  — numpy arrays
    """
    if dataset_name == "wine":
        data = load_wine().data
        if labels: labs = load_wine().target
    elif dataset_name == "single_cell":
        data = pd.read_csv("Single-cell/data.csv")
        data.drop(columns=["Unnamed: 0"], inplace=True)
        if labels:
            labs = pd.read_csv("Single-cell/labels.csv", index_col=0)
    elif dataset_name == "mnist":
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1)
        data = mnist.data.values[:10000, :]
        if labels: labs = mnist.target.values[:10000]
    elif dataset_name == "diabetes":
        data = load_diabetes().data
        if labels: labs = load_diabetes().target
    X = StandardScaler().fit_transform(data)
    if labels:
        return train_test_split(X, labs, test_size=test_size, random_state=seed)
    return train_test_split(X, test_size=test_size, random_state=seed)

def compute_losses(model, X, teacher_z=None, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
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


def get_teacher_embeddings(method, X_train, X_test, **teacher_kwargs):
    """
    method: str, e.g. "umap", "pca", "mlp"
    teacher_kwargs: hyperparams for that method
    Returns:
      Z_train, Z_test
    """
    teacher_kwargs_cp = teacher_kwargs.copy()
    teacher_kwargs_cp.pop("save_teacher_model", None)
    teacher_kwargs_cp.pop("save_teacher_path", None)
    if method == "umap":
        model = umap.UMAP(**teacher_kwargs_cp)
        Z_train = model.fit_transform(X_train)
        Z_test  = model.transform(X_test)
    elif method == "pca":
        model = PCA(**teacher_kwargs_cp)
        Z_train = model.fit_transform(X_train)
        Z_test  = model.transform(X_test)
    elif method == "tsne":
        model = TSNE(**teacher_kwargs_cp)
        Z_train = model.fit_transform(X_train)
        Z_test  = model.transform(X_test)
    elif method == "isomap":
        model = Isomap(**teacher_kwargs_cp)
        Z_train = model.fit_transform(X_train)
        Z_test  = model.transform(X_test)
    # elif method == "mlp":
    #     # build a small MLP as teacher
    #     teacher = make_mlp(input_dim=X_train.shape[1], **teacher_kwargs).eval()
    #     with torch.no_grad():
    #         Z_train = teacher(torch.tensor(X_train, dtype=torch.float32)).numpy()
    #         Z_test  = teacher(torch.tensor(X_test,  dtype=torch.float32)).numpy()
    else:
        raise ValueError(f"Unknown teacher method {method}")
    
    if teacher_kwargs.get("save_teacher_model", False):
        try:
            model_path = Path(teacher_kwargs.get("save_teacher_path"))
        except:
            raise RuntimeError("Please provide a valid path for saving the model")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Teacher model saved to {model_path}")
    return Z_train, Z_test

def make_student(method, input_dim=None, hidden_dims=None, latent_dim = 2,
                 symmetric=True, constrained=False, **student_kwargs):
    """
    Builds and returns a DRD student with the requested architecture.
    """
    if method == "drd":
        if input_dim is None or hidden_dims is None:
            raise ValueError("For DRD, input_dim and hidden_dims must be specified.")
        config = dict(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        constrained = constrained,
        **student_kwargs
        )
        return DRD(**config)
    elif method == "pca":
        return PCA(n_components=student_kwargs.get("n_components", 2), random_state = student_kwargs.get("random_state", 0))
    else:
        raise ValueError(f"Unknown student method {method}")

def fit_student(student, X_train, Z_train, optimize="joint", **fit_kwargs):
    """
    optimize: "joint", "encoder", or "decoder"
    """
    student.fit(X_train, teacher_Z=Z_train,
                # optimize=optimize, 
                **fit_kwargs)
    return student

def eval_student(student, X, Z):
    rmse, dmse = compute_losses(model=student.model,
                                X=X, teacher_z=Z)
    return {"recon_mse": rmse, "distill_mse": dmse}

def eval_pca_baseline(pca_model, X_tr, X_te):
    pca_model.fit(X_tr)
    return {"recon_test_mse": mean_squared_error(X_te, pca_model.inverse_transform(pca_model.transform(X_te))),
            "recon_train_mse": mean_squared_error(X_tr, pca_model.inverse_transform(pca_model.transform(X_tr)))}