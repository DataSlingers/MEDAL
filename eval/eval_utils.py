import torch
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.decomposition import PCA
from src.drd import DRD

def load_and_split(dataset_name, test_size=0.5, seed=0):
    """
    Returns:
      X_train, X_test  — numpy arrays
    """
    if dataset_name == "wine":
        data = load_wine().data
    # elif dataset_name == "single_cell":
    #     data = load_my_single_cell()
    # … add more datasets as needed …
    X = StandardScaler().fit_transform(data)
    return train_test_split(X, test_size=test_size, random_state=seed)

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

def get_teacher_embeddings(method, X_train, X_test, **teacher_kwargs):
    """
    method: str, e.g. "umap", "pca", "mlp"
    teacher_kwargs: hyperparams for that method
    Returns:
      Z_train, Z_test
    """
    if method == "umap":
        model = umap.UMAP(n_components=teacher_kwargs.get("n_components", 2),
                          random_state=teacher_kwargs.get("random_state", 0))
        Z_train = model.fit_transform(X_train)
        Z_test  = model.transform(X_test)
    elif method == "pca":
        pca = PCA(n_components=teacher_kwargs.get("n_components", 2), random_state=0)
        Z_train = pca.fit_transform(X_train)
        Z_test  = pca.transform(X_test)
    # elif method == "mlp":
    #     # build a small MLP as teacher
    #     teacher = make_mlp(input_dim=X_train.shape[1], **teacher_kwargs).eval()
    #     with torch.no_grad():
    #         Z_train = teacher(torch.tensor(X_train, dtype=torch.float32)).numpy()
    #         Z_test  = teacher(torch.tensor(X_test,  dtype=torch.float32)).numpy()
    else:
        raise ValueError(f"Unknown teacher method {method}")
    return Z_train, Z_test

def make_student(input_dim, hidden_dims, latent_dim = 2,
                 symmetric=True, constrained=False, **drd_kwargs):
    """
    Builds and returns a DRD student with the requested architecture.
    """
    config = dict(
      input_dim=input_dim,
      hidden_dims=hidden_dims,
      latent_dim=latent_dim,
    #   symmetric=symmetric,
    #   constrained=bottleneck_dim if constrained else None,
      **drd_kwargs
    )
    
    return DRD(**config)

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
