import torch
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap, SpectralEmbedding
from openTSNE import TSNE
import umap
from sklearn.decomposition import PCA
import pandas as pd, numpy as np
from pathlib import Path
import pickle
#import scanpy as sc

def load_and_split(dataset_name, test_size=0.5, seed=0, labels=False, needs_scaling_input = None):
    X, X_test, labs_train, labs_test = None, None, None, None
    needs_scaling = False
    if dataset_name == "wine":
        data = load_wine().data
        if labels: labs = load_wine().target
        needs_scaling = True
    elif dataset_name == "gene_cancer":
        X = pd.read_csv("/share/ctn/users/bnc2119/drd_data/PANCAN-801x20531/data.csv")
        X.drop(columns=["Unnamed: 0"], inplace=True)
        if labels:
            labs = pd.read_csv("/share/ctn/users/bnc2119/drd_data/PANCAN-801x20531/labels.csv", index_col=0)
        needs_scaling = True
    elif dataset_name == "mnist":
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1)
        X = mnist.data.values[:10000, :]
        if labels: labs = mnist.target.values[:10000]
        X = X.astype('float64') / 255.0
        needs_scaling = False
    elif dataset_name == "diabetes":
        data = load_diabetes().data
        labs = load_diabetes().target
    elif dataset_name == "darmanis":
        data = pd.read_csv('/share/ctn/users/bnc2119/drd_data/GBM_HVG500_with_metadata.csv', index_col=0)
        X = data.iloc[:, 29:].to_numpy()
        labs = data['Location']
    elif dataset_name == "hydra":
        data = pd.read_csv('/shared/share_mala/irchang/drd/Hydra500_official.csv')
        labs = pd.read_csv('/user/bnc2119/drd/Hydra_labels.csv')['cluster.manuscript'].values
        X = data.drop('labels', axis=1).to_numpy()
    #elif dataset_name == "pbmc":
        # OG dim (5858, 33694)
    #    adata  = sc.read_h5ad('/user/bnc2119/drd/inDrops_afterscale.h5ad') 
    #    print("projecting onto first 200 PCs")
    #    X = PCA(n_components=200, random_state=seed).fit_transform(adata.layers['scaledata'])
    #    labs = adata.obs['CellType']
    elif dataset_name == "astro":
        # OG dim (3286, 19)
        X = pd.read_csv('/user/bnc2119/drd/data_mean_imputed_with_ids_all.csv', index_col=0).to_numpy()
        labs = pd.read_csv('/user/bnc2119/drd/cluster_labels_final.csv', index_col=0).to_numpy().flatten()
        needs_scaling = True
    elif dataset_name == "cortical":
        X = np.load('/user/bnc2119/drd/preprocessed-data.npy')
        labs = np.load('/user/bnc2119/drd/tasic_cluster_labels.npy', allow_pickle = True)
    elif dataset_name == "macaque":
        data = pd.read_csv('/shared/share_mala/irchang/drd/tmp/macaque1_pc100.csv')
        labs = data['labels']
        X = data.drop('labels', axis=1).to_numpy()
    elif dataset_name == "synthetic":
        X = np.load('/user/bnc2119/drd/x_arr_train.npy')

    needs_scaling = needs_scaling_input if needs_scaling_input is not None else needs_scaling
    if labels:
        if test_size > 0: 
            X_train, X_test, labs_train, labs_test = train_test_split(X, labs, test_size=test_size, random_state=seed)
        else:
            X_train, labs_train = X, labs

        if needs_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test) if X_test is not None else None
        return X_train, X_test, labs_train, labs_test
    
    if test_size > 0:
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=seed)
    else: X_train = X

    if needs_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test) if X_test is not None else None
    return X_train, X_test

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

def get_teacher_embeddings(method, X_train, **teacher_kwargs):
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
    elif method == "pca":
        model = PCA(**teacher_kwargs_cp)
        Z_train = model.fit_transform(X_train)
    elif method == "tsne":
        model = TSNE(**teacher_kwargs_cp, negative_gradient_method="fft")
        Z_train = model.fit_transform(X_train)
    elif method == "isomap":
        model = Isomap(**teacher_kwargs_cp)
        Z_train = model.fit_transform(X_train)
    elif method == "spectral":
        model = SpectralEmbedding(**teacher_kwargs_cp)
        Z_train = model.fit_transform(X_train)
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
    return Z_train


def eval_student(student, X, Z):
    rmse, dmse = compute_losses(model=student.model,
                                X=X, teacher_z=Z)
    return {"recon_mse": rmse, "distill_mse": dmse}

def process_single_cell_data(data_fp, labels_fp = None):
    sep_regex = r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)'
    print("Loading single-cell data from:", data_fp)
    df = pd.read_csv(
        data_fp,
        sep=sep_regex,
        engine="python",
        quotechar='"',
        header=0,
        skipinitialspace=True
    )

    # strip any remaining quotes from the column names
    df.columns = [c.strip('"') for c in df.columns]
    df = df.T
    if labels_fp is not None:
        meta = pd.read_csv(
            labels_fp,
            sep=sep_regex,
            engine="python",
            quotechar='"',
            skipinitialspace=True  # drop any spurious space after splitting
        )

        # 3) Clean up column names (strip any remaining quotes)
        meta.columns = [col.strip('"') for col in meta.columns]

        # 4) (Optional) Convert obvious numeric columns
        numeric = [
            "Total_reads","Unique_reads","Unique_reads_percent",
            "Splice_sites_total","Splice_sites_Annotated","Splice_sites_GT.AG",
            "Splice_sites_GC.AG","Splice_sites_AT.AC","Splice_sites_non_canonical",
            "Multimapping_reads_percent","Unmapped_mismatch","Unmapped_short",
            "Unmapped_other","ERCC_reads","Non_ERCC_reads","ERCC_to_non_ERCC",
            "Genes_detected","Cluster_2d"
        ]
        for c in numeric:
            if c in meta:
                meta[c] = pd.to_numeric(meta[c], errors="coerce")
        meta.index = meta.index.str.replace(r'^"|"$', '', regex=True)
        return df, meta

    return df
