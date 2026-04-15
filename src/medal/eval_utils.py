# backward-compatibility shim — new code should import from medal.teacher and medal.io
from medal.teacher import get_teacher_embeddings  # noqa: F401
from medal.io import compute_losses, eval_student  # noqa: F401

import torch
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd, numpy as np
from pathlib import Path
import pickle

def load_and_split(dataset_name, test_size=0.5, seed=0, labels=False, needs_scaling_input = None, id_set = None):
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
        data = pd.read_csv('/share/ctn/users/bnc2119/drd_data/Hydra500_official.csv')
        labs = pd.read_csv('/share/ctn/users/bnc2119/drd_data/Hydra_labels.csv')['cluster.manuscript'].values
        X = data.drop('labels', axis=1).to_numpy()
    elif dataset_name == "astro":
        # OG dim (3286, 19)
        X = pd.read_csv('/share/ctn/users/bnc2119/drd_data/data_mean_imputed_with_ids_all.csv', index_col=0).to_numpy()
        labs = pd.read_csv('/share/ctn/users/bnc2119/drd_data/cluster_labels_final.csv', index_col=0).to_numpy().flatten()
        needs_scaling = True
    elif dataset_name == "tasic":
        X = np.load('/share/ctn/users/bnc2119/drd_data/preprocessed-data.npy')
        labs = np.load('/share/ctn/users/bnc2119/drd_data/tasic_cluster_labels.npy', allow_pickle = True)
    elif dataset_name == "macaque":
        data = pd.read_csv('/share/ctn/users/bnc2119/drd_data/macaque1_pc100.csv')
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

        if id_set is not None:
            assert len(id_set) <= X_train.shape[0], "id_set needs to be smaller than number of samples"
            X_train = X_train[id_set]
            
        if needs_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test) if X_test is not None else None
        return X_train, X_test, labs_train, labs_test
    
    if test_size > 0:
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=seed)
    else: X_train = X
    
    if id_set is not None:
        assert len(id_set) <= X_train.shape[0], "id_set needs to be smaller than number of samples"
        X_train = X_train[id_set]

    if needs_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test) if X_test is not None else None
    return X_train, X_test


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