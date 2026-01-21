import scanpy as sc
import pandas as pd, numpy as np
from scipy.io import mmread
import anndata as ad

base = "/shared/share_mala/irchang/drd/Hydra_plain"

def combine_hydra_matrices():
    ############
    # Step 1: Combine matrices and metadata into AnnData and save as Hydra.h5ad
    ############
    # Load matrices (these are genes x cells from writeMM)
    counts_gxc = mmread(f"{base}/counts.mtx").tocsr()
    logn_gxc   = mmread(f"{base}/logcounts.mtx").tocsr()

    # Transpose to cells x genes (what AnnData expects)
    counts = counts_gxc.T.tocsr()
    logn   = logn_gxc.T.tocsr()

    # Load names
    genes = pd.read_csv(f"{base}/genes.tsv", header=None)[0].astype(str).values  # n_genes
    cells = pd.read_csv(f"{base}/cells.tsv", header=None)[0].astype(str).values  # n_cells

    # Load and align metadata
    meta = pd.read_csv(f"{base}/meta.csv").set_index("cell_id")
    # Ensure the same order as 'cells'
    meta = meta.reindex(cells)
    assert meta.index.notnull().all(), "Some cells in cells.tsv missing from meta.csv"
    assert meta.shape[0] == counts.shape[0] == logn.shape[0], "n_cells mismatch after reindex"

    # Final sanity checks
    assert counts.shape[1] == logn.shape[1] == len(genes), "n_genes mismatch"
    assert counts.shape[0] == logn.shape[0] == len(cells), "n_cells mismatch"

    # Build AnnData: X = logcounts, keep raw counts in a layer
    adata = ad.AnnData(
        X=logn, 
        obs=meta.reset_index(drop=False).rename(columns={"cell_id": "index"}).set_index("index"),
        var=pd.DataFrame(index=genes)
    )
    adata.layers["counts"] = counts

    # Optional: PCA if you exported it
    try:
        pca = pd.read_csv(f"{base}/pca_cells.csv", index_col=0)
        pca = pca.reindex(adata.obs_names)
        adata.obsm["X_pca"] = pca.values.astype(np.float32)
    except FileNotFoundError:
        pass

    adata.write_h5ad("Hydra.h5ad", compression="gzip")

def preprocess_hydra():
    ############
    # Step 2: Preprocess and save slim core AnnData
    ############
    # --- Preprocess exactly once ---
    adata = sc.read_h5ad("Hydra.h5ad")

    # QC (your example)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[(adata.obs.n_genes < 8000) & (adata.obs.nCount_RNA < 70000) & (adata.obs.nCount_RNA > 400)]
    adata.var['mt'] = adata.var_names.str.startswith('MT_')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5]

    # Normalize, log1p, HVGs, scale
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.write_h5ad("hydra_core_preprocessed_full_feats.h5ad", compression="lzf")
    sc.pp.highly_variable_genes(adata, min_mean=0.05, max_mean=4, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata)
    adata.write_h5ad("hydra_core_preprocessed_no_pca.h5ad", compression="lzf")

    # PCA + (optional) neighbors/leiden
    sc.tl.pca(adata, n_comps=40)        # keep extra PCs; you can still use first 19 later
    sc.pp.neighbors(adata, n_pcs=19)    # optional, but often handy
    sc.tl.leiden(adata, resolution=1.5) # optional

    # Save the slim, reusable core (contains X_pca, obs/var, etc.)
    adata.write_h5ad("hydra_core_preprocessed.h5ad", compression="lzf")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process Hydra dataset into AnnData")
    parser.add_argument("--step", type=int, choices=[1, 2], required=True, help="Step 1: combine matrices; Step 2: preprocess")
    args = parser.parse_args()
    if args.step == 1:
        combine_hydra_matrices()
    elif args.step == 2:
        preprocess_hydra()
    print("Done.")