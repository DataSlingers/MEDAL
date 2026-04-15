import os
import glob
import time
import numpy as np
import pandas as pd

from EMBEDR import EMBEDR


# ============================================================
# 1) n_neighbors grids (by dataset)  <-- replaces perplexity grids
# ============================================================
def get_n_neighbors_list(df_name: str):
    key = df_name.strip().upper()
    if key == "MNIST":
        return [5, 6, 9, 13, 18, 25, 35, 49, 69, 96, 134, 186, 258, 359, 499]
    elif key == "HYDRA":
        return [5, 9, 18, 36, 71, 139, 271, 528, 1027, 2000]
    elif key == "TASIC":
        return [5, 9, 18, 36, 71, 139, 271, 528, 1027, 2000]
    elif key == "ASTRO":
        return [5, 6, 9, 13, 18, 25, 35, 49, 69, 96, 134, 186, 258, 359, 499]
    else:
        raise ValueError(f"Unknown dataset key: {df_name}. Supported: MNIST, HYDRA, TASIC, ASTRO")


def filter_n_neighbors(neighbors, n):
    # UMAP constraint: 2 <= n_neighbors <= n-1
    neighbors2 = [k for k in neighbors if (k >= 2) and (k <= (n - 1))]
    if len(neighbors2) == 0:
        raise ValueError(f"No valid n_neighbors after filtering by [2, n-1]. n={n}.")
    return neighbors2


# ============================================================
# 2) Robust p-value extractor (handles EMBEDR version differences)
# ============================================================
def extract_pvals_from_embedr_obj(emb_obj, n_points: int):
    """
    Try common p-value attribute names; fallback scans __dict__ for a length-n vector in [0,1].
    """
    candidate_attrs = ["pvals", "p_vals", "p_values", "pValues", "pvals_", "p_vals_", "p_values_"]
    for a in candidate_attrs:
        if hasattr(emb_obj, a):
            v = getattr(emb_obj, a)
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == n_points:
                return np.asarray(v, dtype=float).reshape(-1)

    for k, v in getattr(emb_obj, "__dict__", {}).items():
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == n_points:
            arr = np.asarray(v, dtype=float)
            if arr.ndim == 1:
                finite = arr[np.isfinite(arr)]
                if finite.size > 0:
                    frac_in_01 = np.mean((finite >= 0.0) & (finite <= 1.0))
                    if frac_in_01 > 0.8:
                        return arr

    keys = list(getattr(emb_obj, "__dict__", {}).keys())
    raise RuntimeError(
        "Could not locate per-point p-values on EMBEDR object.\n"
        f"Available keys (sample): {keys[:80]}"
    )


# ============================================================
# 3) Dataset key parsing: "hydra_train.csv" -> "HYDRA"
# ============================================================
def dataset_key_from_filename(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]     # hydra_train
    key = stem.split("_")[0].upper()     # HYDRA
    return key


# ============================================================
# 4) One dataset runner (writes the same 3 CSV outputs)
# ============================================================
def run_one_dataset(
    csv_path: str,
    out_root: str = "results_umap",
    n_data_embed: int = 3,
    n_null_embed: int = 1,
    n_jobs: int = -1,
    verbose: int = 0,
    do_cache: bool = False,
    dataset = None,
    load_this_seed = 0
):
    base = os.path.basename(csv_path)          # hydra_train.csv
    stem = os.path.splitext(base)[0]           # hydra_train
    key = dataset_key_from_filename(csv_path)  # HYDRA
    order = base[:-4].split('_')[-1]
    print("order is: ", order)
    
    dataset_out = os.path.join(out_root, stem)
    os.makedirs(dataset_out, exist_ok=True)

    print("\n==============================")
    print(f"Dataset file: {base}")
    print(f"Dataset key : {key}")
    print("==============================")

    # Read CSV
    X_df = pd.read_csv(csv_path)

    # Drop label/split columns (case-insensitive)
    drop_cols = [c for c in X_df.columns if c.lower() in ("label", "split", "labels")]
    if len(drop_cols) > 0:
        print("Dropped label/split columns:", ", ".join(drop_cols))
        X_df = X_df.drop(columns=drop_cols)

    # Keep only numeric columns (mirror R “safety”)
    num_df = X_df.select_dtypes(include=[np.number])
    dropped_nonnum = [c for c in X_df.columns if c not in num_df.columns]
    if len(dropped_nonnum) > 0:
        print("Dropping non-numeric columns:", ", ".join(dropped_nonnum))
    X_df = num_df

    # Convert to numpy
    X = X_df.to_numpy(dtype=float, copy=False)
    n = X.shape[0]
    p = X.shape[1]
    print(f"n = {n}, p = {p}")

    # n_neighbors (filtered by n)
    neighbors = filter_n_neighbors(get_n_neighbors_list(key), n=n)
    print("Using n_neighbors:", neighbors)

    all_point_rows = []
    summary_rows = []

    for k in neighbors:
        print(f"  -> n_neighbors = {k} ({neighbors.index(k)+1}/{len(neighbors)})")

        t0 = time.perf_counter()
        emb = EMBEDR(
            project_name=f"{stem}_nn_{k}",
            project_dir=dataset_out,   # safe place to write if caching ever turns on
            dataset = dataset,
            order = order,
            DRA="umap",                # CHANGED: tsne -> umap
            DRA_params={},
            n_neighbors=int(k),        # CHANGED: perplexity -> n_neighbors
            n_data_embed=n_data_embed,
            n_null_embed=n_null_embed,
            load_this_seed = load_this_seed,
            n_jobs=-1,
            verbose=verbose,
            do_cache=do_cache,         # default False for cluster robustness
            random_state = load_this_seed
        )
        emb.fit(X)
        runtime = time.perf_counter() - t0

        pvals = extract_pvals_from_embedr_obj(emb, n_points=n)

        all_point_rows.append(pd.DataFrame({
            "n_neighbors": int(k),
            "point_id": np.arange(1, n + 1),
            "embedr_pval": pvals.astype(float),
        }))

        summary_rows.append({
            "n_neighbors": int(k),
            "mean_embedr_pval": float(np.nanmean(pvals)),
            "runtime_seconds": float(runtime),
        })

    scores_df = pd.concat(all_point_rows, ignore_index=True)
    elbow_df = pd.DataFrame(summary_rows).sort_values("n_neighbors").reset_index(drop=True)

    # Choose best n_neighbors by minimum mean p-value (NO elbow logic)
    best_k = int(elbow_df.loc[elbow_df["mean_embedr_pval"].idxmin(), "n_neighbors"])
    print(f"Chosen best n_neighbors (min mean p-value) = {best_k}")

    # Add ONLY the best-n_neighbors pvals back onto original numeric data
    X_out = X_df.copy()
    best_vec = (
        scores_df.loc[scores_df["n_neighbors"] == best_k]
        .sort_values("point_id")["embedr_pval"]
        .to_numpy()
    )
    X_out[f"embedr_pval_meanbest_{best_k}"] = best_vec
    X_out["embedr_meanbest_n_neighbors"] = best_k

    # Write outputs (same three CSVs)
    elbow_path = os.path.join(dataset_out, "elbow_df.csv")
    scores_path = os.path.join(dataset_out, "scores_per_point_all_n_neighbors.csv")
    xout_path = os.path.join(dataset_out, "X_with_best_scores.csv")

    elbow_df.to_csv(elbow_path, index=False)
    scores_df.to_csv(scores_path, index=False)
    X_out.to_csv(xout_path, index=False)

    return {
        "dataset": stem,
        "key": key,
        "best_n_neighbors": int(best_k),
        "out_dir": dataset_out,
    }


# ============================================================
# 5) Driver: scan input_dir for *_train.csv and run all
# ============================================================
if __name__ == "__main__":
    # Same convention as your R script: code directory with ../data
    input_dir = '/share/ctn/users/bnc2119/MEDAL/comparisons/data'
    for lts in [2,10]:
        out_root = "results_embedr_umap_seed" + str(lts)
        os.makedirs(out_root, exist_ok=True)
        csv_files = sorted(glob.glob(os.path.join(input_dir, "astro_train.csv"))) + sorted( 
            glob.glob(os.path.join(input_dir, "*_TRAIN.csv"))
        )
        # de-dup in case both patterns match
        csv_files = sorted(list(set(csv_files)))
    
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No *_train.csv files found in input_dir: {os.path.abspath(input_dir)}")
    
        print(f"Found {len(csv_files)} dataset(s) in {os.path.abspath(input_dir)}")
    
        run_log_rows = []
        for csv_path in csv_files:
            res = run_one_dataset(
                csv_path=csv_path,
                out_root=out_root,
                n_data_embed=1,
                n_null_embed=1,
                n_jobs=-1,
                verbose=1,
                do_cache=False,   # keep cluster runs simple/robust
                dataset = "astro",
                load_this_seed = lts
            )
            run_log_rows.append(res)
    
        run_log = pd.DataFrame(run_log_rows)
        run_log_path = os.path.join(out_root, "run_log.csv")
        run_log.to_csv(run_log_path, index=False)
        print("Wrote run log:", os.path.abspath(run_log_path))