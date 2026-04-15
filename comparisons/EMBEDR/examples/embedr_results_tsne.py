import os
import glob
import time
import numpy as np
import pandas as pd

from EMBEDR import EMBEDR


# ============================================================
# 1) Perplexity grids (same as before)
# ============================================================
def get_perplexity_list(df_name: str):
    key = df_name.strip().upper()
    if key == "MNIST":
        return [5, 11, 27, 62, 146, 341, 793, 1846]
    elif key == "HYDRA":
        return [5, 10, 23, 49, 107, 232, 499, 1077, 2320, 4999]
    elif key == "TASIC":
        return [5, 10, 24, 53, 116, 256, 564, 1241, 2729, 6000]
    elif key == "ASTRO":
        return [3, 4, 6, 8, 12, 18, 26, 55, 80, 115, 167, 240, 346, 499]
    else:
        raise ValueError(f"Unknown dataset key: {df_name}. Supported: MNIST, HYDRA, TASIC, ASTRO")


def filter_perplexities(perps, n):
    # common t-SNE constraint: perplexity < (n-1)/3
    max_safe = (n - 1) / 3.0
    perps2 = [p for p in perps if p < max_safe]
    if len(perps2) == 0:
        raise ValueError(f"No valid perplexities after filtering by (n-1)/3. n={n}.")
    return perps2


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
    key = stem.split("_")[0].upper()    # HYDRA
    return key


# ============================================================
# 4) One dataset runner (writes the same 3 CSV outputs)
# ============================================================
def run_one_dataset(
    csv_path: str,
    out_root: str = "results",
    n_data_embed: int = 3,
    n_null_embed: int = 1,
    n_jobs: int = -1,
    verbose: int = 0,
    do_cache: bool = False,
    dataset = None,
    load_this_seed = 0
):
    base = os.path.basename(csv_path)         # hydra_train.csv
    stem = os.path.splitext(base)[0]          # hydra_train
    key = dataset_key_from_filename(csv_path) # HYDRA

    dataset_out = os.path.join(out_root, stem)
    os.makedirs(dataset_out, exist_ok=True)

    print("\n==============================")
    print(f"Dataset file: {base}")
    print(f"Dataset key : {key}")
    print("==============================")

    # Read CSV
    X_df = pd.read_csv(csv_path)
    
    # order
    order = base[:-4].split('_')[-1]
    print("order is: ", order)

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
    print(X_df.head())
    X = X_df.to_numpy(dtype=float, copy=False)
    n = X.shape[0]
    p = X.shape[1]
    print(f"n = {n}, p = {p}")

    # Perplexities (filtered by n)
    perps = filter_perplexities(get_perplexity_list(key), n=n)
    print("Using perplexities:", perps)
    print("Dataset: ", dataset)

    all_point_rows = []
    summary_rows = []

    for perp in perps:
        print(f"  -> perplexity = {perp} ({perps.index(perp)+1}/{len(perps)})")

        t0 = time.perf_counter()
        emb = EMBEDR(
            project_name=f"{stem}_perp_{perp}",
            project_dir=dataset_out,   # safe place to write if caching ever turns on
            DRA="tsne",
            DRA_params={},
            perplexity=perp,
            dataset = dataset,
            order = order,
            load_this_seed = load_this_seed,
            n_data_embed=n_data_embed,
            n_null_embed=n_null_embed,
            n_jobs=n_jobs,
            verbose=verbose,
            do_cache=do_cache,         # default False for cluster robustness
            random_state = load_this_seed       # NEW
        )
        emb.fit(X)
        runtime = time.perf_counter() - t0

        pvals = extract_pvals_from_embedr_obj(emb, n_points=n)

        all_point_rows.append(pd.DataFrame({
            "perplexity": perp,
            "point_id": np.arange(1, n + 1),
            "embedr_pval": pvals.astype(float),
        }))

        summary_rows.append({
            "perplexity": perp,
            "mean_embedr_pval": float(np.nanmean(pvals)),
            "runtime_seconds": float(runtime),
        })

    scores_df = pd.concat(all_point_rows, ignore_index=True)
    elbow_df = pd.DataFrame(summary_rows).sort_values("perplexity").reset_index(drop=True)

    # Choose best perplexity by minimum mean p-value (NO elbow logic)
    best_perp = float(elbow_df.loc[elbow_df["mean_embedr_pval"].idxmin(), "perplexity"])
    print(f"Chosen best perplexity (min mean p-value) = {best_perp}")

    # Add ONLY the best-perplexity pvals back onto original numeric data
    X_out = X_df.copy()
    best_vec = (
        scores_df.loc[scores_df["perplexity"] == best_perp]
        .sort_values("point_id")["embedr_pval"]
        .to_numpy()
    )
    X_out[f"embedr_pval_meanbest_{int(best_perp)}"] = best_vec
    X_out["embedr_meanbest_perplexity"] = int(best_perp)

    # Write outputs (same three CSVs)
    elbow_path = os.path.join(dataset_out, "elbow_df.csv")
    scores_path = os.path.join(dataset_out, "scores_per_point_all_perplexities.csv")
    xout_path = os.path.join(dataset_out, "X_with_best_scores.csv")

    elbow_df.to_csv(elbow_path, index=False)
    scores_df.to_csv(scores_path, index=False)
    X_out.to_csv(xout_path, index=False)

    return {
        "dataset": stem,
        "key": key,
        "best_perplexity": int(best_perp),
        "out_dir": dataset_out,
    }


# ============================================================
# 5) Driver: scan input_dir for *_train.csv and run all
# ============================================================
if __name__ == "__main__":

    # Same convention as your R script: code directory with ../data
    #input_dir = os.path.join("..", "data")
    input_dir = '/share/ctn/users/bnc2119/MEDAL/comparisons/data'
    for lts in [2, 10]:
        out_root = "results_embedr_tsne_seed" + str(lts)
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