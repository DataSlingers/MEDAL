import numpy as np
from typing import Tuple, Dict, Optional, Union
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def _ensure_images_shape(X: np.ndarray) -> Tuple[np.ndarray, bool]:
    X = np.asarray(X)
    if X.ndim == 2 and X.shape[1] == 784:
        X_img = X.reshape(-1, 28, 28)
        was_flattened = True
    elif X.ndim == 3 and X.shape[1:] == (28, 28):
        X_img = X
        was_flattened = False
    else:
        raise ValueError("X must be shape (N,784) or (N,28,28).")
    X_img = X_img.astype(np.float32)
    if X_img.max() > 1.0:
        X_img = X_img / 255.0
    X_img = np.clip(X_img, 0.0, 1.0)
    return X_img, was_flattened

def _restore_shape(X_img: np.ndarray, was_flattened: bool) -> np.ndarray:
    return X_img.reshape(len(X_img), -1) if was_flattened else X_img

# -------------------------
# Mask helpers
# -------------------------
def _normalize_mask(
    pixel_mask: Optional[np.ndarray],
    mask_frac: float,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    Returns a boolean mask of shape (28,28), or None meaning "all pixels".
    If pixel_mask is provided, it is converted to boolean and reshaped if needed.
    If not provided, a random mask is generated using mask_frac.
    """
    if pixel_mask is None:
        if mask_frac >= 1.0:
            return None  # treat as "all pixels"
        if not (0.0 < mask_frac < 1.0):
            raise ValueError("mask_frac must be in (0,1] when pixel_mask is None.")
        m = rng.random((28, 28)) < mask_frac
        return m.astype(bool)

    m = np.asarray(pixel_mask)
    if m.shape == (784,):
        m = m.reshape(28, 28)
    if m.shape != (28, 28):
        raise ValueError("pixel_mask must have shape (28,28) or (784,).")
    return m.astype(bool)

def _maybe_per_image_mask(
    base_mask: Optional[np.ndarray],
    mask_mode: str,
    mask_frac: float,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    If mask_mode == 'per_image', ignore base_mask and sample a new mask.
    If 'shared', return base_mask as-is (could be None meaning all pixels).
    """
    if mask_mode not in ("shared", "per_image"):
        raise ValueError("mask_mode must be 'shared' or 'per_image'.")
    if mask_mode == "shared":
        return base_mask
    # per_image
    if mask_frac >= 1.0:
        return None
    m = rng.random((28, 28)) < mask_frac
    return m.astype(bool)

# -------------------------
# Masked transforms
# -------------------------
def _apply_noise(img: np.ndarray, sigma: float, rng: np.random.Generator, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
        return np.clip(img + noise, 0.0, 1.0)
    out = img.copy()
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    out[mask] = out[mask] + noise[mask]
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _apply_brightness_contrast(img: np.ndarray, alpha: float, beta: float, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return np.clip(alpha * img + beta, 0.0, 1.0).astype(np.float32)
    out = img.copy()
    out[mask] = alpha * out[mask] + beta
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _apply_smoothing(img: np.ndarray, n_iter: int = 1, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute smoothing update as before, but only write back into masked pixels.
    Note: the blur uses full neighborhood context (including unmasked pixels),
    but only masked pixels are changed.
    """
    if mask is None:
        out = img.copy()
        for _ in range(n_iter):
            up    = np.roll(out, shift=-1, axis=0)
            down  = np.roll(out, shift=+1, axis=0)
            left  = np.roll(out, shift=-1, axis=1)
            right = np.roll(out, shift=+1, axis=1)
            out = (out + up + down + left + right) / 5.0
        return out.astype(np.float32)

    out = img.copy()
    tmp = out.copy()
    for _ in range(n_iter):
        up    = np.roll(tmp, shift=-1, axis=0)
        down  = np.roll(tmp, shift=+1, axis=0)
        left  = np.roll(tmp, shift=-1, axis=1)
        right = np.roll(tmp, shift=+1, axis=1)
        blurred = (tmp + up + down + left + right) / 5.0
        # only update masked pixels
        tmp = tmp.copy()
        tmp[mask] = blurred[mask]
    out = tmp
    return out.astype(np.float32)

def _make_checkerboard(h: int = 28, w: int = 28, amp: float = 0.10) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    pat = ((xx + yy) % 2) * 2 - 1
    return (amp * pat).astype(np.float32)

def _apply_checker_watermark(
    img: np.ndarray,
    pattern: np.ndarray,
    clip: bool = True,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if mask is None:
        out = img + pattern
    else:
        out = img.copy()
        out[mask] = out[mask] + pattern[mask]
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)

# -------------------------
# Main function
# -------------------------
def make_artificial_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_fracs: Dict[str, float] = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
    noise_sigma: float = 0.20,
    alpha: float = 1.20,
    beta: float = 0.05,
    smooth_iters: int = 50,
    checker_amp: float = 0.10,
    checker_batch: str = "D",
    checker_clip: bool = True,
    seed: int = 0,
    stratify_by_label: bool = True,
    # --- NEW: pixel subset controls ---
    pixel_mask: Optional[np.ndarray] = None,   # (28,28) or (784,); True=can change
    mask_frac: float = 1.0,                    # used only if pixel_mask is None OR mask_mode="per_image"
    mask_mode: str = "shared",                 # "shared" or "per_image"
    mask_seed: int = 0,                        # separate seed for mask generation
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(sum(batch_fracs.values()) - 1.0) < 1e-6, "batch_fracs must sum to 1.0"
    X_img, was_flattened = _ensure_images_shape(X)
    y = np.asarray(y)
    assert len(X_img) == len(y), "X and y must have same length"

    rng = np.random.default_rng(seed)
    rng_mask = np.random.default_rng(mask_seed)

    N = len(X_img)
    batch_ids = np.empty(N, dtype=object)

    def assign_indices(idxs: np.ndarray, fracs: Dict[str, float]) -> Dict[str, np.ndarray]:
        n = len(idxs)
        counts = {k: int(round(fracs[k] * n)) for k in fracs}
        diff = n - sum(counts.values())
        if diff != 0:
            k_adj = max(fracs, key=fracs.get)
            counts[k_adj] += diff
        rng.shuffle(idxs)
        splits = {}
        start = 0
        for k in fracs:
            end = start + counts[k]
            splits[k] = idxs[start:end]
            start = end
        return splits

    if stratify_by_label:
        for digit in np.unique(y):
            idxs = np.where(y == digit)[0]
            splits = assign_indices(idxs, batch_fracs)
            for k, ss in splits.items():
                batch_ids[ss] = k
    else:
        idxs = np.arange(N)
        splits = assign_indices(idxs, batch_fracs)
        for k, ss in splits.items():
            batch_ids[ss] = k

    if checker_batch not in batch_fracs:
        raise ValueError(f"checker_batch={checker_batch!r} must be one of {list(batch_fracs.keys())}")
    checker_pattern = _make_checkerboard(28, 28, amp=checker_amp)

    # base mask (used if mask_mode="shared")
    base_mask = _normalize_mask(pixel_mask, mask_frac=mask_frac, rng=rng_mask)

    X_out = np.empty_like(X_img, dtype=np.float32)
    for b in ["A", "B", "C", "D"]:
        idxs = np.where(batch_ids == b)[0]
        if len(idxs) == 0:
            continue

        for ii in idxs:
            im = X_img[ii]
            mask = _maybe_per_image_mask(base_mask, mask_mode=mask_mode, mask_frac=mask_frac, rng=rng_mask)

            if b == checker_batch:
                X_out[ii] = _apply_checker_watermark(im, checker_pattern, clip=checker_clip, mask=mask)
            elif b == "A":
                X_out[ii] = im
            elif b == "B":
                X_out[ii] = _apply_noise(im, noise_sigma, rng, mask=mask)
            elif b == "C":
                X_out[ii] = _apply_brightness_contrast(im, alpha, beta, mask=mask)
            elif b == "D":
                X_out[ii] = _apply_smoothing(im, n_iter=smooth_iters, mask=mask)
            else:
                raise ValueError(f"Unknown batch key {b}")

    X_aug = _restore_shape(X_out, was_flattened)
    y_bio = y.copy()
    y_batch = batch_ids.astype(str)
    return X_aug, y_bio, y_batch


# Quantitative comparison of *chosen embeddings* (MEDAL vs EMBEDR vs PCS …)
# Metrics:
#   (1) Linear probe (logreg) on train embedding -> test embedding: accuracy + macro-F1 + per-group accuracy
#   (2) kNN label purity (neighborhood label consistency) on train/test embeddings

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import log_loss


def _as_1d_array(y):
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.reshape(-1)
    return y


def linear_probe_eval(Z,
    y,
    cv: int = 5,
    C: float = 1.0,
    max_iter: int = 1000,
    n_jobs: int | None = None,
    random_state: int | None = None,
    return_estimators: bool = False,
):
    """
    Train a simple linear probe on embeddings and evaluate on test.
    Returns:
      overall dict + per-group accuracy Series
    """
    Z = np.asarray(Z)
    y = _as_1d_array(y)

    # Encode labels to ints for sklearn
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_

    estimator = RandomForestClassifier(random_state=0)
#     estimator = MLPClassifier(random_state = 1, max_iter = 1000)

    # Create a stratified splitter if cv is an integer
    if isinstance(cv, int):
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        splitter = cv

    # Define scoring metrics. Using make_scorer to compute macro‑F1.
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": make_scorer(f1_score, average="macro"),
    }
    cv_results = cross_validate(
        estimator,
        Z,
        y_enc,
        cv=splitter,
        scoring=scoring,
        return_train_score=True,
        return_estimator=return_estimators,
        return_indices=True,
        n_jobs=n_jobs,
    )

    # Aggregate overall metrics
    def _mean_std(key: str) -> Tuple[float, float]:
        arr = np.asarray(cv_results[key])
        return float(arr.mean()), float(arr.std())

    # Compute mean and std for each metric
    train_acc_mean, train_acc_std = _mean_std("train_accuracy")
    train_f1_mean, train_f1_std = _mean_std("train_macro_f1")
    test_acc_mean, test_acc_std = _mean_std("test_accuracy")
    test_f1_mean, test_f1_std = _mean_std("test_macro_f1")

    summary = {
        "train_accuracy": train_acc_mean,
        "train_accuracy_std": train_acc_std,
        "train_macro_f1": train_f1_mean,
        "train_macro_f1_std": train_f1_std,
        "test_accuracy": test_acc_mean,
        "test_accuracy_std": test_acc_std,
        "test_macro_f1": test_f1_mean,
        "test_macro_f1_std": test_f1_std,
    }

    if return_estimators:
        per_group_acc: Dict[str, List[float]] = {cls: [] for cls in class_names}
        # cv_results['estimator'] is a list of estimators; cv_results['indices']['test']
        # contains validation indices for each fold
        for est, test_idx in zip(cv_results["estimator"], cv_results["indices"]["test"]):
            y_test = y_enc[test_idx]
            preds = est.predict(Z[test_idx])
            for c in np.unique(y_test):
                mask = y_test == c
                acc = accuracy_score(y_test[mask], preds[mask])
                per_group_acc[class_names[c]].append(acc)
        per_group_acc_mean = pd.Series({cls: float(np.mean(vals)) for cls, vals in per_group_acc.items()})
        per_group_acc_std = pd.Series({cls: float(np.std(vals)) for cls, vals in per_group_acc.items()})
    else:
        per_group_acc_mean = pd.Series(dtype=float)
        per_group_acc_std = pd.Series(dtype=float)

    return summary, per_group_acc_mean.sort_index(), per_group_acc_std.sort_index()


def evaluate_methods(
    embeddings_by_method,
    y_train,
    y_test,
    cv: int = 5,
    C: float = 1.0,
    max_iter: int = 1000,
    n_jobs: int | None = None,
    random_state: int | None = None,
    return_estimators: bool = False,
    id_sets = None,
):
    """
    embeddings_by_method: dict
      method_name -> (Z_train, Z_test)
      where Z_* are arrays (N x d) and (M x d)

    Returns:
      summary_df: pd.DataFrame (overall metrics per method)
      per_group_acc_df: pd.DataFrame (per-group acc; rows=group, cols=method)
      purity_group_train_df / purity_group_test_df: pd.DataFrame (per-group purity)
    """
    y_train_arr = _as_1d_array(y_train)
    y_test_arr = _as_1d_array(y_test) if y_test is not None else None

    per_group_mean: Dict[str, pd.Series] = {}
    per_group_std: Dict[str, pd.Series] = {}
    rows = []

    # Loop over embedding methods
    for method, (Ztr, Zte) in embeddings_by_method.items():
        
        Z_concat = np.concatenate([Ztr, Zte], axis=0) if Zte is not None else Ztr
        if id_sets is None:
            y_concat = np.concatenate([y_train_arr, y_test_arr], axis=0) if y_test_arr is not None else y_train_arr
        else:
            y_concat = np.concatenate([y_train_arr[id_sets[method]], y_test_arr], axis=0) if y_test_arr is not None else y_train_arr[id_sets[method]]
        # Evaluate via cross‑validation on the full set of embeddings
        summary, pg_mean, pg_std = linear_probe_eval(
            Z_concat,
            y_concat,
            cv=cv,
            C=C,
            max_iter=max_iter,
            n_jobs=n_jobs,
            random_state=random_state,
            return_estimators=return_estimators,
        )
        # Store summary metrics
        row = {"method": method}
        row.update(summary)
        rows.append(row)
        # Store per‑group stats
        if return_estimators:
            per_group_mean[method] = pg_mean
            per_group_std[method] = pg_std

    summary_df = pd.DataFrame(rows).set_index("method").sort_values(
        "test_macro_f1", ascending=False
    )

    # Construct per‑group dataframes if available
    if return_estimators and per_group_mean:
        per_group_acc_mean_df = pd.DataFrame(per_group_mean).sort_index(axis=0)
        per_group_acc_std_df = pd.DataFrame(per_group_std).sort_index(axis=0)
    else:
        per_group_acc_mean_df = pd.DataFrame()
        per_group_acc_std_df = pd.DataFrame()

    return summary_df, per_group_acc_mean_df, per_group_acc_std_df

from sklearn.manifold import trustworthiness

def mc_subsample_trustworthiness(
    X_train, Z_train,
    X_test = None, Z_test = None,
    n_neighbors=15,
    metric="euclidean",
    n_mc=50,
    frac=0.8,
    random_state=None,
):
    """
    Monte Carlo subsampling (WITHOUT replacement).
    Returns mean/std/stderr of trustworthiness on train and test.
    """
    rng = np.random.default_rng(random_state)

    X_train = np.asarray(X_train); Z_train = np.asarray(Z_train)
    n_tr = len(X_train)
    m_tr = max(2 * n_neighbors + 1, int(np.floor(frac * n_tr)))
    if X_test is not None: 
        X_test  = np.asarray(X_test);  Z_test  = np.asarray(Z_test)
        n_te = len(X_test)
        m_te = max(2 * n_neighbors + 1, int(np.floor(frac * n_te)))

    tr_scores, te_scores = [], []

    for _ in range(n_mc):
        idx_tr = rng.choice(n_tr, size=m_tr, replace=False)

        tr_scores.append(trustworthiness(X_train[idx_tr], Z_train[idx_tr],
                                         n_neighbors=n_neighbors, metric=metric))

        if X_test is not None:
            idx_te = rng.choice(n_te, size=m_te, replace=False)
            te_scores.append(trustworthiness(X_test[idx_te], Z_test[idx_te],
                                             n_neighbors=n_neighbors, metric=metric))

    tr_scores = np.asarray(tr_scores)
    if X_test is not None:
        te_scores = np.asarray(te_scores)

    if X_test is not None:
        return {
            "train_mean": float(tr_scores.mean()),
#             "train_std": float(tr_scores.std(ddof=1)),
#             "train_stderr": float(tr_scores.std(ddof=1) / np.sqrt(n_mc)),
            "test_mean": float(te_scores.mean()),
#             "test_std": float(te_scores.std(ddof=1)),
#             "test_stderr": float(te_scores.std(ddof=1) / np.sqrt(n_mc)),
            "n_mc": int(n_mc),
            "frac": float(frac),
        }
    else:
        return {
            "train_mean": float(tr_scores.mean()),
#             "train_std": float(tr_scores.std(ddof=1)),
#             "train_stderr": float(tr_scores.std(ddof=1) / np.sqrt(n_mc)),
            "n_mc": int(n_mc),
            "frac": float(frac),
        }

def evaluate_methods_trustworthiness_mc(
    X_train, X_test,
    embeddings_by_method,
    n_neighbors=15,
    metric="euclidean",
    n_mc=50,
    frac=0.8,
    random_state=0,
):
    rows = []
    for method, (Ztr, Zte) in embeddings_by_method.items():
        out = mc_subsample_trustworthiness(
            X_train, Ztr, X_test, Zte,
            n_neighbors=n_neighbors, metric=metric,
            n_mc=n_mc, frac=frac, random_state=random_state
        )
        if Zte is not None:
            rows.append({
                "method": method,
                "train_tw_mean": out["train_mean"],
#                 "train_tw_stderr": out["train_stderr"],
                "test_tw_mean": out["test_mean"],
#                 "test_tw_stderr": out["test_stderr"],
            })
        else:
            rows.append({
                "method": method,
                "train_tw_mean": out["train_mean"],
#                 "train_tw_stderr": out["train_stderr"],
            })

    return (pd.DataFrame(rows)
            .set_index("method")
            .sort_values("train_tw_mean", ascending=False))