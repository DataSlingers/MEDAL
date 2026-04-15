"""
High-level teacher-hyperparameter sweep orchestrator.

Typical usage::

    from medal.sweep import run_teacher_sweep

    results = run_teacher_sweep(
        X_train,
        output_dir="experiments/my_run",
        teacher="tsne",
        arch_config=best_config,   # from tune_architecture()
    )
    df = results.load_metrics(X_train, X_val, X_test)
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

from medal._paths import (
    TEACHER_PARAM_KEY,
    teacher_embedding_path,
    teacher_norm_path,
    student_ckpt_path,
)
from medal.teacher import build_param_grid, get_teacher_embeddings
from medal.normalizer import GlobalEmbeddingNormalizer
from medal.model import MEDAL
from medal.io import compute_losses, load_model


# ------------------------------------------------------------------
# SweepResults
# ------------------------------------------------------------------

@dataclass
class SweepResults:
    """
    Lightweight container returned by :func:`run_teacher_sweep`.

    Attributes
    ----------
    output_dir : Path
        Root directory where embeddings and checkpoints are stored.
    teacher : str
        Teacher algorithm used in the sweep.
    param_name : str
        Name of the swept hyperparameter (e.g. ``"perplexity"``).
    param_values : list
        Sorted list of unique hyperparameter values.
    seeds : list of int
        Random seeds used.
    n_components : int
        Embedding dimensionality.
    arch_config : dict
        Architecture config used to train every student model.
    """
    output_dir: Path
    teacher: str
    param_name: str
    param_values: List
    seeds: List[int]
    n_components: int = 2
    arch_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Write metadata to ``output_dir/sweep_metadata.json``."""
        meta = {
            "teacher": self.teacher,
            "param_name": self.param_name,
            "param_values": [int(v) if hasattr(v, "item") else v for v in self.param_values],
            "seeds": list(self.seeds),
            "n_components": self.n_components,
            "arch_config": _serialisable(self.arch_config),
        }
        path = self.output_dir / "sweep_metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, output_dir: str | Path) -> "SweepResults":
        """Reconstruct from a previously saved ``sweep_metadata.json``."""
        output_dir = Path(output_dir)
        with open(output_dir / "sweep_metadata.json") as f:
            meta = json.load(f)
        return cls(
            output_dir=output_dir,
            teacher=meta["teacher"],
            param_name=meta["param_name"],
            param_values=meta["param_values"],
            seeds=meta["seeds"],
            n_components=meta.get("n_components", 2),
            arch_config=meta.get("arch_config", {}),
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def load_metrics(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Load every student checkpoint and compute reconstruction / distillation
        metrics for each (param_value, seed, split) combination.

        Parameters
        ----------
        X_train, X_val, X_test : np.ndarray
            The data splits used during the sweep.  Only splits that are not
            ``None`` are evaluated.

        Returns
        -------
        pd.DataFrame
            Columns: ``[param_name, seed, split, recon_mse, distill_mse]``.
        """
        ac = self.arch_config
        splits = {"Train": X_train}
        if X_val is not None:
            splits["Val"] = X_val
        if X_test is not None:
            splits["Test"] = X_test

        rows = []
        for param_val in self.param_values:
            tc = _tc_from_param(self.teacher, self.param_name, param_val, self.n_components)
            for seed in self.seeds:
                # Load teacher embedding + normaliser
                emb_path = teacher_embedding_path(
                    self.output_dir, "data", self.teacher, tc, seed
                )
                if not emb_path.exists():
                    continue
                Z_raw = np.load(emb_path)
                norm_p = teacher_norm_path(emb_path)
                normalizer = GlobalEmbeddingNormalizer.load(norm_p) if norm_p.exists() else None
                Z_norm = normalizer.transform(Z_raw) if normalizer else Z_raw

                # Load student checkpoint
                prefix = _student_prefix(self.teacher, tc, seed)
                ckpt = student_ckpt_path(self.output_dir, prefix)
                if not ckpt.exists():
                    continue
                model = load_model(
                    ckpt,
                    input_dim=X_train.shape[1],
                    hidden_dims=tuple(ac.get("hidden_dims", [128, 128])),
                    latent_dim=self.n_components,
                    activation=ac.get("activation", "SELU"),
                    use_batchnorm=ac.get("use_batchnorm", False),
                    dropout_rate=ac.get("dropout_rate", 0.0),
                )

                for split_name, X_split in splits.items():
                    recon_mse, distill_mse = compute_losses(model, X_split, Z_norm)
                    rows.append({
                        self.param_name: param_val,
                        "seed": seed,
                        "split": split_name,
                        "recon_mse": recon_mse,
                        "distill_mse": distill_mse,
                        # alias used by selection / plotting helpers
                        "recon_loss": recon_mse,
                    })

        return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def run_teacher_sweep(
    X: np.ndarray,
    output_dir: str | Path,
    teacher: str,
    arch_config: dict,
    param_grid: Optional[List[dict]] = None,
    latent_dim: int = 2,
    val_size: float = 0.2,
    seeds: List[int] = None,
    normalize_teacher: bool = True,
    distill_bands: Optional[list] = None,
    resources_per_trial: Optional[dict] = None,
    verbose: bool = True,
) -> SweepResults:
    """
    Run MEDAL over a teacher-hyperparameter sweep and return a
    :class:`SweepResults` object.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training data.
    output_dir : str or Path
        All embeddings, checkpoints, and metadata are written here.
    teacher : str
        Teacher algorithm (``"tsne"``, ``"umap"``, ``"pca"``, etc.).
    arch_config : dict
        MEDAL architecture config — output of :func:`~medal.tuning.get_best_config`
        or a hand-crafted dict with keys such as ``hidden_dims``, ``lambda_d``,
        ``lr``, ``batch_size``, ``max_epochs``, etc.
    param_grid : list of dict, optional
        List of teacher hyperparameter configs.  Defaults to
        :func:`~medal.teacher.build_param_grid` with standard log-spaced values.
    latent_dim : int
        Target embedding dimensionality.
    val_size : float
        Fraction of *X* held out as a validation set (used during training).
    seeds : list of int
        Random seeds for the sweep.  Defaults to ``[0]``.
    normalize_teacher : bool
        Whether to normalise teacher embeddings with
        :class:`~medal.normalizer.GlobalEmbeddingNormalizer` before distillation.
    distill_bands : list of (float, float), optional
        Target distillation-loss bands for stability-based early stopping.
    resources_per_trial : dict, optional
        Ray Tune resource spec, e.g. ``{"cpu": 4, "gpu": 1}``.
    verbose : bool
        Show progress information.

    Returns
    -------
    results : SweepResults
    """
    from ray import tune

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds or [0]

    if param_grid is None:
        param_grid = build_param_grid(teacher, n_components=latent_dim)

    param_name = TEACHER_PARAM_KEY.get(teacher, "param")

    # --- split data ---
    X_train, X_val = train_test_split(X, test_size=val_size, random_state=0)

    # Save splits so the trainable function can load them
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy", X_val)

    # --- precompute teacher embeddings for every (tc, seed) pair ---
    for tc in param_grid:
        for seed in seeds:
            _precompute_one_embedding(
                output_dir, "data", teacher, tc, seed, X_train,
                normalize=normalize_teacher, verbose=verbose,
            )

    # --- build Ray Tune config and run ---
    base_cfg = {
        **arch_config,
        "output_dir":     str(output_dir),
        "dataset_name":   "data",
        "teacher":        teacher,
        "latent_dim":     latent_dim,
        "normalize_teacher": normalize_teacher,
        "distill_bands":  distill_bands or [(1e-12, 9e-6)],
        "verbose":        False,
        "teacher_config": tune.grid_search(param_grid),
        "seed":           tune.grid_search(seeds),
    }

    resources = resources_per_trial or {"cpu": 4, "gpu": 1}

    tune.run(
        _sweep_trainable,
        name="medal_teacher_sweep",
        num_samples=1,
        resources_per_trial=resources,
        config=base_cfg,
        verbose=1 if verbose else 0,
        max_failures=3,
        storage_path=str(output_dir / "ray_results"),
    )

    param_values = sorted({tc[param_name] for tc in param_grid if param_name in tc})
    results = SweepResults(
        output_dir=output_dir,
        teacher=teacher,
        param_name=param_name,
        param_values=param_values,
        seeds=seeds,
        n_components=latent_dim,
        arch_config=arch_config,
    )
    results.save()
    return results


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _precompute_one_embedding(
    output_dir, dataset_name, teacher, tc, seed, X_train,
    normalize=True, verbose=True,
):
    """Compute and cache a single teacher embedding (skips if already exists)."""
    path = teacher_embedding_path(output_dir, dataset_name, teacher, tc, seed)
    if path.exists():
        if verbose:
            print(f"Skipping (exists): {path.name}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    n_components = tc.get("n_components", 2)

    # Build kwargs for get_teacher_embeddings (without n_components key)
    kw = {k: v for k, v in tc.items() if k != "n_components"}

    Z = get_teacher_embeddings(teacher, X_train, n_components=n_components, **kw)

    with open(path, "xb") as f:
        np.save(f, Z)

    if normalize:
        normalizer = GlobalEmbeddingNormalizer().fit(Z)
        normalizer.save(teacher_norm_path(path))

    if verbose:
        print(f"Saved: {path.name}")


def _sweep_trainable(config):
    """Ray Tune trainable for a single (teacher_config, seed) trial."""
    output_dir = Path(config["output_dir"])
    tc          = config["teacher_config"]
    seed        = config["seed"]
    teacher     = config["teacher"]
    latent_dim  = config.get("latent_dim", 2)
    dataset     = config.get("dataset_name", "data")
    norm        = config.get("normalize_teacher", True)

    X_train = np.load(output_dir / "X_train.npy")

    emb_path = teacher_embedding_path(output_dir, dataset, teacher, tc, 0)
    Z_raw = np.load(emb_path)
    if norm:
        normalizer = GlobalEmbeddingNormalizer.load(teacher_norm_path(emb_path))
        Z_train = normalizer.transform(Z_raw)
    else:
        Z_train = Z_raw

    student_kw = {
        "input_dim":         X_train.shape[1],
        "latent_dim":        latent_dim,
        "hidden_dims":       config.get("hidden_dims", (128, 128)),
        "activation":        config.get("activation", "SELU"),
        "bottleneck_activation": config.get("bottleneck_activation", None),
        "final_activation":  config.get("final_activation", None),
        "lambda_d":          config.get("lambda_d", 10),
        "lr":                config.get("lr", 1e-3),
        "epochs":            config.get("max_epochs", 5000),
        "batch_size":        config.get("batch_size", 256),
        "warmup":            config.get("warmup", 0),
        "eta_min":           config.get("eta_min", 1e-7),
        "use_batchnorm":     config.get("use_batchnorm", False),
        "dropout_rate":      config.get("dropout_rate", 0.1),
        "adamw_weight_decay":config.get("adamw_weight_decay", 1e-5),
        "factor":            config.get("t_factor", 0.9),
        "patience":          config.get("t_patience", 20),
        "criterion":         nn.MSELoss,
    }

    student = MEDAL(**student_kw)
    prefix  = _student_prefix(teacher, tc, seed)

    student.fit(
        X_train, Z_train,
        verbose=config.get("verbose", False),
        target_bands=config.get("distill_bands", [(1e-12, 9e-6)]),
        stability_window=20,
        epsilon_distill=1e-7,
        epsilon_recon=1e-3,
        patience=50,
        return_on_stable=True,
        save_dir=str(output_dir),
        prefix=prefix,
    )


def _student_prefix(teacher, tc, seed):
    """Build a short, unique prefix string for checkpoint filenames."""
    n = tc.get("n_components", 2)
    param_name = TEACHER_PARAM_KEY.get(teacher, "param")
    param_val  = tc.get(param_name, "")
    return f"medal_{teacher}{n}_{param_val}_seed{seed}"


def _tc_from_param(teacher, param_name, param_val, n_components):
    """Reconstruct a minimal tc dict from a single param value."""
    tc = {"n_components": n_components, param_name: param_val}
    if teacher == "umap":
        tc["min_dist"] = 0.1
    return tc


def _serialisable(obj):
    """Recursively convert non-JSON-serialisable values (e.g. np.int64)."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialisable(v) for v in obj]
    if hasattr(obj, "item"):          # numpy scalars
        return obj.item()
    if callable(obj):
        return str(obj)
    return obj
