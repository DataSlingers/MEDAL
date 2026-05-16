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
        X_test: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Load every student checkpoint and compute reconstruction / distillation
        metrics for each (param_value, seed, split) combination.

        Train and Val splits are loaded from the files saved by
        :func:`run_teacher_sweep` (``X_train.npy`` / ``X_val.npy`` in
        ``output_dir``).  Only the held-out test set needs to be supplied.

        Parameters
        ----------
        X_test : np.ndarray, optional
            Held-out test data.  When provided, a ``"Test"`` split is included
            in the returned DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``[param_name, seed, split, recon_mse, distill_mse,
            recon_loss]``.  ``distill_mse`` is populated for the Train split
            (where pre-computed teacher embeddings are available) and ``None``
            for Val / Test.
        """
        ac = self.arch_config

        # Load the exact train/val arrays used during the sweep from disk so
        # that their sizes match the pre-computed teacher embeddings.
        X_train_disk = np.load(self.output_dir / "X_train.npy")
        X_val_disk   = np.load(self.output_dir / "X_val.npy")

        # splits that don't have a pre-computed teacher embedding get Z=None
        splits = {
            "Train": (X_train_disk, True),   # (data, has_teacher_emb)
            "Val":   (X_val_disk,   False),
        }
        if X_test is not None:
            splits["Test"] = (X_test, False)

        rows = []
        for param_val in self.param_values:
            tc = _tc_from_param(self.teacher, self.param_name, param_val, self.n_components)

            # Load teacher embedding once per tc (shared across seeds)
            emb_path = teacher_embedding_path(
                self.output_dir, "data", self.teacher, tc
            )
            if not emb_path.exists():
                continue
            Z_raw = np.load(emb_path)
            norm_p = teacher_norm_path(emb_path)
            normalizer = GlobalEmbeddingNormalizer.load(norm_p) if norm_p.exists() else None
            Z_norm = normalizer.transform(Z_raw) if normalizer else Z_raw

            for seed in self.seeds:
                # Load student checkpoint
                prefix = _student_prefix(self.teacher, tc, seed)
                ckpt = student_ckpt_path(self.output_dir, prefix)
                if not ckpt.exists():
                    continue
                model = load_model(
                    ckpt,
                    input_dim=X_train_disk.shape[1],
                    hidden_dims=tuple(ac.get("hidden_dims", [128, 128])),
                    latent_dim=tc.get("n_components", self.n_components),
                    activation=ac.get("activation", "SELU"),
                    use_batchnorm=ac.get("use_batchnorm", False),
                    dropout_rate=ac.get("dropout_rate", 0.0),
                )

                for split_name, (X_split, has_emb) in splits.items():
                    recon_mse, distill_mse = compute_losses(
                        model, X_split, Z_norm if has_emb else None
                    )
                    rows.append({
                        self.param_name: param_val,
                        "seed": seed,
                        "split": split_name,
                        "recon_loss": recon_mse,
                        "distill_mse": distill_mse,
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
    ray_storage_path: Optional[str] = None,
    save_checkpoints: bool = True,
    mode: str = "sweep",
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
    ray_storage_path : str, optional
        Where Ray Tune writes its trial logs and internal checkpoints.
        Defaults to ``output_dir/ray_results``.  Set to a ``/tmp`` path to
        avoid filling shared cluster storage, e.g. ``"/tmp/ray_results"``.
    mode : str
        Label for the sweep mode, e.g. ``"teacher_sweep"``, ``"rank_sweep"``.
        Used as a suffix in the output filename:
        ``sweep_summary_{teacher}_{mode}.csv``.  Defaults to ``"sweep"``.
    verbose : bool
        Show progress information.

    Returns
    -------
    results : SweepResults
    """
    from ray import tune

    output_dir = Path(output_dir).resolve()
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

    # --- precompute one teacher embedding per tc (shared across all AE seeds) ---
    for tc in param_grid:
        _precompute_one_embedding(
            output_dir, "data", teacher, tc, X_train,
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
        "verbose":           False,
        "save_checkpoints":  save_checkpoints,
        "teacher_config":    tune.grid_search(param_grid),
        "seed":              tune.grid_search(seeds),
    }

    resources = resources_per_trial or {"cpu": 4, "gpu": 1}

    storage = ray_storage_path or str(output_dir / "ray_results")

    analysis = tune.run(
        _sweep_trainable,
        name="medal_teacher_sweep",
        num_samples=1,
        resources_per_trial=resources,
        config=base_cfg,
        verbose=1 if verbose else 0,
        max_failures=3,
        storage_path=storage,
    )

    # --- write ablation summary CSV ---
    _save_sweep_summary(analysis, output_dir, param_name, teacher, mode, verbose)

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
    output_dir, dataset_name, teacher, tc, X_train,
    normalize=True, verbose=True,
):
    """Compute and cache a single teacher embedding (skips if already exists).

    One embedding is stored per (teacher, tc) configuration and shared across
    all AE seeds — the seed only affects autoencoder weight initialisation.
    """
    path = teacher_embedding_path(output_dir, dataset_name, teacher, tc)
    if path.exists():
        if verbose:
            print(f"Skipping (exists): {path.name}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    n_components = tc.get("n_components", 2)

    # Build kwargs for get_teacher_embeddings (without n_components key)
    kw = {k: v for k, v in tc.items() if k != "n_components"}

    Z = get_teacher_embeddings(teacher, X_train, n_components=n_components, **kw)

    try:
        with open(path, "xb") as f:
            np.save(f, Z)
    except FileExistsError:
        # Another worker precomputed this embedding concurrently — skip.
        return

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
    # For rank sweeps, n_components varies per tc — let it override latent_dim.
    latent_dim  = tc.get("n_components", config.get("latent_dim", 2))
    dataset     = config.get("dataset_name", "data")
    norm        = config.get("normalize_teacher", True)

    X_train = np.load(output_dir / "X_train.npy")

    emb_path = teacher_embedding_path(output_dir, dataset, teacher, tc)
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

    import torch
    torch.manual_seed(seed)

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
        save_dir=str(output_dir) if config.get("save_checkpoints", True) else None,
        prefix=prefix,
    )

    # Report final summary metrics so they appear in analysis.best_result /
    # trial.last_result and can be collected into a sweep summary CSV.
    try:
        from ray import tune as _tune
        _tune.report({
            "distill_loss":       student.final_distill_loss_,
            "recon_loss":         student.final_recon_loss_,
            "n_epochs":           student.n_epochs_trained_,
            "early_stopped":      student.n_epochs_trained_ < config.get("max_epochs", 5000),
        })
    except RuntimeError:
        pass


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


def _save_sweep_summary(
    analysis,
    output_dir: Path,
    param_name: str,
    teacher: str,
    mode: str,
    verbose: bool = True,
):
    """Extract per-trial final metrics from Ray Tune analysis and write a CSV.

    The file is named ``sweep_summary_{teacher}_{mode}.csv`` so that summaries
    from different teachers (umap, tsne) and sweep modes (teacher_sweep,
    rank_sweep, etc.) do not overwrite one another.

    Columns: param_name, seed, final_distill_loss, final_recon_loss,
             n_epochs, early_stopped.
    """
    rows = []
    for trial in analysis.trials:
        cfg  = trial.config
        last = trial.last_result or {}
        tc   = cfg.get("teacher_config", {})
        rows.append({
            param_name:            tc.get(param_name, None),
            "seed":                cfg.get("seed", None),
            "final_distill_loss":  last.get("distill_loss", float("nan")),
            "final_recon_loss":    last.get("recon_loss",   float("nan")),
            "n_epochs":            last.get("n_epochs",     None),
            "early_stopped":       last.get("early_stopped", None),
        })

    summary_df = pd.DataFrame(rows).sort_values([param_name, "seed"]).reset_index(drop=True)
    summary_path = output_dir / f"sweep_summary_{teacher}_{mode}.csv"
    summary_df.to_csv(summary_path, index=False)
    if verbose:
        print(f"\nSweep summary saved to {summary_path}")
        print(summary_df.to_string(index=False))
    return summary_df


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
