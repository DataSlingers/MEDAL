"""
AE architecture search via Ray Tune.

Typical usage::

    from medal.tuning import tune_architecture, get_best_config

    analysis = tune_architecture(
        X_train,
        output_dir="experiments/arch_search",
        teacher="tsne",
        teacher_params={"perplexity": 30},
    )
    arch_config = get_best_config(analysis)
"""
from __future__ import annotations

import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import Optional

from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

from medal.teacher import get_teacher_embeddings
from medal.normalizer import GlobalEmbeddingNormalizer
from medal.sweep import _sweep_trainable


# ------------------------------------------------------------------
# Default search spaces
# ------------------------------------------------------------------

def _default_search_space(input_dim: int) -> dict:
    """
    Sensible default architecture search space for a dataset with the
    given input dimensionality.
    """
    # Hidden layer width: ~sqrt(input_dim) to input_dim, in three depth variants
    w = max(64, int(input_dim ** 0.5))
    return {
        "hidden_dims": tune.grid_search([
            [w * 2] * 2,
            [w * 2] * 4,
            [w * 4] * 4,
            [w * 4] * 6,
        ]),
        "activation":           tune.grid_search(["SELU", "ReLU"]),
        "bottleneck_activation": None,
        "use_batchnorm":        tune.choice([True, False]),
        "dropout_rate":         tune.choice([0.0, 0.1]),
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def tune_architecture(
    X: np.ndarray,
    output_dir: str | Path,
    teacher: str,
    teacher_params: dict,
    latent_dim: int = 2,
    search_space: Optional[dict] = None,
    val_size: float = 0.2,
    num_samples: int = 1,
    resources_per_trial: Optional[dict] = None,
    scheduler: str = "ahb",
    base_config: Optional[dict] = None,
    verbose: bool = True,
):
    """
    Search for the best autoencoder architecture using Ray Tune.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training data.
    output_dir : str or Path
        Embeddings and Ray Tune results are written here.
    teacher : str
        Teacher algorithm (``"tsne"``, ``"umap"``, etc.).
    teacher_params : dict
        A single set of teacher hyperparameters, e.g.
        ``{"perplexity": 30, "n_components": 2}``.
    latent_dim : int
        Target embedding dimensionality.
    search_space : dict, optional
        Ray Tune config for architecture hyperparameters.  Merged on top of
        (and overrides) the default search space.
    val_size : float
        Fraction of *X* to hold out for validation.
    num_samples : int
        Number of Ray Tune samples per configuration (``tune.run`` argument).
    resources_per_trial : dict, optional
        E.g. ``{"cpu": 4, "gpu": 1}``.
    scheduler : str
        ``"ahb"`` (AsyncHyperBand, default) or ``None`` for no scheduler.
    base_config : dict, optional
        Base MEDAL training config (learning rate, epochs, etc.).  Defaults
        are filled in if not provided.
    verbose : bool

    Returns
    -------
    analysis : ray.tune.ExperimentAnalysis
        Pass to :func:`get_best_config` to extract the winning architecture.
    """
    from sklearn.model_selection import train_test_split

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val = train_test_split(X, test_size=val_size, random_state=0)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy", X_val)

    # Precompute teacher embedding for the single provided teacher_params
    n_components = teacher_params.get("n_components", latent_dim)
    kw = {k: v for k, v in teacher_params.items() if k != "n_components"}
    Z_raw = get_teacher_embeddings(teacher, X_train, n_components=n_components, **kw)
    normalizer = GlobalEmbeddingNormalizer().fit(Z_raw)
    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    emb_path = emb_dir / "arch_search_teacher.npy"
    norm_path = emb_dir / "arch_search_teacher.norm.pkl"
    np.save(emb_path, Z_raw)
    normalizer.save(norm_path)

    # Build config
    defaults = {
        "lambda_d":          10,
        "lr":                1e-3,
        "max_epochs":        3000,
        "batch_size":        256,
        "warmup":            0,
        "eta_min":           1e-7,
        "adamw_weight_decay":1e-5,
        "t_factor":          0.9,
        "t_patience":        20,
        "distill_bands":     None,
        "verbose":           False,
    }
    if base_config:
        defaults.update(base_config)

    space = _default_search_space(X_train.shape[1])
    if search_space:
        space.update(search_space)

    config = {
        **defaults,
        **space,
        # Internal keys consumed by the trainable
        "_arch_search":   True,
        "output_dir":     str(output_dir),
        "dataset_name":   "data",
        "teacher":        teacher,
        "latent_dim":     latent_dim,
        "teacher_config": {**teacher_params, "n_components": n_components},
        "seed":           0,
        "normalize_teacher": True,
        "_emb_path":      str(emb_path),
        "_norm_path":     str(norm_path),
    }

    sched = None
    if scheduler == "ahb":
        sched = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="distill_loss",
            mode="min",
            grace_period=500,
            max_t=3000,
        )

    analysis = tune.run(
        _arch_search_trainable,
        name="medal_arch_search",
        num_samples=num_samples,
        resources_per_trial=resources_per_trial or {"cpu": 4, "gpu": 1},
        config=config,
        verbose=1 if verbose else 0,
        max_failures=3,
        scheduler=sched,
        storage_path=str(output_dir / "ray_results"),
    )

    if verbose:
        _print_leaderboard(analysis, top_k=5)

    return analysis


def get_best_config(
    analysis,
    metric: str = "distill_loss",
    mode: str = "min",
    top_k: int = 5,
) -> dict:
    """
    Extract the best architecture config from a :func:`tune_architecture` result.

    Parameters
    ----------
    analysis : ray.tune.ExperimentAnalysis
    metric : str
        Metric to optimise (default ``"distill_loss"``).
    mode : str
        ``"min"`` or ``"max"``.
    top_k : int
        Number of top configs to print.

    Returns
    -------
    dict
        Architecture config ready to pass to :func:`~medal.sweep.run_teacher_sweep`.
    """
    _print_leaderboard(analysis, metric=metric, top_k=top_k)
    best = analysis.get_best_config(metric, mode)
    # Strip internal/Ray-specific keys before returning
    _internal = {"_arch_search", "_emb_path", "_norm_path", "output_dir",
                 "dataset_name", "teacher", "teacher_config", "seed",
                 "normalize_teacher"}
    return {k: v for k, v in best.items() if k not in _internal}


# ------------------------------------------------------------------
# Internal trainable for architecture search
# ------------------------------------------------------------------

def _arch_search_trainable(config):
    """Ray Tune trainable: loads pre-saved teacher embedding directly."""
    import torch.nn as nn

    X_train = np.load(Path(config["output_dir"]) / "X_train.npy")
    Z_raw   = np.load(config["_emb_path"])
    normalizer = GlobalEmbeddingNormalizer.load(config["_norm_path"])
    Z_train = normalizer.transform(Z_raw)

    from medal.model import MEDAL
    student = MEDAL(
        input_dim=X_train.shape[1],
        latent_dim=config.get("latent_dim", 2),
        hidden_dims=config.get("hidden_dims", (128, 128)),
        activation=config.get("activation", "SELU"),
        bottleneck_activation=config.get("bottleneck_activation", None),
        final_activation=config.get("final_activation", None),
        lambda_d=config.get("lambda_d", 10),
        lr=config.get("lr", 1e-3),
        epochs=config.get("max_epochs", 3000),
        batch_size=config.get("batch_size", 256),
        warmup=config.get("warmup", 0),
        eta_min=config.get("eta_min", 1e-7),
        use_batchnorm=config.get("use_batchnorm", False),
        dropout_rate=config.get("dropout_rate", 0.1),
        adamw_weight_decay=config.get("adamw_weight_decay", 1e-5),
        factor=config.get("t_factor", 0.9),
        patience=config.get("t_patience", 20),
        criterion=nn.MSELoss,
    )

    student.fit(
        X_train, Z_train,
        verbose=False,
        target_bands=config.get("distill_bands"),
        stability_window=20,
        epsilon_distill=1e-7,
        epsilon_recon=1e-3,
        patience=50,
        return_on_stable=True,
    )


def _print_leaderboard(analysis, metric: str = "distill_loss", top_k: int = 5):
    try:
        df = analysis.results_df.nsmallest(top_k, metric)
        print(f"\nTop {top_k} configs by {metric}:")
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(
                f"  {i}. {metric}={row[metric]:.4e} | "
                f"hidden_dims={row.get('config/hidden_dims', '?')} | "
                f"lr={row.get('config/lr', '?'):.2e} | "
                f"lambda_d={row.get('config/lambda_d', '?')}"
            )
    except Exception:
        pass
