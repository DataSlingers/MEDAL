# run teacher evaluation as a grid search on ray
from ray import tune
from medal.eval_utils import load_and_split, get_teacher_embeddings
from sklearn.model_selection import train_test_split
import os
import numpy as np
from pathlib import Path
import torch.nn as nn
from medal.core import MEDAL, GlobalEmbeddingNormalizer
from medal.dictionaries import TEACHER_SWEEP_SPECS, INIT_CONFIG, RANK_SWEEP_SPECS, PATH_PREFIX, TEACHER_COMPARISON
import itertools, time
import pickle

def get_dataset_config(dataset_name, model_variant="medal", **update_kws):
    """
    Returns the base config and applies variant-specific overrides.
    Variants: 'medal' (default), 'vanillaAE', 'linearAE'
    """
    config = INIT_CONFIG[dataset_name].copy()
    
    if model_variant == "vanillaAE":
        config["lambda_d"] = 0
        # Add other vanilla-specific defaults if they aren't in INIT_CONFIG
#         if dataset_name == "mnist":
#             config["lr"] = 1e-4
#             config["t_factor"] = 0.95
        
    elif model_variant == "linearAE":
        config["activation"] = None
#         config["use_batchnorm"] = False
        config["dropout_rate"] = 0.0
        
    config["model_variant"] = model_variant
    config["dataset_name"] = dataset_name
    if update_kws:
        config.update(**update_kws)
    print(config)
    return config

def build_teacher_grid(dataset_name, teachers_to_run, seeds=range(1), mode = "teacher_sweep"):
    grid = []
    if mode == "teacher_sweep":
        spec = TEACHER_SWEEP_SPECS.get(dataset_name, {})
    elif mode == "rank_sweep":
        spec = RANK_SWEEP_SPECS.get(dataset_name, {})
    elif mode == "teacher_comparison":
        spec = TEACHER_COMPARISON.get(dataset_name, {})
        
    elif mode == "inter":
        spec = {
                    "umap": {
                        "t_n_neighbors": [18], 
                        "min_dist": [0.1], 
                        "n_components": [2]},
                    "tsne": {
                        "perplexity": [18], 
                        "n_components": [2]},
                }
    
    for teacher_name in teachers_to_run:
        if teacher_name not in spec:
            print(f"Warning: No spec found for {teacher_name} on {dataset_name}. Skipping.")
            continue
            
        params = spec[teacher_name]
        # Get all combinations of parameters for this teacher
        keys, values = zip(*params.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        for seed in seeds:
            for combo in param_combinations:
                # Standardize keys to match your existing precompute/train logic
                config_item = {
                    "teacher": teacher_name,
                    "teacher_seed": seed,
                }
                
                config_item.update({k: v for k, v in combo.items()})
                grid.append(config_item)
                
    return grid

def precompute_teacher_embeddings(tc, config, load_this_seed=None):
    X_tr, X_te = load_and_split(config['dataset_name'], seed=0, test_size=config.get("test_size", 1))
    X_tr, X_te = train_test_split(X_tr, random_state=0, test_size=config.get("test_size", 1))
    print(f"X_tr shape: {X_tr.shape}")

    seed = load_this_seed or 0
    teacher = tc['teacher']
    dataset = config['dataset_name']
    n_components = tc.get('n_components', 2)

    # Build the path suffix and embedding kwargs per teacher
    TEACHER_CONFIGS = {
        "umap": {
            "path_parts": lambda: (
                f"{teacher}_{tc['t_n_neighbors']}_{tc['min_dist']}"
                if n_components == 2
                else f"{teacher}{n_components}_{tc['t_n_neighbors']}_{tc['min_dist']}"
            ),
            "kwargs": {"n_neighbors": tc.get("t_n_neighbors"), "min_dist": tc.get("min_dist"), "random_state": seed},
        },
        "pca": {
            "path_parts": lambda: f"{teacher}{n_components}",
            "kwargs": {"random_state": seed},
        },
        "isomap": {
            "path_parts": lambda: f"{teacher}_{tc['t_n_neighbors']}",
            "kwargs": {"n_neighbors": tc.get("t_n_neighbors")},
        },
        "tsne": {
            "path_parts": lambda: (
                f"{teacher}_{tc['perplexity']}"
                if n_components == 2
                else f"{teacher}{n_components}_{tc['perplexity']}"
            ),
            "kwargs": {"perplexity": tc.get("perplexity"), "learning_rate": tc.get("learning_rate", "auto"), "random_state": seed},
        },
        "spectral": {
            "path_parts": lambda: f"{teacher}{n_components}_{tc['t_n_neighbors']}",
            "kwargs": {"n_neighbors": tc.get("t_n_neighbors"), "random_state": seed},
        },
        "phate": {
            "path_parts": lambda: f"{teacher}{n_components}_{tc['t_n_neighbors']}",
            "kwargs": {"knn": tc.get("t_n_neighbors"), "random_state": seed},
        },
    }

    if teacher not in TEACHER_CONFIGS:
        raise ValueError(f"Unknown teacher: {teacher!r}")

    cfg = TEACHER_CONFIGS[teacher]
    path_suffix = cfg["path_parts"]()
    model_path = Path(PATH_PREFIX) / f"embeddings2/{dataset}_{path_suffix}_{seed}_train.npy"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_teacher_model = config.get("save_teacher_model", False)
    extra_kwargs = {
        "save_teacher_model": save_teacher_model,
        "save_teacher_path": model_path.with_suffix(".pkl") if save_teacher_model else None,
    }

    try:
        with open(model_path, "xb") as f:
            Z_tr = get_teacher_embeddings(teacher, X_tr, n_components=n_components, **cfg["kwargs"], **extra_kwargs)
            np.save(f, Z_tr)
        print(f"Saved: {model_path}")
    except FileExistsError:
        print(f"Skipped (already exists): {model_path}")


def compare_teacher(config):
    """
    Trainable function for PBT optimization of DRD model.
    Reports distill_loss at regular intervals for PBT to use.
    """
    # Load and prepare data, same train-test split
    tc = config["teacher_config"]
    X_tr, X_te = load_and_split(config['dataset_name'], seed=0, test_size=config["test_size"] if "test_size" in config else 1)
    X_tr, X_te = train_test_split(X_tr, random_state=0, test_size=config["test_size"] if "test_size" in config else 1)
    
    print(f"Size of train set: {X_tr.shape[0]}, size of validation set {X_te.shape[0]}")

    load_this_seed = config["load_this_seed"] if config["load_this_seed"] is not None else 0
    
    if tc['teacher'] == "umap":
        if not ('n_components' in tc) or ('n_components' in tc and tc['n_components'] == 2):
            model_path = Path(PATH_PREFIX) / f"embeddings2/{config['dataset_name']}_{tc['teacher']}_{tc['t_n_neighbors']}_{tc['min_dist']}_{load_this_seed}_train.npy"
        else:
            model_path = Path(PATH_PREFIX) / f"embeddings2/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{tc['t_n_neighbors']}_{tc['min_dist']}_{load_this_seed}_train.npy"
    
    elif tc['teacher'] == "pca":
        model_path = Path(PATH_PREFIX) / f"embeddings2/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{load_this_seed}_train.npy"
    
    elif tc['teacher'] == "isomap":
        model_path = Path(PATH_PREFIX) / f"embeddings2/{config['dataset_name']}_{tc['teacher']}_{tc['t_n_neighbors']}_{load_this_seed}_train.npy"
    
    
    elif tc['teacher'] == "tsne":
        if not ('n_components' in tc) or ('n_components' in tc and tc['n_components'] == 2):
            model_path = Path(PATH_PREFIX) / f"embeddings2/{config['dataset_name']}_{tc['teacher']}_{tc['perplexity']}_{load_this_seed}_train.npy"
        else:
            model_path = Path(PATH_PREFIX) / f"embeddings2/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{tc['perplexity']}_{load_this_seed}_train.npy"
    
    
    elif tc['teacher'] == "spectral":
        model_path = Path(PATH_PREFIX) / f"embeddings2/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{tc['t_n_neighbors']}_{load_this_seed}_train.npy"
    elif tc['teacher'] == "phate":
        model_path = Path(PATH_PREFIX) / f"embeddings2/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{tc['t_n_neighbors']}_{load_this_seed}_train.npy"

    print("loading.... ", model_path)
    Z_tr = np.load(model_path)
    
    if config.get("normalize_teacher_embedding", True):
        teacher_normalizer = GlobalEmbeddingNormalizer().fit(Z_tr)
        Z_tr = teacher_normalizer.transform(Z_tr)
        norm_path = model_path.with_suffix(".norm.pkl")
        with open(norm_path, "wb") as f:
            pickle.dump(
                {"mean": teacher_normalizer.mean_, "scale": teacher_normalizer.scale_},
                f
            )

    # Prepare student configuration
    student_kwargs = {
        "hidden_dims": config["hidden_dims"],
        "activation": config['activation'],
        "bottleneck_activation": config["bottleneck_activation"],
        "final_activation": config["final_activation"] if "final_activation" in config else None,
        "lambda_d": config['lambda_d'],
        "lr": config['lr'], 
        "epochs": config['max_epochs'], 
        "batch_size": config['batch_size'], 
        "device": config["device"] if "device" in config else "cuda",
        "clip_grad_norm": config['clip_grad_norm'] if "clip_grad_norm" in config else 1.0,
        "warmup": config["warmup"] if "warmup" in config else 0,
        "eta_min": config["eta_min"] if "eta_min" in config else 0,
        "use_batchnorm": config['use_batchnorm'] if "use_batchnorm" in config else False,
        "dropout_rate": config['dropout_rate'] if "dropout_rate" in config else 0.1,
        "adamw_weight_decay": config['adamw_weight_decay'] if "adamw_weight_decay" in config else 1e-5,
        "factor": config['t_factor'] if "t_factor" in config else 0.9,
        "patience": config["t_patience"] if "t_patience" in config else 20,
        "criterion": config['criterion'] if "criterion" in config else nn.MSELoss, 
    }
    
    # Create student model
    input_dim = X_tr.shape[1]
    if input_dim is None or student_kwargs.get("hidden_dims", None) is None:
        raise ValueError("For MEDAL, input_dim and hidden_dims must be specified.")

    student_config = dict(
        input_dim=input_dim,
        latent_dim=tc["n_components"] if "n_components" in tc else 2,
        **student_kwargs
    )

    student = MEDAL(**student_config)
    
    teacher = tc['teacher']

    if teacher == "umap":
        prefix = f'{teacher}{tc["n_components"]}_{tc["t_n_neighbors"]}_{tc["min_dist"]}_tc{config["load_this_seed"]}_{tc["teacher_seed"]}'
    elif teacher == "tsne":
        prefix = f'{teacher}{tc["n_components"]}_{tc["perplexity"]}_tc{config["load_this_seed"]}_{tc["teacher_seed"]}'
    elif teacher == "pca":
        prefix = f'{teacher}{tc["n_components"]}_tc{config["load_this_seed"]}_{tc["teacher_seed"]}'
    elif teacher == "spectral":
        prefix = f'{teacher}{tc["n_components"]}_{tc["t_n_neighbors"]}_tc{config["load_this_seed"]}_{tc["teacher_seed"]}'
    elif teacher == "phate":
        prefix = f'{teacher}{tc["n_components"]}_{tc["t_n_neighbors"]}_tc{config["load_this_seed"]}_{tc["teacher_seed"]}'

    # accounting for vanillaAE and linearAE
    if config["model_variant"] == "vanillaAE":
        prefix = f'vanillaAE_{tc["n_components"]}_tc{config["load_this_seed"]}_{tc["teacher_seed"]}'
    else:
        prefix = f'{config["model_variant"]}_{prefix}'
    
    prefix = f'{config["dataset_name"]}/{prefix}'

    distill_bands = config["distill_bands"] if "distill_bands" in config else [(1e-12, 9e-6)]
    student.fit(X_tr, Z_tr, config['verbose'], 
                phase="finetune", pretrained_path=config['pretrained_path'],
                target_bands=distill_bands, 
                stability_window=20, 
                epsilon_distill=1e-7, 
                epsilon_recon=1e-3, # 1e-2 for macaque, others 1e-3 
                patience=50, # unit = epoch
                return_on_stable=True,
                # checkpointing
                save_dir = config["save_dir"], 
                prefix = prefix,
                )


if __name__ == "__main__":
    DEVICE = "cuda"
    dataset_names = ["mnist"]
    variant = "medal"
    teachers = ["phate"]
    load_this_seed = [0]
    
    for dataset_name in dataset_names:
        config = get_dataset_config(dataset_name, variant,
                                    verbose= False,
                                    pretrained_path= None,
                                    retrain_teacher= False,
                                    save_dir= PATH_PREFIX + f'/tmp_results/normalize',
                                   ).copy()

        teacher_grid = build_teacher_grid(dataset_name, teachers, seeds=range(5), mode = "teacher_sweep") # {"rank_sweep", "teacher_sweep", "teacher_comparison"}

        config.update({
            "save_teacher_model": False,
        })

        for tc in teacher_grid:
            for lts in load_this_seed:
                precompute_teacher_embeddings(tc, config, load_this_seed = lts)

        config.update({
            "teacher_config": tune.grid_search(teacher_grid),
            "load_this_seed": tune.grid_search(load_this_seed),
        })

        start = time.time()
        # input: dataset name, teacher_method
        analysis = tune.run(
            compare_teacher,
            name="drd_teacher_sweep",
            num_samples=1, 
            resources_per_trial={"cpu": 4, "gpu": 1},  # Adjusted GPU allocation
            config= config,
            verbose=1,
            max_failures=3,
            storage_path="/tmp/ray_results",
        )
        elapsed = time.time() - start
        print(f"Total experiment time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
