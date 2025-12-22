from ray import tune
from ray.tune.schedulers import PopulationBasedTraining, AsyncHyperBandScheduler
from ray.tune import CLIReporter
from utils.eval_utils import make_student, load_and_split, get_teacher_embeddings, eval_student
from torch.utils.data import DataLoader, TensorDataset
import torch, torch.nn as nn
import tqdm, os
import torch.nn.functional as F
import numpy as np
from ray import train
import os, tempfile
import scanpy as sc
from pathlib import Path

DEVICE = "cuda"
dataset_name = ["macaque"]
teacher_name = "tsne"
PATH_PREFIX = "/shared/share_mala/irchang/drd"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7" 
# path = f"{PATH_PREFIX}/tune_results/activation_{dataset_name}_{teacher_name}.csv"

INIT_CONFIG = {
    "gene_cancer": {
        "data_name": "gene_cancer",
        "teacher": "spectral",
        "n_components": 100,
        "t_n_neighbors": 11,
        "lr": tune.loguniform(1e-7, 1e-4),
        "lambda_d": 10000,
        "eta_min1": tune.choice([1e-8, 1e-9,1e-10,1e-11]), 
        "eta_min2": 0.0, 
        "hidden_dims": 
        [1000, 1000, 1000, 1000],
        "activation": "SELU",
        # tune.grid_search(["ReLU", "SELU"]),
        "bottleneck_activation": None,
        # tune.grid_search(["ReLU", "SELU", None]),
        "max_epochs": 7000,
        "T_max_ratio": tune.choice([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "warmup": 0,  
        "seed": 0,
        "batch_size": 100,
        "test_size": 0.2,
        "adamw_weight_decay": tune.loguniform(1e-6, 1e-2),
    },
    "wine": {
        "data_name": "wine",
        "teacher": "umap",
        "t_n_neighbors": 15,
        "perplexity": 30,
        "learning_rate": 200,
        "lr": 0.003233466538536306,
        "lambda_d": 7000,
        "eta_min1": 6.076040435352558e-08,
        "eta_min2": 1.2312179372780289e-17,
        "hidden_dims":  
        # [258, 258, 258, 258],
        tune.grid_search([
            [890,177,126,95],
            [177,890,126,95],
            [177,126,890,95],
            [177,126,95,890],
            # depth
            # [200, 200, 200, 200, 200, 200],
            # [443, 443],
            # [258, 258, 258, 258],
            # [169, 169, 169, 169, 169, 169, 169, 169]
            # size
            # [500, 500],
            # [1000, 1000],
            # [5000, 5000],
            # [10000, 10000]
        ]),
        "activation": "SELU",
        # tune.grid_search(["ReLU", "SELU"]),
        "bottleneck_activation": None,
        # tune.grid_search(["ReLU", "SELU", None]),
        'max_epochs': 200000, 
        'T_max_ratio': 0.7,
        "warmup": 0, 
        "seed": tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "batch_size": 100
    },
    "mnist": { # 10000
        "data_name": "mnist",
        "teacher": "pca",
        "n_components": 511,
        "t_n_neighbors": 15,
        "perplexity": 5,
        "learning_rate": 'auto',
        "lr": tune.loguniform(1e-4, 1e-1),
        "lambda_d": 1e8,
        "eta_min1": tune.loguniform(1e-8, 9e-5),
        "eta_min2": 0,
        "hidden_dims": 
        [1000, 1000, 1000, 1000, 1000],
        # tune.grid_search([
        #     [1585, 1194, 789, 599, 393],
        #     [1194, 1585, 789, 599, 393],
        #     [1194, 789, 1585, 599, 393],
        #     [1194, 789, 599, 1585, 393],
        #     [1194, 789, 599, 393, 1585],
            # depth
            # [1000, 1000, 1000, 1000, 1000],
            # [1363, 1363, 1363],
            # [830, 830, 830, 830, 830, 830, 830],
            # [726, 726, 726, 726, 726, 726, 726, 726],
            # size
            # [200, 200, 200],
            # [500, 500, 500],
            # [1000, 1000, 1000],
            # [5000, 5000, 5000],
        # ]),
        "activation": "SELU", 
        # tune.grid_search(["ReLU", "SELU"]),
        "bottleneck_activation": None, 
        # tune.grid_search(["ReLU", "SELU", None]),
        "max_epochs": 20000,
        "T_max_ratio":1,
        "batch_size": tune.choice([256, 512, 1024]),
        "warmup": 0, 
        "seed": 0,
        "test_size":0.2
    },
    "diabetes": { 
        "teacher": "umap",
        "t_n_neighbors": 15,
        "n_components": 3,
        # "student_latent_dim": 3, 
        "lr": tune.loguniform(1e-2, 1), 
        "lambda_d": tune.choice([3000, 5000, 7000, 10000, 15000]),
        "eta_min1": tune.loguniform(1e-3, 1e-2), 
        "eta_min2": tune.loguniform(1e-6, 1e-3),
        "hidden_dims": tune.choice([[500, 500, 500, 500, 500, 500, 500]]),
        # tune.choice([
        #     [441, 220, 110, 5000],
        #     [5000, 441, 220, 110],
        #     [441, 5000, 220, 110],
        #     [441, 220, 5000, 110],
        # ]),
        "activation": "SELU",
        "bottleneck_activation": None,
        "max_epochs": tune.choice([100000, 120000, 130000, 150000, 200000]),
        "T_max_ratio": tune.choice([0.6, 0.7, 0.8, 0.9]),
        "batch_size": 512,
        "seed": 0, 
        "warmup": 0, 
    },
    "single_cell": {
        "data_name": "single_cell",
        "teacher": "umap",
        "t_n_neighbors": 5,
        "min_dist": 0.05,
        "perplexity": 1,
        "learning_rate": 'auto',
        "lr": tune.uniform(1e-4, 5e-3),
        "lambda_d": 30000,
        "eta_min1": tune.uniform(1e-6, 1e-4),
        "eta_min2": tune.uniform(1e-10, 1e-6),
        "hidden_dims": 
        [294, 294, 294, 294, 294, 294, 294, 294, 294],
        # tune.choice([
            # [250, 250, 250, 2917, 250, 250, 250],
            # [2917, 250, 250, 250, 250, 250, 250],
            # [250, 250, 250, 250, 250, 250, 2917],
            # [250, 250, 2917, 250, 250, 250, 250],
            # [250, 250, 250, 250, 2917, 250, 250],
            # [300,300,300,300,300,300,300],
            # [307,307,307,307,307],
            # [315, 315, 315],
            # [294, 294, 294, 294, 294, 294, 294, 294, 294],
            # [400, 400, 400, 400, 400, 400, 400, 400, 400],
            # size
            # [200, 200, 200],
            # [500, 500, 500],
            # [1000, 1000, 1000],
            # [5000, 5000, 5000],
        # ]),
        "activation": "SELU", 
        # tune.grid_search(["ReLU", "SELU"]),
        "bottleneck_activation": None,
        # tune.grid_search(["ReLU", "SELU", None]),
        'max_epochs': 30000,
        'T_max_ratio': tune.choice([0.6, 0.7, 0.8, 0.9]),
        "warmup": tune.choice([0, 100, 500, 1000, 2000, 3000]), 
        "seed": 0,
        "batch_size": 10000,
        "use_lbfgs": False
    },
    "hydra":{
        "data_name": "hydra",
        "teacher": "umap",
        "t_n_neighbors": 10,
        "min_dist": 0.1,
        "perplexity": 2,
        "learning_rate": 'auto',
        "lr": tune.uniform(1e-4, 1e-2),
        "lambda_d": tune.choice([500, 3000, 10000, 30000]),
        "eta_min1": tune.uniform(1e-6, 1e-4),
        "eta_min2": tune.uniform(1e-10, 1e-6),
        "hidden_dims": 
        tune.choice([
            # [5120, 2560, 1280, 640, 320]
            [309, 1792, 1792, 1792],
            # [588,2751,2751,2751],
            # [300, 300, 300, 300, 300, 300, 300, 300, 300],
            # [2000, 2000, 2000, 2000, 2000],
            # [2000, 2000, 2000, 2000, 2000, 2000],
        ]),
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': tune.choice([7000, 10000]),
        'T_max_ratio': 0.7,
        "warmup": tune.choice([0, 100, 500, 1000, 2000, 3000]), 
        "seed": 0,
        "batch_size": 50000,
        "test_size": 0.2
    },
    "pbmc":{
        "data_name": "pbmc",
        "teacher": "tsne",
        # "t_n_neighbors": tune.choice([5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 80, 160]),
        # "min_dist": tune.choice([0.0125, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]),
        "perplexity": 5, # tune.grid_search([5, 10, 15, 20, 45, 55, 75, 140]),
        "learning_rate": 'auto',
        "lr": tune.uniform(1e-4, 5e-3),
        "lambda_d": tune.choice([500, 3000, 10000, 30000, 50000]),
        "eta_min1": tune.uniform(1e-6, 1e-4),
        "eta_min2": tune.uniform(1e-10, 1e-6),
        "hidden_dims": 
        # [500, 500, 500, 500, 500, 500, 500],
        tune.choice([
            # [1024, 2048, 1024]
        #     [500, 500, 500, 500],
        #     [1000, 1000, 1000, 1000],
            [500, 500, 500, 500, 500],
        #     [500, 500, 500, 500, 500, 500],
        ]),
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 20000,
        'T_max_ratio': tune.choice([0.5, 0.6, 0.7, 0.8, 0.9]),
        "warmup": tune.choice([0, 100, 500, 1000, 2000, 3000]), 
        "seed": 0,
        "batch_size": 10000,
        "use_lbfgs": False
    },
    "astro":{
        "data_name": "astro",
        "teacher": "tsne",
        "perplexity": 499,
        "learning_rate": 'auto',
        "lr": tune.loguniform(1e-4, 5e-2),
        "lambda_d": tune.choice([10000, 1000, 100]),
        "eta_min1": 1e-7,
        "lr_restart": None,
        "eta_min2": 0,
        "hidden_dims": [700] * 15,
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 60000,
        'T_max_ratio': 1,
        "patience": 70,
        "factor": tune.uniform(0.85, 1),
        "warmup": 0,
        "test_size": 0.2, 
        "seed": 0,
        "batch_size": 25600,
    },
    "cortical":{
        "data_name": "cortical",
        "teacher": "spectral",
        "t_n_neighbors": 5,
        "perplexity": 30,
        "learning_rate": 'auto',
        "lr": tune.uniform(1e-6, 1e-3),
        "lambda_d": tune.choice([500, 3000, 10000, 30000, 50000]),
        "eta_min1": tune.loguniform(1e-8, 1e-6),
        "eta_min2": tune.loguniform(1e-10, 1e-8),
        "hidden_dims": [309, 1792, 1792, 1792],
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 15000,
        'T_max_ratio': tune.choice([0.7,0.8]),
        "warmup": 0,
        "seed": 0,
        "batch_size": 50000,
        "test_size":0.2,
    },
    "macaque":{   
        "data_name": "macaque",
        "teacher": "tsne",
        "perplexity": 499,
        "t_n_neighbors": 5,
        "min_dist": 0.1,
        "learning_rate": 'auto',  
        "lr":tune.loguniform(1e-4, 5e-2),
        "lambda_d": 10000, # 50000, 200000
        "eta_min1": 1e-7,
        "eta_min2": 0,
        "lr_restart": None,
        "hidden_dims": [1000] * 15,
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 50000,
        'T_max_ratio': 1,
        "patience": 70,
        "factor": tune.uniform(0.88, 1),
        "warmup": 0, 
        "batch_size": 1024,
        "test_size":0.2,
        "seed": 0,

    },
    "synthetic":{
        "data_name": "synthetic",
        "teacher": "umap",
        "perplexity": 30,
        "t_n_neighbors": 10,
        "min_dist": 0.1,
        "learning_rate": 'auto',
        "lr": tune.loguniform(1e-5, 5e-3),
        "lambda_d": 100000,
        "eta_min1": tune.loguniform(5e-7, 1e-5), # 6.24882e-06, #tsne
        "eta_min2": tune.loguniform(5e-9, 5e-7),# 5e-7, # tsne
        # "lr": 0.000359075,
        # "lambda_d": 50000,
        # "eta_min1": 2.83639e-06,
        # "eta_min2": 1.97165e-08,
        "hidden_dims":[700] * 15,
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 10000,
        'T_max_ratio': tune.choice([0.7, 0.8]), # 0.7,
        "warmup": 0, 
        "seed": 0,
        "batch_size": 1024,
        "test_size": 0.2,
        "clip_grad_norm":1.0,
        "adamw_weight_decay": tune.loguniform(1e-6, 1e-3)
    },
}

def precompute_teacher_embeddings(config):
    X_tr, X_te = load_and_split(config['data_name'], seed=0, test_size=config["test_size"] if "test_size" in config else 1)
    try:
        # 'xb' = create file, fail if it already exists
        if config['teacher'] == "umap":
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['data_name']}_{config['teacher']}_{config['t_n_neighbors']}_{config['min_dist']}_{config['seed']}_train.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    config["teacher"], X_tr,
                    n_components= config["n_components"] if "n_components" in config else 2,
                    n_neighbors=config["t_n_neighbors"], 
                    min_dist=config["min_dist"],
                    random_state=config['seed'],
                )
                np.save(f, Z_tr) 
            
        elif config['teacher'] == "pca":
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['data_name']}_{config['teacher']}{config['n_components']}_{config['seed']}_train.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    config["teacher"], X_tr, 
                    n_components= config["n_components"] if "n_components" in config else 2,
                    random_state=config['seed'],
                )
                np.save(f, Z_tr) 

        elif config['teacher'] == "isomap":
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['data_name']}_{config['teacher']}_{config['t_n_neighbors']}_{config['seed']}_train.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    config["teacher"], X_tr, 
                    n_components= config["n_components"] if "n_components" in config else 2,
                    n_neighbors=config["t_n_neighbors"],
                )
                np.save(f, Z_tr) 

        elif config['teacher'] == "tsne":
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['data_name']}_{config['teacher']}_{config['perplexity']}_{config['seed']}_train.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    config["teacher"], X_tr, 
                    n_components= config["n_components"] if "n_components" in config else 2,
                    perplexity=config["perplexity"],
                    learning_rate=config["learning_rate"],
                    random_state=config['seed'],
                )
                np.save(f, Z_tr) 

        elif config['teacher'] == "spectral":
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['data_name']}_{config['teacher']}_{config['t_n_neighbors']}_{config['seed']}_train.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    config["teacher"], X_tr,
                    n_components = config["n_components"] if "n_components" in config else 2,
                    n_neighbors = config["t_n_neighbors"],
                    random_state = config['seed'],
                )
                np.save(f, Z_tr) 

        print(f"Saved: {model_path}")
    except FileExistsError:
        print(f"Skipped (already exists): {model_path}")


def drd_trainable(config):
    """
    Trainable function for PBT optimization of DRD model.
    Reports distill_loss at regular intervals for PBT to use.
    """
    # Load and prepare data
    X_tr, X_te = load_and_split(config['data_name'], seed=0, test_size=config["test_size"] if "test_size" in config else 1) 

    if config['teacher'] == "umap":
        model_path = Path(PATH_PREFIX) / f"embeddings/{config['data_name']}_{config['teacher']}_{config['t_n_neighbors']}_{config['min_dist']}_{config['seed']}_train.npy"
    elif config['teacher'] == "pca":
        model_path = Path(PATH_PREFIX) / f"embeddings/{config['data_name']}_{config['teacher']}{config['n_components']}_{config['seed']}_train.npy"
    elif config['teacher'] == "isomap":
        model_path = Path(PATH_PREFIX) / f"embeddings/{config['data_name']}_{config['teacher']}_{config['t_n_neighbors']}_{config['seed']}_train.npy"
    elif config['teacher'] == "tsne":
        model_path = Path(PATH_PREFIX) / f"embeddings/{config['data_name']}_{config['teacher']}_{config['perplexity']}_{config['seed']}_train.npy"
    elif config['teacher'] == "phate":
        model_path = Path(PATH_PREFIX) / f"embeddings/{config['data_name']}_{config['teacher']}_{config['t_n_neighbors']}_{config['seed']}_train.npy"
    elif config['teacher'] == "spectral":
        if not ('n_components' in config) or ('n_components' in config and config['n_components'] == 2):
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['data_name']}_{config['teacher']}_{config['t_n_neighbors']}_{config['seed']}_train.npy"
        else:  
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['data_name']}_{config['teacher']}{config['n_components']}_{config['t_n_neighbors']}_{config['seed']}_train.npy"
        
    
    Z_tr = np.load(model_path)
    if np.isnan(Z_tr).any() or np.isinf(Z_tr).any():
        raise ValueError(f"Teacher embeddings at {model_path} contain NaNs or Infs!")
    
    # Prepare student configuration
    student_kwargs = {
        "epochs": config['max_epochs'], 
        "batch_size": config['batch_size'], 
        "lambda_reg": 0.0, 
        "warmup": config.get('warmup', 0),  # Use get for optional params
        "lr": config['lr'], 
        "eta_min1": config["eta_min1"], 
        "eta_min2": config['eta_min2'],
        "lr_restart": config['lr_restart'],
        "T_max": int(config['max_epochs'] * config["T_max_ratio"]), 
        "lambda_d": config['lambda_d'],
        "activation": config['activation'],
        "bottleneck_activation": config["bottleneck_activation"],
        "latent_dim": config["n_components"] if "n_components" in config else 2,
        "clip_grad_norm": config["clip_grad_norm"] if "clip_grad_norm" in config else 1.0,
        "adamw_weight_decay": config["adamw_weight_decay"] if "adamw_weight_decay" in config else 1e-5,
        "factor": config['factor'],
        "patience": config["patience"]
    }
    

    # Create student model
    student = make_student(
        method="drd",
        input_dim=X_tr.shape[1],
        hidden_dims=config['hidden_dims'],
        device=DEVICE,
        **student_kwargs,
    )

    student.fit(X_tr, Z_tr, verbose=False,
                target_bands=[(1e-12, 9e-6)], 
                stability_window=20, 
                epsilon_distill=1e-7, epsilon_recon=1e-3, 
                patience=100, # unit = epoch
                return_on_stable=True,
                )

ahbs = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    metric="distill_loss", 
    mode="min",
    grace_period=500,
    max_t = 1000,
)

# Run the experiment (simplified approach without RunConfig for compatibility)
for data_name in dataset_name:
    path = f"{PATH_PREFIX}/tune_results/{data_name}_{teacher_name}.csv"
    precompute_teacher_embeddings(INIT_CONFIG[data_name].copy())
    analysis = tune.run(
        drd_trainable,
        name="drd_asynchyperband_distill_optimization",
        num_samples=10, 
        resources_per_trial={"cpu": 4, "gpu": 1},  # Adjusted GPU allocation
        config= INIT_CONFIG[data_name],
        verbose=1,
        max_failures=3,
        scheduler=ahbs,
        storage_path="/tmp/ray_results"
    )

    # analysis.results_df.to_csv(path)
    # print(f"Save results to {path}")
    print("="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    best_config = analysis.get_best_config("distill_loss", "min")
    print(f"Best config: {best_config}")

    best_trial = analysis.get_best_trial("distill_loss", "min")
    print(f"Best distill_loss achieved: {best_trial.last_result['distill_loss']}")

    # Print top 5 configurations
    print("\nTop 5 configurations:")
    df = analysis.results_df.nsmallest(5, 'distill_loss')
    for i, (idx, row) in enumerate(df.iterrows()):
        print(f"{i+1}. Distill Loss: {row['distill_loss']}")
        print(f"   Config: lr={row['config/lr']:.2e}, lambda_d={row['config/lambda_d']}, "
            f"hidden_dims={row['config/hidden_dims']}")
        print(f"   Architecture depth: {len(row['config/hidden_dims'])}")
        print()


