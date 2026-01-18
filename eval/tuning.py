from ray import tune
from ray.tune.schedulers import PopulationBasedTraining, AsyncHyperBandScheduler
from ray.tune import CLIReporter
from src.medal.dictionaries import INIT_CONFIG
from src.medal.compare_teachers import precompute_teacher_embeddings, compare_teacher
import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ray import train
from pathlib import Path
from copy import deepcopy

def architecture_analysis_grid(dataset, mode):
    BASE_INIT_CONFIG = deepcopy(INIT_CONFIG[dataset])
    BASE_INIT_CONFIG.update({
        "model_variant": "medal",
        "dataset_name": dataset,
        "verbose": False,
        "pretrained_path": None,
        "retrain_teacher": False,
        "save_dir": None,
        "distill_bands": None,
        "eta_min1": 1e-11, # let the network distill as much as possible
        "use_batchnorm": False
    })

    if mode == "activation":
        BASE_INIT_CONFIG["activation"] = tune.grid_search(["ReLU", "SELU", None])
        BASE_INIT_CONFIG["bottleneck_activation"] = tune.grid_search(["ReLU", "SELU", None])
    elif mode == "depth":
        if dataset == "mnist":
            BASE_INIT_CONFIG["hidden_dims"] = tune.grid_search([[2378] * 2, [1000] * 4, [867] * 6, [734] * 8])
        elif dataset == "gene_cancer":
            BASE_INIT_CONFIG["hidden_dims"] = tune.grid_search([[5734] * 2, [1000] * 4, [1895] * 6, [1563] * 8])
        elif dataset == "darmanis":
            BASE_INIT_CONFIG["hidden_dims"] = tune.grid_search([[1809] * 2, [1000] * 4, [830] * 6, [682] * 8])
    elif mode == "size":
        BASE_INIT_CONFIG["hidden_dims"] = tune.grid_search([[200] *3, [500] * 3, [1000] * 3, [5000]*3])
    elif mode == "order":
        BASE_INIT_CONFIG["hidden_dims"] = tune.grid_search([
             [2000, 800, 400, 200], 
             [800, 2000, 400, 200],
             [800, 400, 2000, 200],
             [800, 400, 200, 2000]])
    elif mode == "bnorm":
        base_lr = BASE_INIT_CONFIG["lr"]
        BASE_INIT_CONFIG["lr"] = tune.grid_search([base_lr *0.1, base_lr * 0.5,base_lr *  1, base_lr * 5, base_lr *10])
        BASE_INIT_CONFIG["use_batchnorm"] = tune.choice([True, False])
        if dataset == "mnist":
            BASE_INIT_CONFIG["hidden_dims"] = tune.grid_search([[2378] * 2, [1000] * 4, [734] * 8, [477] * 16])
        elif dataset == "gene_cancer":
            BASE_INIT_CONFIG["hidden_dims"] = tune.grid_search([[5734] * 2, [1000] * 4, [1563] * 8, [743] * 16])
        elif dataset == "darmanis":
            BASE_INIT_CONFIG["hidden_dims"] = tune.grid_search([[1809] * 2, [1000] * 4, [682] * 8, [467] * 16])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return BASE_INIT_CONFIG

TEACHER_SPECS = {
    "gene_cancer": {
        "umap": {"t_n_neighbors": 5, "min_dist": 0.1, "n_components": 2},
        "spectral": {"t_n_neighbors": 5, "n_components": 2},
        "tsne": {"perplexity": 5, "learning_rate": "auto","n_components": 2},
	"pca": {"n_components": 2}
    },
    "mnist": {
        "umap": {"t_n_neighbors": 5, "min_dist": 0.1, "n_components": 2},
        "spectral": {"t_n_neighbors": 5, "n_components": 2},
        "tsne": {"perplexity": 5, "learning_rate": "auto","n_components": 2},
	"pca": {"n_components": 2}
    },
    "darmanis": {
	"umap": {"t_n_neighbors": 5, "min_dist": 0.1, "n_components": 2},
	"spectral": {"t_n_neighbors": 5, "n_components": 2},
        "tsne": {"perplexity": 5, "learning_rate": "auto", "n_components": 2},
	"pca": {"n_components": 2}
    }
}



def build_teacher_grid(dataset_name, teacher_name):
    spec = TEACHER_SPECS.get(dataset_name, {})
    
    if teacher_name not in spec:
        print(f"Warning: No spec found for {teacher_name} on {dataset_name}. Skipping.")
            
    params = spec[teacher_name].copy()
    params.update({
        "teacher": teacher_name,
        "teacher_seed": 0,
    })
        
    return params


MODE = "bnorm"
MODE = "size"
DEVICE = "cuda"
dataset_name = ["darmanis"]
teacher_name = "pca" # replace with "umap", "isomap", "spectral", "pca", ...
PATH_PREFIX = "/share/ctn/users/bnc2119/drd_data" #"/shared/share_mala/irchang/drd"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7" 

ahbs = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    metric="distill_loss", 
    mode="min",
    grace_period=1000,
    max_t = 1000,
)

# Run the experiment (simplified approach without RunConfig for compatibility)
for data_name in dataset_name:
    path = f"{PATH_PREFIX}/tune_results/{MODE}_{data_name}_{teacher_name}.csv"

    # activation analysis
    config = architecture_analysis_grid(data_name, mode=MODE)
    
    teacher_config = build_teacher_grid(data_name, teacher_name)
    precompute_teacher_embeddings(teacher_config, config)
    config.update({
        "teacher_config": teacher_config,
    })

    analysis = tune.run(
        compare_teacher,
        name="drd_asynchyperband_distill_optimization",
        num_samples=10, 
        resources_per_trial={"cpu": 4, "gpu": 1},  # Adjusted GPU allocation
        config= config,
        verbose=1,
        max_failures=0,
        scheduler=ahbs,
        storage_path="/tmp/ray_results",
    )

    analysis.results_df.to_csv(path)
    print(f"Save results to {path}")

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


