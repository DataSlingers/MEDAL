# run teacher evaluation as a grid search on ray
from ray import tune
from utils.eval_utils import load_and_split, get_teacher_embeddings
import os
import numpy as np
from pathlib import Path
import torch.nn as nn
from src.medal import MEDAL
from src.medal.dictionaries import TEACHER_SWEEP_SPECS, INIT_CONFIG, RANK_SWEEP_SPECS, PATH_PREFIX
import itertools
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7" 

def get_dataset_config(dataset_name, model_variant="medal", **update_kws):
    """
    Returns the base config and applies variant-specific overrides.
    Variants: 'medal' (default), 'vanillaAE', 'linearAE'
    """
    config = INIT_CONFIG[dataset_name].copy()
    
    if model_variant == "vanillaAE":
        config["lambda_d"] = 0
        # Add other vanilla-specific defaults if they aren't in INIT_CONFIG
        if dataset_name == "mnist":
            config["lr"] = 1e-4
            config["t_factor"] = 0.95
        
    elif model_variant == "linearAE":
        config["activation"] = None
        config["use_batchnorm"] = False
        config["dropout_rate"] = 0.0
        
    config["model_variant"] = model_variant
    config["dataset_name"] = dataset_name
    if update_kws:
        config.update(update_kws)
    return config

def build_teacher_grid(dataset_name, teachers_to_run, seeds=range(1), mode = "teacher_sweep"):
    grid = []
    if mode == "teacher_sweep":
        spec = TEACHER_SWEEP_SPECS.get(dataset_name, {})
    elif mode == "rank_sweep":
        spec = RANK_SWEEP_SPECS.get(dataset_name, {})
    
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

def precompute_teacher_embeddings(tc, config):
    X_tr, X_te = load_and_split(config['dataset_name'], seed=0, test_size=config["test_size"] if "test_size" in config else 1)
    load_this_seed = 0
    try:
        # 'xb' = create file, fail if it already exists
        if tc['teacher'] == "umap":
            if not ('n_components' in tc) or ('n_components' in tc and tc['n_components'] == 2):
                model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['t_n_neighbors']}_{tc['min_dist']}_{load_this_seed}_train.npy"
                model_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{tc['t_n_neighbors']}_{tc['min_dist']}_{load_this_seed}_train.npy"
                model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    tc["teacher"], X_tr,
                    n_components= tc["n_components"] if "n_components" in tc else 2,
                    n_neighbors=tc["t_n_neighbors"], 
                    min_dist=tc["min_dist"],
                    random_state=load_this_seed,
                )
                np.save(f, Z_tr) 
            
        elif tc['teacher'] == "pca":
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{load_this_seed}_train.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    tc["teacher"], X_tr, 
                    n_components= tc["n_components"] if "n_components" in tc else 2,
                    random_state=load_this_seed,
                )
                np.save(f, Z_tr) 

        elif tc['teacher'] == "isomap":
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['t_n_neighbors']}_{load_this_seed}_train.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    tc["teacher"], X_tr, 
                    n_components= tc["n_components"] if "n_components" in tc else 2,
                    n_neighbors=tc["t_n_neighbors"],
                )
                np.save(f, Z_tr) 

        elif tc['teacher'] == "tsne":
            if not ('n_components' in tc) or ('n_components' in tc and tc['n_components'] == 2):
                model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['perplexity']}_{load_this_seed}_train.npy"
            else:
                model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{tc['perplexity']}_{load_this_seed}_train.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    tc["teacher"], X_tr, 
                    n_components= tc["n_components"] if "n_components" in tc else 2,
                    perplexity=tc["perplexity"],
                    learning_rate=tc["learning_rate"],
                    random_state=load_this_seed,
                )
                np.save(f, Z_tr) 

        elif tc['teacher'] == "spectral":
            if not ('n_components' in tc) or ('n_components' in tc and tc['n_components'] == 2):
                model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['t_n_neighbors']}_{load_this_seed}_train.npy"
            else:
                model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{tc['t_n_neighbors']}_{load_this_seed}_train.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True) 
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    tc["teacher"], X_tr,
                    n_components = tc["n_components"] if "n_components" in tc else 2,
                    n_neighbors = tc["t_n_neighbors"],
                    random_state = load_this_seed,
                )
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

    load_this_seed = 0 # teacher seed
    
    if tc['teacher'] == "umap":
        if not ('n_components' in tc) or ('n_components' in tc and tc['n_components'] == 2):
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['t_n_neighbors']}_{tc['min_dist']}_{load_this_seed}_train.npy"
        else:
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{tc['t_n_neighbors']}_{tc['min_dist']}_{load_this_seed}_train.npy"
    elif tc['teacher'] == "pca":
        model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{load_this_seed}_train.npy"
    elif tc['teacher'] == "isomap":
        model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['t_n_neighbors']}_{load_this_seed}_train.npy"
    elif tc['teacher'] == "tsne":
        if not ('n_components' in tc) or ('n_components' in tc and tc['n_components'] == 2):
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['perplexity']}_{load_this_seed}_train.npy"
        else:
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{tc['perplexity']}_{load_this_seed}_train.npy"
    elif tc['teacher'] == "spectral":
        if not ('n_components' in tc) or ('n_components' in tc and tc['n_components'] == 2):
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['t_n_neighbors']}_{load_this_seed}_train.npy"
        else:  
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{tc['t_n_neighbors']}_{load_this_seed}_train.npy"

    Z_tr = np.load(model_path)
    
    # Prepare student configuration
    student_kwargs = {
        "epochs": config['max_epochs'], 
        "batch_size": config['batch_size'], 
        "warmup": config.get('warmup', 0),  # Use get for optional params
        "lr": config['lr'], 
        "eta_min1": config["eta_min1"], 
        "T_max": int(config['max_epochs'] * config["T_max_ratio"]) if "T_max_ratio" in config else config["T_max"], 
        "lambda_d": config['lambda_d'],
        "activation": config['activation'],
        "bottleneck_activation": config["bottleneck_activation"],
        "final_activation": config["final_activation"] if "final_activation" in config else None,
        "hidden_dims": config["hidden_dims"],
        "adamw_weight_decay": config['adamw_weight_decay'] if "adamw_weight_decay" in config else 1e-5,
        "factor": config['t_factor'] if "t_factor" in config else 0.9,
        "patience": config["t_patience"],
        "use_batchnorm": config['use_batchnorm'] if "use_batchnorm" in config else False,
        "dropout_rate": config['dropout_rate'] if "dropout_rate" in config else 0.1,
        "criterion": config['criterion'] if "criterion" in config else nn.MSELoss
    }
    
    # Create student model
    input_dim = X_tr.shape[1]
    if input_dim is None or student_kwargs.get("hidden_dims", None) is None:
        raise ValueError("For SMEDAL, input_dim and hidden_dims must be specified.")

    student_config = dict(
        input_dim=input_dim,
        latent_dim=tc["n_components"] if "n_components" in tc else 2,
        device="cuda",
        **student_kwargs
    )

    student = MEDAL(**student_config)
    
    teacher = tc['teacher']

    if teacher == "umap":
        prefix = f'{teacher}{tc["n_components"]}_{tc["t_n_neighbors"]}_{tc["min_dist"]}_{tc["teacher_seed"]}'
    elif teacher == "tsne":
        prefix = f'{teacher}{tc["n_components"]}_{tc["perplexity"]}_{tc["teacher_seed"]}'
    elif teacher == "pca":
        prefix = f'{teacher}{tc["n_components"]}_{tc["teacher_seed"]}'
    elif teacher == "spectral":
        prefix = f'{teacher}{tc["n_components"]}_{tc["t_n_neighbors"]}_{tc["teacher_seed"]}'

    # accounting for vanillaAE and linearAE
    if config["model_variant"] == "vanillaAE":
        prefix = f'vanillaAE_{tc["n_components"]}_{tc["teacher_seed"]}'
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

# def pretrain_task(config, num_pretrain_epochs=10):
#     X_tr, _ = load_and_split(config['dataset_name'], seed=config['teacher_seed'], test_size=1)

#     student_kwargs = {
#         "epochs": num_pretrain_epochs, 
#         "batch_size": config['batch_size'], 
#         "lambda_reg": 0.0, 
#         "warmup": config.get('warmup', 0),  # Use get for optional params
#         "lr": config['lr'], 
#         "eta_min1": config["eta_min1"], 
#         "eta_min2": config['eta_min2'],
#         "T_max": num_pretrain_epochs, 
#         "lambda_d": config['lambda_d'],
#         "activation": config['activation'],
#         "bottleneck_activation": config["bottleneck_activation"],
#         "hidden_dims": config["hidden_dims"],
#         "use_lbfgs": config["use_lbfgs"]
#     }

#     # Create student model
#     student = make_student(
#         method="drd",
#         input_dim=X_tr.shape[1],
#         latent_dim = 2,
#         device=DEVICE,
#         **student_kwargs,
#     )

#     return student.fit(X_tr, None, config['verbose'],
#                 phase="pretrain", 
#                 pretrained_path='/user/bnc2119/drd/results/pretrain',
#                 prefix=f"{config['dataset_name']}",
#                 )


if __name__ == "__main__":
    DEVICE = "cuda"
    dataset_name = "mnist"
    variant = "medal"
    teachers = ["umap"]

    config = get_dataset_config(dataset_name, variant, update_kws={
        "verbose": False,
        "pretrained_path": None,
        "retrain_teacher": False,
        "save_dir": PATH_PREFIX + f'/tmp_results/chkpt',
        # f'/user/bnc2119/drd/results/pretrain/{dataset_name}_pretrain.pt',
    }).copy()
    # config.update({
    #     "verbose": False,
    #     "pretrained_path": None,
    #     "retrain_teacher": False,
    #     # f'/user/bnc2119/drd/results/pretrain/{dataset_name}_pretrain.pt',
    # })
    # pretrain_ckpt_path = pretrain_task(config, num_pretrain_epochs=1000)
    # print(f"Pretraining completed. Checkpoint saved at {pretrain_ckpt_path}")

    teacher_grid = build_teacher_grid(dataset_name, teachers, seeds=range(5), mode = "teacher_sweep") # {"rank_sweep", "teacher_sweep"}
    print(f"Running {variant} on {dataset_name} with {len(teacher_grid)} teacher configs.")

    for tc in teacher_grid:
        precompute_teacher_embeddings(tc, config)

    config.update({
        "teacher_config": tune.grid_search(teacher_grid),
    })
    print(f"Total number of teacher configurations to evaluate: {len(teacher_grid)}")

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
