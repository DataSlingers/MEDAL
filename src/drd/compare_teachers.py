# run teacher evaluation as a grid search on ray
from ray import tune
from utils.eval_utils import make_student, load_and_split, eval_student, get_teacher_embeddings
from utils.process_astro import clean_astro_data
import os
import numpy as np
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7" 

PATH_PREFIX = "/shared/share_mala/irchang/drd"
DISTILL_BANDS_DICT = {
    "gene_cancer": [(1e-12, 9e-9)],
    "mnist": [(1e-12, 9e-8)],
    "single_cell": [(1e-12, 5e-7)],
    "wine": [(1e-12, 9e-8)],
    "hydra": [(1e-12, 9e-8)],
    "pbmc": [(1e-12, 9e-7)],
    "astro": [(1e-12, 5e-7)],
    "cortical": [(1e-12, 9e-8)],
    "macaque": [(1e-12, 5e-7)],
}

N_SAMPLES = {
    "gene_cancer": 801,
    "mnist": 10000,
    "wine": 178,
    "single_cell": 3589,
    "hydra": 25052,
    "pbmc":5858,
    "astro": 3286,
    "cortical": 23822
}

INIT_CONFIG = {
    "gene_cancer": {
        "lr": 1e-3,  
        "lambda_d": 15000, # 1500
        "eta_min1": 1e-5, #1e-9
        "eta_min2": 0.0, 
        "hidden_dims":[1000, 1000, 1000, 1000],
        "activation": "SELU",
        "bottleneck_activation": None,
        "max_epochs": 6000,
        # "T_max_ratio": 0.5,
        "T_max": 3000,
        "warmup": 0,  
        "seed": tune.grid_search([0, 1, 2, 3, 4]),
        "batch_size": 100
    },
    "wine": {
        "lr": 0.02, #0.003233466538536306,
        "lambda_d": 10000,
        "eta_min1": 1e-6, #6.076040435352558e-08,
        "eta_min2": 1.2312179372780289e-17,
        "hidden_dims": [258, 258, 258, 258],
        "activation": "SELU",
        "bottleneck_activation":  None,
        'max_epochs': 130000, 
        'T_max_ratio': 0.9,# 0.7,
        "warmup": 0, 
        "seed": tune.grid_search([0, 1, 2, 3, 4]),
        "batch_size": 100
    },
    "mnist": { # 10k or 1k
        "lr": 0.000269,	
        "lambda_d": 10000, # 3000
        # "lambda_d": 0, # for vanilla AE
        "eta_min1": 1e-8, # 7.256237e-10,
        "eta_min2": 1.587436e-16,
        "hidden_dims": [1000, 1000, 1000, 1000, 1000], 
        "activation": "SELU",
        "bottleneck_activation": None,
        # "bottleneck_activation": "SELU",
        "max_epochs": 3500,
        "T_max_ratio":0.6,
        "batch_size": 256,
        "warmup": 0, 
        "test_size":0.2
    },
    "diabetes": {  #(442, 10)
        "lr": 0.0253689, 
        "lambda_d": 10000,
        "eta_min1": 2e-4, 
        "eta_min2": 1e-7,
        "hidden_dims": [500, 500, 500, 500, 500, 500, 500],
        "activation": "SELU",
        "bottleneck_activation": None,
        "max_epochs": 100000,
        "T_max_ratio": 0.5,
        "batch_size": 512,
        "seed": tune.grid_search([0,1,2,3,4]), 
        "warmup": 0, 
    },
    "single_cell": {
        "lr": 0.001,
        "lambda_d": 50000,
        "eta_min1": 5.01855e-05, 
        "eta_min2": 4.1779e-07,
        "hidden_dims": [294, 294, 294, 294, 294, 294, 294, 294, 294],
        "activation": "SELU",
        "bottleneck_activation": None,
        'max_epochs': 30000, 
        'T_max_ratio': 0.6,
        "warmup": 0, 
        "seed": tune.grid_search([0, 1, 2, 3, 4]),
        "batch_size": 10000,
    },
    "hydra":{
        "lr": 0.004247903976428593,  # (umap)0.004247903976428593; (tsne)  0.000140849,
        "lambda_d": 30000, #(umap) 30000; tsne 10000
        "eta_min1":  9.10708e-06, 
        "eta_min2": 6.779079133197359e-07, # umap 6.779079133197359e-07;; tsne 8.51602e-10
        "hidden_dims": [309, 1792, 1792, 1792],
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 7000,
        'T_max_ratio': 0.7,
        "warmup": 1500, 
        # "seed": tune.grid_search([0, 1, 2, 3, 4]),
        "batch_size": 51200,
        "test_size": 0.2
    },
    "pbmc":{
        "lr": 0.001, 
        "lambda_d": 50000, 
        "eta_min1":3.4443667740771654e-05,
        "eta_min2": 1.069202395077256e-07,
        "hidden_dims": [500, 500, 500, 500, 500],
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 20000,
        'T_max_ratio': 0.7,
        "warmup": 3000, 
        "seed": tune.grid_search([0, 1, 2, 3, 4]),
        "batch_size": 10000,
    },
    # "astro":{
    #     "lr": 0.026063168211691228,
    #     "lambda_d": 50000,
    #     "eta_min1": 7.634351255256293e-05,
    #     "eta_min2": 1.3331622499633123e-06,
    #     "hidden_dims": [1000, 1000, 1000, 1000, 500, 500, 500, 500, 500],
    #     "activation": "SELU", 
    #     "bottleneck_activation": None,
    #     'max_epochs': 20000,
    #     'T_max_ratio': 0.75,
    #     "warmup": 500, 
    #     "seed": tune.grid_search([0, 1, 2, 3, 4]),
    #     "batch_size": 5000,
    #     "use_lbfgs": False
    # },
    "astro":{
        "lr": 0.021855091469559523,
        "lambda_d": 10000,
        "eta_min1": 7.634351255256293e-05, #9.286424485512266e-06,
        "eta_min2": 9.286424485512266e-06, # 3.365940709851557e-06,
        "hidden_dims": [512, 256, 128, 64, 32, 32],
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 20000,
        'T_max_ratio': 0.6,
        "warmup": 0, 
        "seed": tune.grid_search([0, 1, 2, 3, 4]),
        "batch_size": 512,
        "use_lbfgs": False,
        "test_size": 0.2
    },
    "cortical":{
        "lr": 0.00268681,
        "lambda_d": 30000,
        "eta_min1": 7.936e-05,
        "eta_min2": 1.61441e-07,
        "hidden_dims": [309, 1792, 1792, 1792],
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 6000,
        'T_max_ratio': 0.7,
        "warmup": 100, 
        "batch_size": 50000,
        "test_size": 0.2
    },
    "macaque":{       
        "lr": 0.001, 
        "lambda_d": 50000,
        "eta_min1":  6.24882e-06, 
        "eta_min2": 6e-7, 
        "hidden_dims": [700] * 15,
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 12000,
        'T_max_ratio': 0.7,
        "warmup": 0, 
        "batch_size": 1024,
        "test_size":0.2
    },
}

def precompute_teacher_embeddings(tc, config):
    X_tr, X_te = load_and_split(config['dataset_name'], seed=0, test_size=config["test_size"] if "test_size" in config else 1)
    try:
        # 'xb' = create file, fail if it already exists
        if tc['teacher'] == "umap":
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['t_n_neighbors']}_{tc['min_dist']}_{tc['seed']}_train.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    tc["teacher"], X_tr,
                    n_components= tc["n_components"] if "n_components" in tc else 2,
                    n_neighbors=tc["t_n_neighbors"], 
                    min_dist=tc["min_dist"],
                    random_state=tc['seed'],
                )
                np.save(f, Z_tr) 
            
        elif tc['teacher'] == "pca":
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{tc['seed']}_train.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    tc["teacher"], X_tr, 
                    n_components= tc["n_components"] if "n_components" in tc else 2,
                    random_state=tc['seed'],
                )
                np.save(f, Z_tr) 

        elif tc['teacher'] == "isomap":
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['t_n_neighbors']}_{tc['seed']}_train.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    tc["teacher"], X_tr, 
                    n_components= tc["n_components"] if "n_components" in tc else 2,
                    n_neighbors=tc["t_n_neighbors"],
                )
                np.save(f, Z_tr) 

        elif tc['teacher'] == "tsne":
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['perplexity']}_{tc['seed']}_train.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    tc["teacher"], X_tr, 
                    n_components= tc["n_components"] if "n_components" in tc else 2,
                    perplexity=tc["perplexity"],
                    learning_rate=tc["learning_rate"],
                    random_state=tc['seed'],
                )
                np.save(f, Z_tr) 

        elif tc['teacher'] == "spectral":
            model_path = Path(PATH_PREFIX / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['t_n_neighbors']}_{tc['seed']}_train.npy")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    tc["teacher"], X_tr,
                    n_components = tc["n_components"] if "n_components" in tc else 2,
                    n_neighbors = tc["t_n_neighbors"],
                    random_state = tc['seed'],
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
    if config['dataset_name'] == "astro":
        # extra processing needed for the astro dataset
        result = clean_astro_data(X_tr, X_te)
        X_tr, X_te = result["train"].to_numpy(), result["test"].to_numpy()
    
    if tc['teacher'] == "umap":
        model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['t_n_neighbors']}_{tc['min_dist']}_{tc['seed']}_train.npy"
    elif tc['teacher'] == "pca":
        model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{tc['seed']}_train.npy"
    elif tc['teacher'] == "isomap":
        model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['t_n_neighbors']}_{tc['seed']}_train.npy"
    elif tc['teacher'] == "tsne":
        model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['perplexity']}_{tc['seed']}_train.npy"
    elif tc['teacher'] == "spectral":
        model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}_{tc['t_n_neighbors']}_{tc['seed']}_train.npy"

    Z_tr = np.load(model_path)
    
    # Prepare student configuration
    student_kwargs = {
        "epochs": config['max_epochs'], 
        "batch_size": config['batch_size'], 
        "lambda_reg": 0.0, 
        "warmup": config.get('warmup', 0),  # Use get for optional params
        "lr": config['lr'], 
        "eta_min1": config["eta_min1"], 
        "eta_min2": config['eta_min2'],
        "T_max": int(config['max_epochs'] * config["T_max_ratio"]) if "T_max_ratio" in config else config["T_max"], 
        "lambda_d": config['lambda_d'],
        "activation": config['activation'],
        "bottleneck_activation": config["bottleneck_activation"],
        "hidden_dims": config["hidden_dims"],
    }
    
    # Create student model
    student = make_student(
        method="drd",
        input_dim=X_tr.shape[1],
        latent_dim = tc["n_components"] if "n_components" in tc else 2,
        device=DEVICE,
        **student_kwargs,
    )
    
    distill_bands = DISTILL_BANDS_DICT[config['dataset_name']]
    student.fit(X_tr, Z_tr, config['verbose'], 
                phase="finetune", pretrained_path=config['pretrained_path'],
                target_bands=distill_bands, 
                stability_window=10, 
                epsilon_distill=1e-7, epsilon_recon=1e-3, 
                patience=20, # unit = epoch
                return_on_stable=True,
                # checkpointing
                save_dir = PATH_PREFIX + f'/results/chkpt/{config["dataset_name"]}', 
                # prefix=f'{tc["teacher"]}_{tc["t_n_neighbors"]}_{tc["min_dist"]}_{tc["seed"]}',
                # prefix=f'{tc["teacher"]}_{tc["t_n_neighbors"]}',
                # prefix=f'{tc["teacher"]}_{tc["perplexity"]}_{tc["seed"]}',
                prefix=f'{tc["teacher"]}{tc["n_components"]}_{tc["seed"]}',
                )

# def pretrain_task(config, num_pretrain_epochs=10):
#     X_tr, _ = load_and_split(config['dataset_name'], seed=config['seed'], test_size=1)

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
    dataset_name = "cortical"
    path = f"{PATH_PREFIX}/compare_teachers"
    filename=f"{dataset_name}_pca.csv" 

    config = INIT_CONFIG[dataset_name].copy()
    config.update({
        "dataset_name": dataset_name,
        "verbose": False,
        "pretrained_path": None
        # f'/user/bnc2119/drd/results/pretrain/{dataset_name}_pretrain.pt',
    })
    # pretrain_ckpt_path = pretrain_task(config, num_pretrain_epochs=1000)
    # print(f"Pretraining completed. Checkpoint saved at {pretrain_ckpt_path}")

    teacher_grid = []

    # sample_lim = int(INIT_CONFIG[dataset_name]["test_size"] * (N_SAMPLES[dataset_name] - 1)) if "test_size" in INIT_CONFIG[dataset_name] else N_SAMPLES[dataset_name] - 1

    # UMAP combos
    # for n in np.unique(np.logspace(np.log(5), np.log10(1000), 15).astype(int))[:-3]:
    # for n in np.unique(np.logspace(0, np.log10(1000), 20).astype(int))[1:]:
    # for n in [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 80, 160]:
    # for n in [5, 15]:
    # for n in np.unique(np.arange(5, 200, 25).astype(int)): # Hydra
        # for md in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
        # for seed in [0, 1, 2, 3, 4]:
        #     teacher_grid.append({
        #         "teacher": "umap",
        #         "n_components": 2,
        #         "t_n_neighbors": n,
        #         "min_dist": 0.1,
        #         "seed": seed
        #     })
        #     precompute_teacher_embeddings(teacher_grid[-1], config)

    # t-SNE combos
    # for perp in np.unique(np.logspace(np.log10(5), np.log10(200), 10).astype(int))[::-1]:
    # for perp in np.unique(np.arange(10, 420, 20).astype(int)):
    # for perp in np.unique(np.arange(5, 141, 5).astype(int)):
        # for seed in [1]: # [0, 1, 2, 3, 4]:
            # teacher_grid.append({
            #     "teacher": "tsne",
            #     "n_components": 2,
            #     "perplexity": perp,
            #     "learning_rate": 'auto',
            #     "seed": seed
            # })
            # precompute_teacher_embeddings(teacher_grid[-1], config)

    # Isomap combos
    # for n in np.unique(np.logspace(0, 2.9, 6).astype(int))[1:]:
    # # for n in [15, 50]:
    #     teacher_grid.append({
    #         "teacher": "isomap",
    #         "t_n_neighbors": n,
    #     })

    # Spectral combos
    # for n in np.unique(np.logspace(np.log10(5), np.log10(1000), 10).astype(int)):
    #     for seed in [0, 1, 2, 3, 4]:
    #         teacher_grid.append({
    #             "teacher": "spectral",
    #             "n_components": 2,
    #             "t_n_neighbors": n,
    #             "random_state": seed
    #         })
    #         precompute_teacher_embeddings(teacher_grid[-1])

    # PCA
    # for c in range(2, 21):
    for seed in [0, 1, 2, 3, 4]:
        teacher_grid.append({
            "teacher": "pca",
            "n_components": 2,
            "seed": seed
        })
        precompute_teacher_embeddings(teacher_grid[-1], config)


    config.update({
        "teacher_config": tune.grid_search(teacher_grid),
    })
    print(f"Total number of teacher configurations to evaluate: {len(teacher_grid)}")

    # input: dataset name, teacher_method
    analysis = tune.run(
        compare_teacher,
        name="drd_teacher_sweep",
        num_samples=1, 
        resources_per_trial={"cpu": 5, "gpu": 0.25},  # Adjusted GPU allocation
        config= config,
        verbose=1,
        max_failures=3,
        storage_path="/tmp/ray_results"
    )

    base = Path(path)
    base.mkdir(parents=True, exist_ok=True)
    save_path = base / filename
    analysis.results_df.to_csv(save_path)

    print(f"Results saved to {save_path}")