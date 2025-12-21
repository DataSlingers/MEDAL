# run teacher evaluation as a grid search on ray
from ray import tune
from utils.eval_utils import make_student, load_and_split, eval_student, get_teacher_embeddings
import os
import numpy as np
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7" 

PATH_PREFIX = "/shared/share_mala/irchang/drd"
DISTILL_BANDS_DICT = {
    "gene_cancer": [(1e-12, 5e-7)],
    "mnist": [ (1e-12, 9e-6)], #(1e-1, 5), (1e-2, 1e-1), (1e-4, 1e-2), (1e-6,1e-4), (9e-8, 1e-6),
    "single_cell": [(1e-12, 5e-7)],
    "wine": [(1e-12, 9e-8)],
    "hydra": [(1e-12, 9e-6)],
    #[(1e-1, 1e8), (1e-2, 1e-1), (1e-4, 1e-2), (1e-6,1e-4), (9e-8, 1e-6), (1e-12, 9e-8)],
    "pbmc": [(1e-12, 9e-6)],
    "astro": [(1e-12, 9e-6)],
    "cortical": [(1e-12, 9e-6)],
    "macaque": [(1e-12, 9e-6)],
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
        "lr": 1e-3, #tune.grid_search([1e-5, 1e-4, 5e-4, 5e-5]), #1e-3,  
        "lambda_d": 10000,#30000, # 1500
        # "lambda_d": 0,
        "eta_min1": 1e-11, #1e-9-
        "eta_min2": 0, 
        "hidden_dims":[1000, 1000, 1000, 1000],
        "activation": "SELU",
        # "activation": None,
        "bottleneck_activation": None,
        "max_epochs": 7000,
        "T_max_ratio": 1, #0.5,
        # "T_max": 7000,
        "warmup": 0, 
        "test_size": 0.2,
        "batch_size": 100
    },
    "wine": {
        "lr": 0.02, #0.003233466538536306,
        "lambda_d": 20000,
        "eta_min1": 1e-8, #6.076040435352558e-08,
        "eta_min2": 1.2312179372780289e-17,
        "hidden_dims": [258, 258, 258, 258],
        "activation": "SELU",
        "bottleneck_activation":  None,
        'max_epochs': 130000, 
        'T_max_ratio': 0.9,# 0.7,
        "warmup": 0, 
        "batch_size": 100
    },
    "mnist": { 
        "lr": 1e-3, #1e-3 (ROP), # 2e-5, #0.000269 (tsne),	
        "lambda_d": 10000, # 3000
        # "lambda_d": 0, # for vanilla AE
        "eta_min1": 1e-7, #1e-5, # 7.256237e-10, 1e-10(spectral)
        "hidden_dims": [1000, 1000, 1000, 1000, 1000], 
        "activation": "SELU",
        # "activation": None,
        "bottleneck_activation": None,
        "max_epochs":30000, #100000,
        "T_max_ratio":1,
        "batch_size": 256,
        "warmup": 0, 
        "test_size":0.2,
        "t_patience": 20,
    },
    "single_cell": {
        "lr": 0.001,
        "lambda_d": 50000,
        "eta_min1": 5.01855e-05, 
        "eta_min2": 4.1779e-07,
        "hidden_dims": [294, 294, 294, 294, 294, 294, 294, 294, 294],
        "activation": "SELU",
        "bottleneck_activation": None,
        'max_epochs': 20000, 
        'T_max_ratio': 0.6,
        "warmup": 0, 
        "batch_size": 10000,
    },
    "hydra":{
        "lr": 0.005, #0.0005 (old lr),# 0.005 (new lr), 
        "lambda_d": 30000, #30000,
        # "lambda_d": 0,
        # "eta_min1":  9.10708e-06, 
        # "eta_min2": 8.51602e-10, 
        "eta_min1":  1e-07, 
        "hidden_dims": [309, 1792, 1792, 1792],
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 20000,
        'T_max_ratio': 1,  #0.8,
        "warmup": 1500, 
        "batch_size": 51200,
        "test_size": 0.2,
        "t_patience":20,
        "t_factor": 0.95,
        "use_batchnorm": True
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
        "batch_size": 10000,
    },
    "astro":{
        "lr": 0.0005, #0.00139911,
        "lambda_d": 10000,
        "eta_min1": 1e-07, #3.55767e-07,
        "hidden_dims": [700] * 15,
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 60000,
        'T_max_ratio': 1, #0.75,
        "warmup": 0, 
        "batch_size": 25600,
        "test_size": 0.2,
        "t_patience":70,
        "t_factor": 0.9,
        "use_batchnorm": False
    },
    "cortical":{
        "lr": 0.00268681,
        "lambda_d": 50000, #30000,
        # "eta_min1": 7.936e-05, #7.936e-05,
        # "eta_min2": 1e-7,
        "eta_min1": 1e-7,
        "hidden_dims": [309, 1792, 1792, 1792],
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 6000,
        'T_max_ratio': 1, #0.7
        "warmup": 100, 
        "batch_size": 50000,
        "test_size": 0.2,
        "t_patience":20,
        "t_factor": 0.95,
        "use_batchnorm": True
    },
    "macaque":{       
        "lr":0.000341178, 
        "lambda_d": 10000, # 50000, 200000
        "eta_min1": 1e-7, #6.24882e-06 (tsne), 1e-7 (umap)
        "hidden_dims": [700] * 15,
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 25000,
        'T_max_ratio': 1, #0.7, 0.8
        "warmup": 0, 
        "batch_size": 1024,
        "test_size":0.2,
        "t_patience":70,
        "t_factor": 0.9,
        "use_batchnorm": False
    },
}

def precompute_teacher_embeddings(tc, config):
    X_tr, X_te = load_and_split(config['dataset_name'], seed=0, test_size=config["test_size"] if "test_size" in config else 1)
    load_this_seed = tc['teacher_seed'] if config['retrain_teacher'] else 0
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
                    random_state=tc['teacher_seed'],
                )
                np.save(f, Z_tr) 
            
        elif tc['teacher'] == "pca":
            model_path = Path(PATH_PREFIX) / f"embeddings/{config['dataset_name']}_{tc['teacher']}{tc['n_components']}_{load_this_seed}_train.npy"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "xb") as f:
                Z_tr = get_teacher_embeddings(
                    tc["teacher"], X_tr, 
                    n_components= tc["n_components"] if "n_components" in tc else 2,
                    random_state=tc['teacher_seed'],
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
                    random_state=tc['teacher_seed'],
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
                    random_state = tc['teacher_seed'],
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

    load_this_seed = tc['teacher_seed'] if config['retrain_teacher'] else 0
    
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
        "lambda_reg": 0.0, 
        "warmup": config.get('warmup', 0),  # Use get for optional params
        "lr": config['lr'], 
        "eta_min1": config["eta_min1"], 
        "eta_min2": config["eta_min2"] if "eta_min2" in config else 0, 
        "T_max": int(config['max_epochs'] * config["T_max_ratio"]) if "T_max_ratio" in config else config["T_max"], 
        "lambda_d": config['lambda_d'],
        "activation": config['activation'],
        "bottleneck_activation": config["bottleneck_activation"],
        "hidden_dims": config["hidden_dims"],
        "adamw_weight_decay": config['adamw_weight_decay'] if "adamw_weight_decay" in config else 1e-5,
        "factor": config['t_factor'] if "t_factor" in config else 0.9,
        "patience": config["t_patience"],
        "use_batchnorm": config['use_batchnorm'] if "use_batchnorm" in config else False,

    }
    
    # Create student model
    student = make_student(
        method="drd",
        input_dim=X_tr.shape[1],
        latent_dim = tc["n_components"] if "n_components" in tc else 2,
        device=DEVICE,
        **student_kwargs,
    )
    
    teacher = tc['teacher']

    if teacher == "umap":
        prefix = f'{teacher}{tc["n_components"]}_{tc["t_n_neighbors"]}_{tc["min_dist"]}_{tc["teacher_seed"]}'
    elif teacher == "tsne":
        prefix = f'{teacher}{tc["n_components"]}_{tc["perplexity"]}_{tc["teacher_seed"]}'
    elif teacher == "pca":
        prefix = f'{teacher}{tc["n_components"]}_{tc["teacher_seed"]}'
    elif teacher == "spectral":
        prefix = f'{teacher}{tc["n_components"]}_{tc["t_n_neighbors"]}_{tc["teacher_seed"]}'


    distill_bands = DISTILL_BANDS_DICT[config['dataset_name']]
    student.fit(X_tr, Z_tr, config['verbose'], 
                phase="finetune", pretrained_path=config['pretrained_path'],
                target_bands=distill_bands, 
                stability_window=20, 
                epsilon_distill=1e-7, epsilon_recon=1e-2, # 1e-2 for macauqe, others 1e-3 
                patience=50, # unit = epoch
                return_on_stable=True,
                # checkpointing
                # save_dir=None,
                # save_dir = '/tmp/results/chkpt/{config["dataset_name"]}',
                save_dir = PATH_PREFIX + f'/tmp_results/chkpt/{config["dataset_name"]}',
                # save_dir = f'/tmp/results/chkpt/{config["dataset_name"]}', 
                prefix = prefix,
                # prefix=f'vanillaAE_{tc["n_components"]}_{tc["teacher_seed"]}',
                # prefix=f'linearAE_{tc["teacher"]}{tc["n_components"]}_{tc["teacher_seed"]}'
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
    dataset_name = "macaque"
    path = f"{PATH_PREFIX}/compare_teachers"
    filename=f"{dataset_name}_umap.csv" 

    config = INIT_CONFIG[dataset_name].copy()
    config.update({
        "dataset_name": dataset_name,
        "verbose": False,
        "pretrained_path": None,
        "retrain_teacher": False,
        # f'/user/bnc2119/drd/results/pretrain/{dataset_name}_pretrain.pt',
    })
    # pretrain_ckpt_path = pretrain_task(config, num_pretrain_epochs=1000)
    # print(f"Pretraining completed. Checkpoint saved at {pretrain_ckpt_path}")

    teacher_grid = []

    # UMAP combos

    for teacher_seed in range(5):
        # for n in np.unique(np.logspace(np.log10(5), np.log10(500), 15).astype(int)): # astro, mnist
        # for n in np.unique(np.logspace(np.log10(5), np.log10(2000), 10).astype(int)): # tasic, hydra
        #     teacher_grid.append({
        #         "teacher": "umap",
        #         "n_components": 2,
        #         "t_n_neighbors": int(n),
        #         "min_dist": 0.1,
        #         "teacher_seed": teacher_seed
        #     })
        #     precompute_teacher_embeddings(teacher_grid[-1], config)

        # t-SNE combos
        # for perp in np.unique(np.logspace(np.log10(5), np.log10(5000), 10).astype(int)): #newHydra
        # for perp in np.unique(np.logspace(np.log10(5), np.log10(6000), 10).astype(int)): # tasic
        for perp in np.unique(np.logspace(np.log10(5), np.log10(500), 10).astype(int)): # Macaque
        # for perp in np.unique(np.logspace(np.log10(3), np.log10(500), 15).astype(int)): #newAstro 
        # for perp in np.unique(np.arange(10, 420, 20).astype(int)): # Hydra
        # for perp in [5, 11,  27, 62, 146,  341, 793, 1846, 4297]:# MNIST
            teacher_grid.append({
                "teacher": "tsne",
                "n_components": 2,
                "perplexity": int(perp),
                "learning_rate": 'auto',
                "teacher_seed": teacher_seed
            })
            precompute_teacher_embeddings(teacher_grid[-1], config)

        # Isomap combos
        # for n in np.unique(np.logspace(0, 2.9, 6).astype(int))[1:]:
        # # for n in [15, 50]:
        #     teacher_grid.append({
        #         "teacher": "isomap",
        #         "t_n_neighbors": n,
        #     })

        # Spectral combos
        # for n in np.unique(np.logspace(np.log10(5), np.log10(200), 15).astype(int)): #Tasic, Hydra
        # for n in np.unique(np.logspace(np.log10(5), np.log10(500), 15).astype(int)): #MNIST, Astro
            # teacher_grid.append({
            #     "teacher": "spectral",
            #     "n_components": 2,
            #     "t_n_neighbors":int(n),
            #     "teacher_seed": teacher_seed
            # })
            # precompute_teacher_embeddings(teacher_grid[-1], config)

        # PCA
        # for c in np.unique(np.logspace(np.log10(2), np.log10(100), 15).astype(int))[::-1]:
        for c in [2]:
            teacher_grid.append({
                "teacher": "pca",
                "n_components": c,
                "teacher_seed": teacher_seed
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
        resources_per_trial={"cpu": 4, "gpu": 1},  # Adjusted GPU allocation
        config= config,
        verbose=1,
        max_failures=3,
        storage_path="/tmp/ray_results"
    )

    base = Path(path) 
    base.mkdir(parents=True, exist_ok=True)
    save_path = base / filename
    # analysis.results_df.to_csv(save_path)

    # print(f"Results saved to {save_path}")
