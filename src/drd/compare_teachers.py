# run teacher evaluation as a grid search on ray
from ray import tune
from utils.eval_utils import make_student, load_and_split, fit_student, eval_student, get_teacher_embeddings
import os
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7" 

PATH_PREFIX = "/shared/share_mala/irchang/drd"
DISTILL_BANDS_DICT = {
    "gene_cancer": [(1e-12, 9e-9)],
    "mnist": [(1e-12, 9e-7)],
    "single_cell": [(1e-12, 9e-7)],
    "diabetes": [(1e-12, 9e-7)],
}

INIT_CONFIG = {
    "gene_cancer": {
        "lr": 1e-3,  
        "lambda_d": 1500, # 1500
        "eta_min1": 1e-9, 
        "eta_min2": 0.0, 
        "hidden_dims":[1000, 1000, 1000, 1000],
        "activation": "SELU",
        "bottleneck_activation": None,
        "max_epochs": 4000,
        "T_max_ratio": 0.5,
        "warmup": 0,  
        "seed": tune.grid_search([0, 1, 2, 3, 4]),
        "batch_size": 100
    },
    "wine": {
        "lr": 0.003233466538536306,
        "lambda_d": 5000,
        "eta_min1": 6.076040435352558e-08,
        "eta_min2": 1.2312179372780289e-17,
        "hidden_dims": [258, 258, 258, 258],
        "activation": "SELU",
        "bottleneck_activation":  None,
        'max_epochs': 130000, 
        'T_max_ratio': 0.7,
        "warmup": 0, 
        "seed": 0 ,
        "batch_size": 100
    },
    "mnist": { # 10k or 1k
        "lr": 0.000269,	
        "lambda_d": 10000, # 3000
        "eta_min1": 1e-8, #7.256237e-10,
        "eta_min2": 1.587436e-16,
        "hidden_dims": [1000, 1000, 1000, 1000, 1000],
        "activation": "SELU",
        "bottleneck_activation": None,
        "max_epochs": 3500,
        "T_max_ratio":0.6,
        "batch_size": 256,
        "seed": 0,
        # tune.grid_search([0, 1, 2, 3, 4]),
        "warmup": 0, 
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
        "lr": 0.00020521674907073966,
        "lambda_d": 3000,
        "eta_min1": 3.0205023628787573e-06, 
        "eta_min2": 1.4506322965991254e-09,
        "hidden_dims": [300,300,300,300,300,300,300],
        "activation": "SELU",
        "bottleneck_activation": None,
        'max_epochs': 20000, 
        'T_max_ratio': 0.7,
        "warmup": 0, 
        "seed": tune.grid_search([0, 1, 2, 3, 4]),
        "batch_size": 10000
    }
}

def compare_teacher(config):
    """
    Trainable function for PBT optimization of DRD model.
    Reports distill_loss at regular intervals for PBT to use.
    """
    # Load and prepare data
    tc = config["teacher_config"]
    X_tr, X_te = load_and_split(config['dataset_name'], seed=config['seed'], test_size=1)
    if tc['teacher'] == "umap":
        Z_tr, _ = get_teacher_embeddings(
            tc["teacher"], X_tr, 
            n_components= tc["n_components"] if "n_components" in tc else 2,
            n_neighbors=tc["t_n_neighbors"], 
            min_dist=tc["min_dist"],
            random_state=config['seed'],
        )
    elif tc['teacher'] == "pca":
        Z_tr, _ = get_teacher_embeddings(
            tc["teacher"], X_tr, 
            n_components= tc["n_components"] if "n_components" in tc else 2,
            random_state=config['seed'],
        )
    elif tc['teacher'] == "isomap":
        Z_tr, _ = get_teacher_embeddings(
            tc["teacher"], X_tr, 
            n_components= tc["n_components"] if "n_components" in tc else 2,
            n_neighbors=tc["t_n_neighbors"],
        )
    elif tc['teacher'] == "tsne":
        Z_tr, _ = get_teacher_embeddings(
            tc["teacher"], X_tr, 
            n_components= tc["n_components"] if "n_components" in tc else 2,
            perplexity=tc["perplexity"],
            learning_rate=tc["learning_rate"],
        )
    
    # Prepare student configuration
    student_kwargs = {
        "epochs": config['max_epochs'], 
        "batch_size": config['batch_size'], 
        "lambda_reg": 0.0, 
        "warmup": config.get('warmup', 0),  # Use get for optional params
        "lr": config['lr'], 
        "eta_min1": config["eta_min1"], 
        "eta_min2": config['eta_min2'],
        "T_max": int(config['max_epochs'] * config["T_max_ratio"]), 
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
                target_bands=distill_bands, stability_window=10, 
                epsilon_distill=1e-7, epsilon_recon=1e-3, 
                patience=20, # unit = epoch
                return_on_stable=True,
                # checkpointing
                save_dir = '/user/bnc2119/drd/results/chkpt', prefix=f'{tc["teacher"]}_{tc["t_n_neighbors"]}',
                )

def pretrain_task(config, num_pretrain_epochs=10):
    X_tr, _ = load_and_split(config['dataset_name'], seed=config['seed'], test_size=1)

    student_kwargs = {
        "epochs": num_pretrain_epochs, 
        "batch_size": config['batch_size'], 
        "lambda_reg": 0.0, 
        "warmup": config.get('warmup', 0),  # Use get for optional params
        "lr": config['lr'], 
        "eta_min1": config["eta_min1"], 
        "eta_min2": config['eta_min2'],
        "T_max": num_pretrain_epochs, 
        "lambda_d": config['lambda_d'],
        "activation": config['activation'],
        "bottleneck_activation": config["bottleneck_activation"],
        "hidden_dims": config["hidden_dims"]
    }

    # Create student model
    student = make_student(
        method="drd",
        input_dim=X_tr.shape[1],
        latent_dim = 2,
        device=DEVICE,
        **student_kwargs,
    )

    return student.fit(X_tr, None, config['verbose'],
                phase="pretrain", 
                pretrained_path='/user/bnc2119/drd/results/pretrain',
                prefix=f"{config['dataset_name']}",
                )
    

if __name__ == "__main__":
    DEVICE = "cuda"
    dataset_name = "mnist"
    path = f"{PATH_PREFIX}/compare_teachers/{dataset_name}_umap_viz2.csv" 

    # config = INIT_CONFIG[dataset_name].copy()
    # config.update({
    #     "dataset_name": dataset_name,
    #     "verbose": True,
    #     "eta_min1": 1e-8,
    #     "seed": 0,
    # })
    # pretrain_ckpt_path = pretrain_task(config, num_pretrain_epochs=1000)
    # print(f"Pretraining completed. Checkpoint saved at {pretrain_ckpt_path}")

    teacher_grid = []

    # UMAP combos
    # for n in [10, 15, 20]:
    # for n in [15, 50, 80, 100, 200, 500, 1000]:
    for n in [15, 500]:
        # for md in [0.1]:
        teacher_grid.append({
            "teacher": "umap",
            "n_components": 100,
            "t_n_neighbors": n,
            "min_dist": 0.1,
        })

    # t-SNE combos
    # for perp in [10, 50, 110, 200]:
    # for lr in np.arange(100, 500, 20):
        # teacher_grid.append({
        #     "teacher": "tsne",
        #     "perplexity": perp,
        #     "learning_rate": 200,
        # })

    # Isomap combos
    # for n in [5, 10, 15, 20]:
    # for n in [15, 50]:
    #     teacher_grid.append({
    #         "teacher": "isomap",
    #         "t_n_neighbors": n,
    #     })
    
    # for c in range(2, 21):
    #     teacher_grid.append({
    #         "teacher": "pca",
    #         "n_components": c
    #     })

    config = INIT_CONFIG[dataset_name].copy()
    config.update({
        "dataset_name": dataset_name,
        "teacher_config": tune.grid_search(teacher_grid),
        "verbose": False,
        "pretrained_path": None
        # f'/user/bnc2119/drd/results/pretrain/{dataset_name}_pretrain.pt',
    })

    # input: dataset name, teacher_method
    analysis = tune.run(
        compare_teacher,
        name="drd_teacher_sweep",
        num_samples=1, 
        resources_per_trial={"cpu": 5, "gpu": 0.5},  # Adjusted GPU allocation
        config= config,
        verbose=1,
        max_failures=3,
        storage_path="/shared/share_mala/irchang/drd/ray_results"
    )

    analysis.results_df.to_csv(path)
    print(f"Results saved to {path}")