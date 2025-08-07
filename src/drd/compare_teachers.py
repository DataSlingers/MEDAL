# run teacher evaluation as a grid search on ray
from ray import tune
from utils.eval_utils import make_student, load_and_split, fit_student, eval_student, get_teacher_embeddings
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6" 

PATH_PREFIX = "/shared/share_mala/irchang/drd"

INIT_CONFIG = {
    "gene_cancer": {
        "lr": 0.0001,  
        "lambda_d": 1500,
        "eta_min1": 1e-16, 
        "eta_min2": 0.0, 
        "hidden_dims":[1000, 800, 400, 200],
        "activation": "ReLU",
        "bottleneck_activation": None,
        "max_epochs": 1500,
        "T_max_ratio": 2/3,
        "warmup": 0,  
        "seed": 0,
        "batch_size": 100
    },
    "wine": {
        "lr": 0.003233466538536306,
        "lambda_d": 5000,
        "eta_min1": 6.076040435352558e-08,
        "eta_min2": 1.2312179372780289e-17,
        "hidden_dims": [1500, 1000, 500, 250, 2000],
        "activation": "SELU",
        "bottleneck_activation":  None,
        'max_epochs': 130000, 
        'T_max_ratio': 0.7,
        "warmup": 0, 
        "seed": 0 ,
        "batch_size": 100
    },
    "mnist": { # 10000
        "lr": 0.000269,	
        "lambda_d": 2000,
        "eta_min1": 7.256237e-10,
        "eta_min2": 1.587436e-16,
        "hidden_dims": tune.choice([
            [50000, 10000, 5000, 2500, 1000],
            [10000, 5000, 2500, 1000, 50000],
            [10000, 50000, 5000, 2500, 1000],
            [10000, 5000, 50000, 2500, 1000],
            [10000, 5000, 2500, 50000, 1000],
        ]),
        "activation": tune.choice(["ReLU", "SELU"]),
        "bottleneck_activation": tune.choice(["ReLU", "SELU", None]),
        "max_epochs": 3500,
        "T_max_ratio":0.6,
        "batch_size": 256,
        "seed": tune.choice([0, 1, 2, 3]),
        "warmup": 0, 
    },
    "diabetes": { 
        "lr": 0.0253689, 
        "lambda_d": 5000,
        "eta_min1": 2e-4, 
        "eta_min2": 1.2637e-6,
        "hidden_dims": tune.choice([
            [441, 220, 110, 5000],
            [5000, 441, 220, 110],
            [441, 5000, 220, 110],
            [441, 220, 5000, 110],
        ]),
        "activation": tune.choice(["ReLU", "SELU"]),
        "bottleneck_activation": tune.choice(["ReLU", "SELU", None]),
        "max_epochs": 100000,
        "T_max_ratio": 0.9,
        "batch_size": 512,
        "seed": tune.choice([0, 1, 2]), 
        "warmup": 0, 
    },
    "single_cell": {
        "lr": 1e-4,
        "lambda_d": 1000,
        "eta_min1":1e-6, 
        "eta_min2":1e-9,
        "hidden_dims": tune.choice([
            [35890, 17945, 8972, 4486],
            [17945, 35890, 8972, 4486],
            [17945, 8972, 35890, 4486],
            [17945, 8972, 4486, 35890],
        ]),
        "activation": tune.choice(["ReLU", "SELU"]),
        "bottleneck_activation": tune.choice(["ReLU", "SELU", None]),
        'max_epochs': 1500, 
        'T_max_ratio': 0.5,
        "warmup": 0, 
        "seed": tune.choice([0, 1, 2]),
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
            n_components=2,
            n_neighbors=tc["t_n_neighbors"], 
            min_dist=tc["min_dist"],
            random_state=config['seed'],
        )
    elif tc['teacher'] == "pca":
        Z_tr, _ = get_teacher_embeddings(
            tc["teacher"], X_tr, 
            n_components=2,
            random_state=config['seed'],
        )
    elif tc['teacher'] == "isomap":
        Z_tr, _ = get_teacher_embeddings(
            tc["teacher"], X_tr, 
            n_components=2,
            n_neighbors=tc["t_n_neighbors"],
        )
    elif tc['teacher'] == "tsne":
        Z_tr, _ = get_teacher_embeddings(
            tc["teacher"], X_tr, 
            n_components=2,
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
    
    # Handle checkpointing (if resuming from checkpoint)
    
    fit_student(student, X_tr, Z_tr, verbose=config['verbose'])

if __name__ == "__main__":
    DEVICE = "cuda"
    dataset_name = "wine"
    path = f"{PATH_PREFIX}/compare_teachers/{dataset_name}.csv" 

    teacher_grid = []

    # UMAP combos
    for n in [10, 15, 20]:
        for md in [0.01, 0.1]:
            teacher_grid.append({
                "teacher": "umap",
                "t_n_neighbors": n,
                "min_dist": md,
            })

    # t-SNE combos
    for perp in [30, 50]:
        for lr in [200, 300]:
            teacher_grid.append({
                "teacher": "tsne",
                "perplexity": perp,
                "learning_rate": lr,
            })

    # Isomap combos
    for n in [5, 10, 15, 20]:
        teacher_grid.append({
            "teacher": "isomap",
            "t_n_neighbors": n,
        })

    teacher_grid.append({
        "teacher": "pca",
    })

    config = INIT_CONFIG[dataset_name].copy()
    config.update({
        "dataset_name": dataset_name,
        "teacher_config": tune.grid_search(teacher_grid),
        "verbose": False,
    })

    # input: dataset name, teacher_method
    analysis = tune.run(
        compare_teacher,
        name="drd_teacher_sweep",
        num_samples=1, 
        resources_per_trial={"cpu": 6, "gpu": 0.1},  # Adjusted GPU allocation
        config= config,
        verbose=1,
        max_failures=3,
        storage_path="/shared/share_mala/irchang/drd/ray_results"
    )

    analysis.results_df.to_csv(path)