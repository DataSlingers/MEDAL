from ray import tune
from ray.tune.schedulers import PopulationBasedTraining, AsyncHyperBandScheduler
from ray.tune import CLIReporter
from utils.eval_utils import make_student, load_and_split, get_teacher_embeddings, fit_student, eval_student
from torch.utils.data import DataLoader, TensorDataset
import torch, torch.nn as nn
import tqdm, os
import torch.nn.functional as F
import numpy as np
from ray import train
import os, tempfile
import ray.cloudpickle as pickle
from ray.train import Checkpoint

DEVICE = "cuda"
dataset_name = "single_cell"
teacher_name = "isomap_neigh10"
PATH_PREFIX = "/shared/share_mala/irchang/drd"
path = f"{PATH_PREFIX}/tune_results/activation_{dataset_name}_{teacher_name}.csv"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,4,5,6,7" 

# HYPERPARAMETER_CONFIG = {
#     "single_cell": {
#         "lr": tune.loguniform(1e-5, 1e-1),  # Multiply by these factors
#         "lambda_d": tune.choice([25, 50, 100, 200, 500, 1000, 1500, 2000, 3000]),
#         "eta_min1": tune.loguniform(1e-12, 1e-5),  # Multiply by these factors
#         "eta_min2": tune.loguniform(1e-20, 1e-12), # More aggressive mutation for eta_min2
#         "hidden_dims": tune.choice([[1000, 800, 400, 200],
#                                    [200, 400, 80, 1000],
#                                    [8010]]),
#         "max_epochs": tune.choice([1500, 100000, 150000]),
#         "T_max_ratio": tune.choice([0.25, 0.5, 0.75, 0.9]),
#     },
#     "wine": {
#         "lr": tune.loguniform(1e-3, 1e-2),  
#         "lambda_d": tune.choice([1500, 2000, 3000, 4000, 5000]),
#         "eta_min1": tune.loguniform(1e-8, 1e-6), 
#         "eta_min2": tune.loguniform(1e-20, 1e-14), 
#         "hidden_dims": tune.choice([
#             [708, 531, 354, 177],
#             [885, 708, 531, 354, 177],
#             [1000, 750, 500, 250],
#             [2000, 1500, 1000, 500],
#             [2000, 1500, 1000, 500, 250]
#         ]),
#         "activation": tune.choice(["ReLU", None]),
#         "max_epochs": tune.choice([100000, 120000, 130000, 150000, 200000]),
#         "T_max_ratio": tune.choice([0.6, 0.7, 0.8, 0.9]),
#     },
#     "mnist": { 
#         "lr": tune.loguniform(1e-5, 1e-1), 
#         "lambda_d": tune.choice([25, 50, 100, 200, 500, 1000, 1500, 2000, 3000]),
#         "eta_min1": tune.loguniform(1e-12, 1e-5), 
#         "eta_min2": tune.loguniform(1e-20, 1e-12),
#         "hidden_dims": tune.choice([
#             [50000],
#             [20000, 10000],
#             [10000, 5000, 2500, 1000],
#             [50000, 10000, 5000, 2500, 1000],
#             [10000, 5000, 2500, 1000, 50000],
#             [10000]
#         ]),
#         "activation": tune.choice(["ReLU", None]),
#         "max_epochs": tune.choice([1500, 2000, 2500, 3000]),
#         "T_max_ratio": tune.choice([0.6, 0.7, 0.8, 0.9]),
#     },
#     "diabetes": { 
#         "lr": tune.loguniform(1e-5, 1e-1), 
#         "lambda_d": tune.choice([25, 50, 100, 200, 500, 1000, 1500, 2000, 3000]),
#         "eta_min1": tune.loguniform(1e-12, 1e-5), 
#         "eta_min2": tune.loguniform(1e-20, 1e-12),
#         "hidden_dims": tune.choice([
#             [441],
#             [882],
#             [441, 220, 110],
#             [441, 220, 110, 55],
#             [2000, 441, 220, 110],
#             [441, 220, 110, 2000]
#         ]),
#         "activation": tune.choice(["ReLU", None]),
#         "max_epochs": tune.choice([3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]),
#         "T_max_ratio": tune.choice([0.6, 0.7, 0.8, 0.9]),
#     }
# }

INIT_CONFIG = {
    "gene_cancer": {
        "teacher": "tsne",
        "t_n_neighbors": 10,
        "perplexity": 30,
        "learning_rate": 200,
        "lr": 0.0001,  
        "lambda_d": 1500,
        "eta_min1": 1e-16, 
        "eta_min2": 0.0, 
        "hidden_dims": tune.grid_search([
            # [2000, 800, 400, 200],
            #  [800, 2000, 400, 200],
            #  [800, 400, 2000, 200],
            #  [800, 400, 200, 2000]
            [800, 800, 800],
             ]),
        "activation": tune.grid_search(["ReLU", "SELU"]),
        "bottleneck_activation": tune.grid_search(["ReLU", "SELU", None]),
        "max_epochs": 1500,
        "T_max_ratio": 2/3,
        "warmup": 0,  
        "seed": tune.grid_search([0, 1, 2]),
        "batch_size": 100
    },
    "wine": {
        "teacher": "tsne",
        "t_n_neighbors": 15,
        "perplexity": 30,
        "learning_rate": 200,
        "lr": 0.003233466538536306,
        "lambda_d": 7000,
        "eta_min1": 6.076040435352558e-08,
        "eta_min2": 1.2312179372780289e-17,
        "hidden_dims": tune.grid_search([
            # [1500, 1000, 500, 250, 2000],
            # [2000, 1500, 1000, 500, 250],
            # [1500, 2000, 1000, 500, 250],
            # [1500, 1000, 2000, 500, 250],
            # [1500, 1000, 500, 2000, 250],
            # [200, 200, 200, 200, 200, 200],
            # [447, 447],
            [258, 258, 258, 258],
            # [169, 169, 169, 169, 169, 169, 169, 169]
        ]),
        "activation": tune.grid_search(["ReLU", "SELU"]),
        "bottleneck_activation": tune.grid_search(["ReLU", "SELU", None]),
        'max_epochs': 200000, 
        'T_max_ratio': 0.7,
        "warmup": 0, 
        "seed": tune.grid_search([0, 1, 2]),
        "batch_size": 100
    },
    "mnist": { # 10000
        "teacher": "tsne",
        "t_n_neighbors": 15,
        "perplexity": 30,
        "learning_rate": 200,
        "lr": 0.000269,	
        "lambda_d": 5000,
        "eta_min1": 7.256237e-10,
        "eta_min2": 1.587436e-16,
        "hidden_dims": tune.grid_search([
            # [50000, 10000, 5000, 2500, 1000],
            # [10000, 5000, 2500, 1000, 50000],
            # [10000, 50000, 5000, 2500, 1000],
            # [10000, 5000, 50000, 2500, 1000],
            # [10000, 5000, 2500, 50000, 1000],
            # [1323, 1323],
            [764, 764, 764, 764],
            # [592, 592, 592, 592, 592, 592],
            # [500, 500, 500, 500, 500, 500, 500, 500],
            # [441, 441, 441, 441, 441, 441, 441, 441, 441, 441]
        ]),
        "activation": tune.grid_search(["ReLU", "SELU"]),
        "bottleneck_activation": tune.grid_search(["ReLU", "SELU", None]),
        "max_epochs": 4000,
        "T_max_ratio":0.6,
        "batch_size": 256,
        "seed": tune.grid_search([0, 1, 2]),
        "warmup": 0, 
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
        "teacher": "isomap",
        "t_n_neighbors": 10,
        "perplexity": 30,
        "learning_rate": 200,
        "lr": 0.00020521674907073966,
        "lambda_d": 5000,
        "eta_min1": 3.0205023628787573e-06,
        "eta_min2": 1.4506322965991254e-09,
        "hidden_dims": tune.grid_search([
            # [35890, 3589, 1794, 897, 448],
            # [3589, 35890, 1794, 897, 448],
            # [3589, 1794, 35890, 897, 448],
            # [3589, 1794, 897, 35890, 448],
            # [3589, 1794, 897, 448, 17945],
            # [3589, 3589, 3589, 3589],
            [289,289,289,289,289,289,289],
            # [354,354,354,354,354],
            # [500, 500, 500]
        ]),
        "activation": tune.grid_search(["ReLU", "SELU"]),
        "bottleneck_activation": tune.grid_search(["ReLU", "SELU", None]),
        'max_epochs': 30000,
        'T_max_ratio': 0.7,
        "warmup": 0, 
        "seed": tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "batch_size": 10000
    }
}

def drd_trainable(config):
    """
    Trainable function for PBT optimization of DRD model.
    Reports distill_loss at regular intervals for PBT to use.
    """
    # Load and prepare data
    X_tr, X_te = load_and_split(dataset_name, seed=config['seed'], test_size=1)
    if config['teacher'] == "umap":
        Z_tr, _ = get_teacher_embeddings(
            config["teacher"], X_tr, 
            n_components=config["n_components"] if "n_components" in config else 2,
            n_neighbors=config["t_n_neighbors"], 
            # min_dist=config["min_dist"],
            random_state=config['seed'],
        )
    elif config['teacher'] == "pca":
        Z_tr, _ = get_teacher_embeddings(
            config["teacher"], X_tr, 
            n_components= config["n_components"] if "n_components" in config else 2,
            random_state=config['seed'],
        )
    elif config['teacher'] == "isomap":
        Z_tr, _ = get_teacher_embeddings(
            config["teacher"], X_tr, 
            n_components= config["n_components"] if "n_components" in config else 2,
            n_neighbors=config["t_n_neighbors"],
        )
    elif config['teacher'] == "tsne":
        Z_tr, _ = get_teacher_embeddings(
            config["teacher"], X_tr, 
            n_components=config["n_components"] if "n_components" in config else 2,
            perplexity=config["perplexity"],
            learning_rate=config["learning_rate"],
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
        "latent_dim": config["student_latent_dim"] if "student_latent_dim" in config else 2
    }
    
    # Create student model
    student = make_student(
        method="drd",
        input_dim=X_tr.shape[1],
        hidden_dims=config['hidden_dims'],
        device=DEVICE,
        **student_kwargs,
    )
    
    # Handle checkpointing (if resuming from checkpoint)
    # checkpoint = train.get_checkpoint()
    # print(f"checkpoint {checkpoint}")
    # start_epoch = 0
    # if checkpoint:
    #     with checkpoint.as_directory() as checkpoint_dir:
    #         checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
    #         with open(checkpoint_path, 'rb') as fp:
    #             checkpoint_dict = pickle.load(fp)
    #             student.model.load_state_dict(checkpoint_dict['model_state_dict'])
    #             student.opt_joint.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    #             start_epoch = checkpoint_dict.get('epoch', 0)
    
    # Custom training loop with periodic reporting
    X_tensor = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    Z_tensor = torch.tensor(Z_tr, dtype=torch.float32).to(DEVICE)
    dataset = TensorDataset(X_tensor, Z_tensor)
    loader = DataLoader(dataset, batch_size=student.batch_size, shuffle=True)
    
    student.model.train()
    report_interval = max(1, config['max_epochs'] // 100)  # Report 100 times during training
    
    for epoch in range(start_epoch, config['max_epochs']):
        epoch_distill_loss = 0.0
        epoch_recon_loss = 0.0
        num_batches = 0
        
        for batch in loader:
            x = batch[0].to(DEVICE, non_blocking=True)
            teacher_z = batch[1].to(DEVICE, non_blocking=True)
            
            student.opt_joint.zero_grad()
            x_rec, z = student.model(x)
            
            # Calculate losses
            recon_loss = student.criterion(x_rec, x)
            distill_loss = student.criterion(z, teacher_z)
            
            # Apply warmup to lambda_d
            if epoch < student.warmup_epochs:
                lambda_d = student.lambda_d * (epoch / student.warmup_epochs)
            else:
                lambda_d = student.lambda_d
            
            total_loss = recon_loss + lambda_d * distill_loss
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(student.model.parameters(), max_norm=student.clip_grad_norm)
            student.opt_joint.step()
            
            epoch_distill_loss += distill_loss.item()
            epoch_recon_loss += recon_loss.item()
            num_batches += 1
        
        # Update learning rate schedulers
        if epoch < student.T_max:
            student.scheduler1.step()
        else:
            if student.scheduler2 is None:
                student.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                    student.opt_joint, 
                    T_max=config['max_epochs'] - epoch, 
                    eta_min=student.eta_min2
                )
            student.scheduler2.step()
        
        # Report metrics at regular intervals and handle checkpointing
        checkpoint_freq = config.get('checkpoint_freq', 1e10)
        if (epoch + 1) % report_interval == 0 or epoch == config['max_epochs'] - 1:
            avg_distill_loss = epoch_distill_loss / num_batches
            avg_recon_loss = epoch_recon_loss / num_batches
            current_lr = student.opt_joint.param_groups[0]['lr']
            
            # Create checkpoint data
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': student.model.state_dict(),
                'optimizer_state_dict': student.opt_joint.state_dict(),
                'distill_loss': avg_distill_loss,
                'recon_loss': avg_recon_loss,
            }
            
            # Report with or without checkpoint based on frequency
            if (epoch + 1) % checkpoint_freq == 0 or epoch == config['max_epochs'] - 1:
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
                    with open(checkpoint_path, 'wb') as fp:
                        pickle.dump(checkpoint_data, fp)
                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    tune.report({'distill_loss': avg_distill_loss, 'recon_loss': avg_recon_loss}, checkpoint=checkpoint)
            else:
                # Report without checkpoint (for more frequent metric updates)
                tune.report({'distill_loss': avg_distill_loss, 'recon_loss': avg_recon_loss})

ahbs = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    metric="distill_loss", 
    mode="min",
    grace_period=100,
    max_t = 100,
)

# Run the experiment (simplified approach without RunConfig for compatibility)
analysis = tune.run(
    drd_trainable,
    name="drd_asynchyperband_distill_optimization",
    # scheduler=ahbs,
    num_samples=8, 
    resources_per_trial={"cpu": 6, "gpu": 0.5},  # Adjusted GPU allocation
    config= INIT_CONFIG[dataset_name],
    verbose=1,
    max_failures=3,
    storage_path="/shared/share_mala/irchang/drd/ray_results"
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


