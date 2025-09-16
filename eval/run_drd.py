# experiment.py
import numpy as np, pandas as pd
if not hasattr(np, "product"):
    np.product = np.prod
from pathlib import Path
import os
import argparse, json
import pprint
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dataclasses import dataclass, field
from typing import Any, Dict, List
from utils.eval_utils import load_and_split, get_teacher_embeddings, make_student, eval_student, eval_pca_baseline

DISTILL_BANDS_DICT = {
    "gene_cancer": [(1e-12, 9e-9)],
    "mnist": [(1e-12, 9e-7)],
    "single_cell": [(1e-12, 9e-7)],
    "wine": [(1e-12, 9e-8)],
}

@dataclass
class ExperimentConfig:
    dataset: str
    student_method: str 
    teacher_method: str = "None"
    teacher_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    student_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    hidden_layers: List[int] = field(default_factory=lambda: [4, 5, 6])
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    device: str = "cpu"
    verbose: bool = False

def run_single_config(cfg: ExperimentConfig):
    if cfg.student_method == "drd":
        return run_drd_config(cfg)
    elif cfg.student_method == "pca":
        return run_pca_config(cfg)

def run_drd_config(cfg: ExperimentConfig):
    results = []
    device = torch.device(cfg.student_kwargs.get("device", cfg.device) or("cuda" if torch.cuda.is_available() else "cpu"))
    for seed in cfg.seeds:
        # 1) load data
        X_tr, X_te = load_and_split(cfg.dataset, seed=seed)
        tc = cfg.teacher_kwargs

        # 2) teacher embeddings
        if cfg.teacher_method == "umap":
            Z_tr, _ = get_teacher_embeddings(
                cfg.teacher_method, X_tr, 
                n_components= tc["n_components"] if "n_components" in tc else 2,
                n_neighbors=tc["n_neighbors"], 
                min_dist=tc["min_dist"],
                random_state=seed,
            )
        elif cfg.teacher_method == "pca":
            Z_tr, _ = get_teacher_embeddings(
                cfg.teacher_method, X_tr, 
                n_components= tc["n_components"] if "n_components" in tc else 2,
                random_state=seed,
            )
        elif cfg.teacher_method == "isomap":
            Z_tr, _ = get_teacher_embeddings(
                cfg.teacher_method, X_tr, 
                n_components= tc["n_components"] if "n_components" in tc else 2,
                n_neighbors=tc["n_neighbors"],
            )
        elif cfg.teacher_method == "tsne":
            Z_tr, _ = get_teacher_embeddings(
                cfg.teacher_method, X_tr, 
                n_components= tc["n_components"] if "n_components" in tc else 2,
                perplexity=tc["perplexity"],
                learning_rate=tc["learning_rate"],
            )
           
        student = make_student(
            method="drd",
            input_dim=X_tr.shape[1],
            latent_dim = tc["n_components"] if "n_components" in tc else 2,
            hidden_dims= cfg.hidden_layers,
            device=device,
            **cfg.student_kwargs,
        )
        # 4) train & eval
        distill_bands = DISTILL_BANDS_DICT[cfg.dataset]
        student.fit(X_tr, Z_tr, cfg.verbose, 
                phase="finetune", 
                target_bands=distill_bands, 
                stability_window=10, 
                epsilon_distill=1e-7, epsilon_recon=1e-3, 
                patience=20, # unit = epoch
                return_on_stable=True,
                # checkpointing
                save_dir = cfg.student_kwargs.get("save_model_path") if cfg.student_kwargs.get("save_model", False) else None, 
                print_tag=True,
                )
        train_metrics = eval_student(student, X_tr, Z_tr)
        # test_metrics  = eval_student(student, X_te, Z_te)

        # 5) record
        res = dict({
            "dataset": cfg.dataset,
            "teacher_method": cfg.teacher_method,
            "student_method": cfg.student_method,
            "seed": seed,
            "hidden_layers": cfg.hidden_layers,
            "distill_train":    train_metrics["distill_mse"],
            "recon_train":      train_metrics["recon_mse"],
            # "distill_test":     test_metrics["distill_mse"],
            # "recon_test":       test_metrics["recon_mse"],
        })
        results.append(res)

    return results

def run_pca_config(cfg: ExperimentConfig):
    results = []
    for seed in cfg.seeds:
        X_tr, X_te = load_and_split(cfg.dataset, seed=seed)
        # PCA comparison
        pca_baseline = make_student(
            method = "pca",
            **{"n_components": 2, # bottleneck is fixed at 2
                "random_state": seed}
        )
        pca_metrics = eval_pca_baseline(pca_baseline, X_tr, X_te)

        results.append({
            "dataset": cfg.dataset,
            "student_method": cfg.student_method,
            "seed": seed,
            "depth": np.nan,
            "distill_train":    np.nan,
            "recon_train":      pca_metrics["recon_train_mse"],
            "distill_test":     np.nan,
            "recon_test":        pca_metrics["recon_test_mse"],
        })

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DRD simulation on selected dataset")
    # experiment setup
    parser.add_argument("--dataset", type=str, default='wine', help="Dataset to run the simulation on")
    parser.add_argument("--teacher_method", type=str, default='umap', help="Method to use for teacher embeddings (e.g., 'umap', 'pca')")
    parser.add_argument("--seeds", type=json.loads, default="[0,1,2,3,4,5,6,7,8,9]", help="List of random seeds for reproducibility")

    # teacher model config
    parser.add_argument("--teacher_kwargs", type=json.loads, default='{"random_state": 0}', help="Hyperparameters for the teacher method")

    # student model config
    parser.add_argument("--student_kwargs", type=json.loads, default='{"epochs": 30, "batch_size": 256, "lambda_reg": 0.0, "lr": 1e-3, "clip_grad_norm": 1.0}', help="Hyperparameters for the student model")
    parser.add_argument("--hidden_layers", type=json.loads, default="[4, 5, 6]", help="List of hidden layer sizes as powers of 2 (e.g., [4, 5, 6] means [16, 32, 64])")

    parser.add_argument("--device", type=str, default=None, help="Device to run the model on (e.g., 'cuda', 'cpu')")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument('-o', "--output_filename", type=str, default=None)
    args = parser.parse_args()

    config_list = []

    teacher_kwargs_cp = args.teacher_kwargs.copy()
    student_kwargs_cp = args.student_kwargs.copy()

    config = ExperimentConfig(
        dataset=args.dataset,
        teacher_method=args.teacher_method,
        student_method="drd",
        teacher_kwargs=teacher_kwargs_cp,
        student_kwargs=student_kwargs_cp,
        seeds=args.seeds,
        hidden_layers=args.hidden_layers,  
        device=args.device,
        verbose=args.verbose,
    )
    config_list.append(config)
    # Add PCA baseline config
    # pca_config = ExperimentConfig(
    #     dataset=dataset,
    #     student_method="pca",
    #     seeds=args.seeds,
    # )
    # config_list.append(pca_config)
    
    print(f"Running {len(config_list)} configurations...")
    all_results = []
    for cfg in config_list:
        print("......Running config:")
        pprint.pprint(cfg)
        all_results += run_single_config(cfg)

    df = pd.DataFrame(all_results)
    print(df)

    if args.output_filename is None:
        print("Results not saved, please provide an output filename")
    else:
        df.to_csv(f'{args.output_filename}.csv', index=False)
        json.dump(args.__dict__, open(f'{args.output_filename}.json', 'w'), indent=2)

