# experiment.py
import numpy as np, pandas as pd
if not hasattr(np, "product"):
    np.product = np.prod
from pathlib import Path
import os
import argparse, json

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dataclasses import dataclass, field
from typing import Any, Dict, List
from eval_utils import load_and_split, get_teacher_embeddings, make_student, fit_student, eval_student

@dataclass
class ExperimentConfig:
    dataset: str
    teacher_method: str
    teacher_kwargs: Dict[str, Any]
    student_kwargs: Dict[str, Any]
    hidden_exponents: List[int]    
    var_name: str                   
    var_values: List[Any]
    constrained: bool
    symmetric: bool
    optimize: str                   # "joint"/"encoder"/"decoder"
    seeds: List[int] = field(default_factory=lambda: list(range(10)))


def run_single_config(cfg: ExperimentConfig):
    results = []
    for seed in cfg.seeds:
        # 1) load data
        X_tr, X_te = load_and_split(cfg.dataset, seed=seed)

        # 2) teacher embeddings
        Z_tr, Z_te = get_teacher_embeddings(
            cfg.teacher_method, X_tr, X_te, **cfg.teacher_kwargs
        )

        # 3) sweep over your var_name & hidden-dim settings
        for val in cfg.var_values:
            setattr(cfg, cfg.var_name, val)
            for idx, depth in enumerate(cfg.hidden_exponents):
                hidden_dims = tuple(2**i for i in cfg.hidden_exponents[:idx+1])
                
                student = make_student(
                    input_dim=X_tr.shape[1],
                    hidden_dims=hidden_dims,
                    # latent_dim=cfg.bottleneck_dim,
                    # constrained=cfg.constrained,
                    # symmetric=cfg.symmetric,
                    **{cfg.var_name: val}
                )
                # 4) train & eval
                fit_student(student, X_tr, Z_tr, optimize=cfg.optimize)
                train_metrics = eval_student(student, X_tr, Z_tr)
                test_metrics  = eval_student(student, X_te, Z_te)

                # 5) record
                results.append({
                    "dataset": cfg.dataset,
                    "teacher_method": cfg.teacher_method,
                    "seed": seed,
                    cfg.var_name: val,
                    "depth": idx+1,
                    "distill_drd_train":    train_metrics["distill_mse"],
                    "recon_drd_train":      train_metrics["recon_mse"],
                    "distill_drd_test":     test_metrics["distill_mse"],
                    "recon_drd_test":       test_metrics["recon_mse"],
                    "constrained": cfg.constrained,
                    "symmetric":   cfg.symmetric,
                    "optimize":    cfg.optimize,
                })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DRD simulation on selected dataset")
    # experiment setup
    parser.add_argument("--dataset", type=json.loads, default='["wine"]', help="Dataset to run the simulation on")
    parser.add_argument("--teacher_method", type=json.loads, default='["umap"]', help="Method to use for teacher embeddings (e.g., 'umap', 'pca')")
    parser.add_argument("--seeds", type=json.loads, default="[0,1,2,3,4,5,6,7,8,9]", help="List of random seeds for reproducibility")

    # teacher model config
    parser.add_argument("--teacher_kwargs", type=json.loads, default='{"n_components": 2, "random_state": 0}', help="Hyperparameters for the teacher method")

    # student model config
    parser.add_argument("--student_kwargs", type=json.loads, default='{"latent_dim": 2, "epochs": 30, "batch_size": 256, "lambda_d": 1.0, "lambda_reg": 0.0, "lr": 1e-3, "clip_grad_norm": 1.0}', help="Hyperparameters for the student model")
    parser.add_argument("--hidden_exponents", type=json.loads, default="[4, 5, 6]", help="List of hidden layer sizes as powers of 2 (e.g., [4, 5, 6] means [16, 32, 64])")
    parser.add_argument("--constrained", action="store_true", help="Whether to use a constrained bottleneck")
    parser.add_argument("--symmetric", action="store_true", help="Whether to use a symmetric architecture")
    parser.add_argument("--optimize", type=str, default="joint", choices=["joint", "encoder", "decoder"], help="Optimization strategy for the student model")

    # variable config
    parser.add_argument("--var_name", type=str, default="lambda_d", help="Variable name to vary")
    parser.add_argument("--var_values", type=json.loads, help="Values for the variable")

    parser.add_argument("--device", type=str, default=None, help="Device to run the model on (e.g., 'cuda', 'cpu')")
    args = parser.parse_args()

    config_list = []
    for dataset in args.dataset:
        for method in args.teacher_method:
            print(f"Running simulation for dataset={dataset}, teacher_method={method}")
            config = ExperimentConfig(
                dataset=dataset,
                teacher_method=method,
                teacher_kwargs=args.teacher_kwargs,
                student_kwargs=args.student_kwargs,
                seeds=args.seeds,
                hidden_exponents=args.hidden_exponents,  
                var_name=args.var_name,      
                var_values=args.var_values, 
                constrained=args.constrained,
                symmetric=args.symmetric,
                optimize=args.optimize,
            )
            config_list.append(config)
    
    all_results = []
    for cfg in config_list:
        all_results += run_single_config(cfg)
    
    print(pd.DataFrame(all_results))

    # all_results = run_simulation(args.var_name, args.var_values, param_dict)
    # Path("results").mkdir(exist_ok=True)
    # df = pd.DataFrame(all_results)
    # df.to_csv(f"results/drd_losses_{args.var_name}_size&depth.csv", index=False)
