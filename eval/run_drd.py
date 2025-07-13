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
from eval_utils import load_and_split, get_teacher_embeddings, make_student, fit_student, eval_student, eval_pca_baseline

@dataclass
class ExperimentConfig:
    dataset: str
    student_method: str 
    teacher_method: str = "None"
    teacher_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    student_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    hidden_layers: List[int] = field(default_factory=lambda: [4, 5, 6])
    student_var_name: str  = "None"                 
    student_var_value: int = np.nan
    teacher_var_name: str   = "None"                      
    teacher_var_value: int = np.nan
    constrained: bool = False
    symmetric: bool = False
    optimize: str = "None"        # "joint"/"encoder"/"decoder"        
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

        # 2) teacher embeddings
        Z_tr, Z_te = get_teacher_embeddings(
            cfg.teacher_method, X_tr, X_te, 
            **cfg.teacher_kwargs
        )
           
        student = make_student(
            method = "drd",
            input_dim=X_tr.shape[1],
            hidden_dims=cfg.hidden_layers,
            constrained=cfg.constrained,
            symmetric=cfg.symmetric,
            device=device,
            **cfg.student_kwargs,
        )
        # 4) train & eval
        fit_student(student, X_tr, Z_tr, optimize=cfg.optimize, verbose=cfg.verbose)
        train_metrics = eval_student(student, X_tr, Z_tr)
        test_metrics  = eval_student(student, X_te, Z_te)
        print(f'distill_train {train_metrics["distill_mse"]}, recon_train {train_metrics["recon_mse"]}')
        if cfg.student_kwargs.get("save_model", False):
            try:
                model_path = Path(cfg.student_kwargs.get("save_model_path"))
            except:
                raise RuntimeError("Please provide a valid path for saving the model")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(student.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        # 5) record
        res = dict({
            "dataset": cfg.dataset,
            "teacher_method": cfg.teacher_method,
            "student_method": cfg.student_method,
            "seed": seed,
            "hidden_layers": cfg.hidden_layers,
            "distill_train":    train_metrics["distill_mse"],
            "recon_train":      train_metrics["recon_mse"],
            "distill_test":     test_metrics["distill_mse"],
            "recon_test":       test_metrics["recon_mse"],
            "constrained": cfg.constrained,
            "symmetric":   cfg.symmetric,
            "optimize":    cfg.optimize,
        })
        if cfg.student_var_name is not None:
            res[cfg.student_var_name] = cfg.student_var_value
        if cfg.teacher_var_name is not None:
            res[cfg.teacher_var_name] = cfg.teacher_var_value
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
    parser.add_argument("--teacher_kwargs", type=json.loads, default='{"random_state": 0}', help="Hyperparameters for the teacher method")

    # student model config
    parser.add_argument("--student_kwargs", type=json.loads, default='{"epochs": 30, "batch_size": 256, "lambda_reg": 0.0, "lr": 1e-3, "clip_grad_norm": 1.0}', help="Hyperparameters for the student model")
    parser.add_argument("--hidden_layers", type=json.loads, default="[4, 5, 6]", help="List of hidden layer sizes as powers of 2 (e.g., [4, 5, 6] means [16, 32, 64])")
    parser.add_argument("--constrained", action="store_true", help="Whether to use a constrained bottleneck")
    parser.add_argument("--symmetric", action="store_true", help="Whether to use a symmetric architecture")
    parser.add_argument("--optimize", nargs="+", default=["sep_opt","sep_freeze",  "joint"], help="Optimization strategies for the student model")

    # variable config
    parser.add_argument("--var_name", type=json.loads, default='{"student": "lambda_d", "teacher":"n_components"}', help="Variable name to vary")
    parser.add_argument("--var_values", type=json.loads, help="Values for the variable")

    parser.add_argument("--device", type=str, default=None, help="Device to run the model on (e.g., 'cuda', 'cpu')")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument('-o', "--output_filename", type=str, default=None)
    args = parser.parse_args()

    config_list = []
    student_var =  args.var_name.get("student")
    teacher_var =  args.var_name.get("teacher")
    # If there's no student_var, we’ll iterate exactly once with s_val=None
    student_vals = args.var_values.get("student", []) if student_var else [None]
    # If there's no teacher_var, we’ll iterate exactly once with t_val=None
    teacher_vals = args.var_values.get("teacher", []) if teacher_var else [None]
    for dataset in args.dataset:
        for t_method in args.teacher_method:
            for t_val in teacher_vals:
                for s_val in student_vals:
                    for opt in args.optimize:
                        teacher_kwargs_cp = args.teacher_kwargs.copy()
                        student_kwargs_cp = args.student_kwargs.copy()
                        if teacher_var is not None:
                            teacher_kwargs_cp[teacher_var] = t_val
                        if student_var is not None:
                            student_kwargs_cp[student_var] = s_val

                        student_kwargs_cp["update_mode"] = opt

                        config = ExperimentConfig(
                            dataset=dataset,
                            teacher_method=t_method,
                            student_method="drd",
                            teacher_kwargs=teacher_kwargs_cp,
                            student_kwargs=student_kwargs_cp,
                            seeds=args.seeds,
                            hidden_layers=args.hidden_layers,  
                            student_var_name=student_var,   
                            student_var_value= s_val,
                            teacher_var_name=teacher_var,
                            teacher_var_value=t_val, 
                            constrained=args.constrained,
                            symmetric=args.symmetric,
                            optimize=opt,
                            device=args.device,
                            verbose=args.verbose,
                        )
                        config_list.append(config)
        # Add PCA baseline config
        pca_config = ExperimentConfig(
            dataset=dataset,
            student_method="pca",
            seeds=args.seeds,
        )
        config_list.append(pca_config)
    
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

