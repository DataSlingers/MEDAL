python eval/run_drd.py \
  --dataset "pbmc" \
  --teacher_method "umap" \
  --hidden_layers '[500, 500, 500, 500, 500]' \
  --teacher_kwargs '{"random_state": 0, "n_components" : 2, "n_neighbors": 160, "min_dist": 0.9, "save_teacher_model": false, "save_teacher_path": null}' \
  --student_kwargs '{"epochs": 20000, "batch_size": 10000, "lr": 0.001, "lambda_d": 50000, "warmup": 3000, "eta_min1": 3.4443667740771654e-05, "eta_min2": 1.069202395077256e-07, "T_max": 14000, "activation": "SELU", "bottleneck_activation": null, "use_lbfgs": false, "save_model": true, "save_model_path": "/shared/share_mala/irchang/drd/models/pbmc_umap160_0.9_s0.pt"}' \
  --seeds '[0]' \
  --device "cuda:5" \
  --verbose