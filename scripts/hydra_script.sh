python eval/run_drd.py \
  --dataset "hydra" \
  --teacher_method "tsne" \
  --hidden_layers '[309, 1792, 1792, 1792]' \
  --teacher_kwargs '{"random_state": 0, "n_components" : 2, "perplexity": 40, "learning_rate": "auto", "save_teacher_model": false, "save_teacher_path": null}' \
  --student_kwargs '{"epochs": 20000, "batch_size": 51200, "lr": 0.000140849 , "lambda_d": 20000, "warmup": 1500, "eta_min1": 9.10708e-06, "eta_min2": 8.51602e-10, "T_max": 14000, "save_model": true, "activation": "SELU", "bottleneck_activation": null, "save_model_path": "/shared/share_mala/irchang/drd/models/hydra_tsne40_s0.pt", "use_lbfgs": false}' \
  --seeds '[0]' \
  --device "cuda:2" \
  --verbose