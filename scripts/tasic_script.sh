python eval/run_drd.py \
  --dataset "cortical" \
  --teacher_method "tsne" \
  --hidden_layers '[309, 1792, 1792, 1792]' \
  --teacher_kwargs '{"random_state": 0, "n_components" : 2, "perplexity": 28, "learning_rate": "auto", "save_teacher_model": false, "save_teacher_path": null}' \
  --student_kwargs '{"epochs": 5000, "batch_size": 51200, "lr": 0.00268681, "lambda_d": 30000, "warmup": 50, "eta_min1": 7.936e-05, "eta_min2": 1.61441e-07, "T_max": 3500, "save_model": true, "activation": "SELU", "bottleneck_activation": null, "save_model_path": "/shared/share_mala/irchang/drd/models/tasic_tsne28_s0.pt"}' \
  --seeds '[0]' \
  --device "cuda:2" \
  --verbose 
