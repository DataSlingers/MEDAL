python eval/run_drd.py \
  --dataset "macaque" \
  --teacher_method "tsne" \
  --hidden_layers '[700,700,700,700,700,700,700,700,700,700,700,700,700,700,700]' \
  --teacher_kwargs '{"random_state": 0, "n_components" : 2, "perplexity": 30, "learning_rate": "auto", "save_teacher_model": false, "save_teacher_path": null}' \
  --student_kwargs '{"epochs": 10000, "batch_size": 1024, "lr": 0.001, "lambda_d": 100000, "warmup": 0, "eta_min1":9.24882e-06, "eta_min2": 5e-7, "T_max": 7000, "save_model": true, "activation": "SELU", "bottleneck_activation": null, "save_model_path": "/shared/share_mala/irchang/drd/models/macaque_tsne30_s0.pt"}' \
  --seeds '[0]' \
  --device "cuda:4" \
  --verbose \
  --test_size 0.2
