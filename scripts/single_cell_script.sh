# (3589, 23368)
python eval/run_drd.py \
  --dataset "single_cell" \
  --teacher_method "tsne" \
  --hidden_layers '[294, 294, 294, 294, 294, 294, 294, 294, 294]' \
  --teacher_kwargs '{"random_state": 0, "n_components" : 2, "perplexity": 3587, "learning_rate": "auto", "save_teacher_model": false, "save_teacher_path": "/shared/share_mala/irchang/drd/models/teachers/single_cell_tsne3587_s0.pt"}' \
  --student_kwargs '{"epochs": 30000, "batch_size": 10000, "lr": 0.002, "lambda_d": 30000, "warmup": 100, "eta_min1": 3.925429581596387e-05, "eta_min2": 4.3892424083623014e-07, "T_max": 18000, "save_model": true, "activation": "SELU", "bottleneck_activation": null, "save_model_path": "/shared/share_mala/irchang/drd/models/single_cell_tsne3587_s0.pt"}' \
  --seeds '[0]' \
  --device "cuda:7"\
  --verbose