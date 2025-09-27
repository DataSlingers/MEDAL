python eval/run_drd.py \
  --dataset "gene_cancer" \
  --teacher_method "tsne" \
  --hidden_layers '[1000, 1000, 1000, 1000]' \
  --teacher_kwargs '{"n_components" : 2, "perplexity": 794, "learning_rate": 50, "save_teacher_model": false}' \
  --student_kwargs '{"epochs": 6000, "batch_size": 100, "lr": 1e-3, "activation": "SELU", "bottleneck_activation": null, "T_max": 3000, "lambda_d": 15000, "warmup": 0, "eta_min1": 1e-05, "eta_min2": 0.0, "save_model": true, 
  "save_model_path": "/shared/share_mala/irchang/drd/models/pancan_tsne794_s0.pt"}' \
  --seeds '[0]' \
  --device "cuda:4" \
  --verbose