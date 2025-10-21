
python eval/run_drd.py \
  --dataset "mnist" \
  --teacher_method "umap" \
  --hidden_layers '[1000, 1000, 1000, 1000, 1000]' \
  --teacher_kwargs '{"n_components" : 2, "n_neighbors": 6, "min_dist":0.1, "save_teacher_model": false, "save_teacher_path": null}' \
  --student_kwargs '{"epochs": 6000, "batch_size": 256, "lr": 0.000269, "lambda_d": 10000, "warmup": 0, "eta_min1":1e-8, "eta_min2": 1.587436e-16, "T_max": 2100, "save_model": true, "activation": "SELU", "bottleneck_activation": null, "save_model_path": "/shared/share_mala/irchang/drd/results/chkpt/mnist"}' \
  --seeds '[0,1,2,3,4]' \
  --device "cuda:1" \
  --test_size 0
