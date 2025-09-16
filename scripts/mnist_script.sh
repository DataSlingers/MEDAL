
# PYTHONPATH=
python eval/run_drd.py \
  --dataset '["mnist"]' \
  --teacher_method '["pca"]' \
  --hidden_layers '[1000, 1000, 1000, 1000, 1000]' \
  --teacher_kwargs '{"random_state": 0, "n_components" : 300, "save_teacher_model": false, "save_teacher_path": "/shared/share_mala/irchang/drd/models/teacher/mn_t_umap_neigh=10_seed=0.pkl"}' \
  --student_kwargs '{"epochs": 3500, "batch_size": 256, "lr": 0.000269, "lambda_d": 10000, "warmup": 0, "eta_min1":1e-8, "eta_min2": 1.587436e-16, "T_max": 2100, "save_model": true, "activation": "SELU", "bottleneck_activation": null, "save_model_path": "/shared/share_mala/irchang/drd/models/mnist_pca2.pt"}' \
  --seeds '[0]' \
  --device "cuda:1" 
  # --o "/shared/share_mala/irchang/drd/results/mnist_pca2.pt"``