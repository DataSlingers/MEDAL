# (178, 13)
python eval/run_drd.py \
  --dataset '["wine"]' \
  --teacher_method '["pca"]' \
  --hidden_layers '[2000, 1500, 1000, 500, 250]' \
  --teacher_kwargs '{"n_components" : 2, "random_state": 0, "save_teacher_model": true, "save_teacher_path": "/shared/share_mala/irchang/drd/models/teacher/wi_t_pca_seed=0.pkl"}' \
  --student_kwargs '{"epochs": 130000, "batch_size": 100, "lambda_reg": 0.0, "lr": 0.003233466538536306, "warmup": 0, "eta_min1": 6.076040435352558e-08, "eta_min2": 1.2312179372780289e-17, "T_max": 91000}' \
  --var_name '{"student": "lambda_d"}' \
  --var_values '{"student": [5000]}' \
  --seeds '[0]' \
  --optimize "joint" \
  --device "cuda:1" \
  --split_test_size 1 \
  --verbose
  # -o /shared/share_mala/irchang/drd/results/wine_umap_neigh10_depth5 \
