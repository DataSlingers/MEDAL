# (442, 10)
# python eval/run_drd.py \
#   --dataset '["diabetes"]' \
#   --teacher_method '["pca"]' \
#   --hidden_layers '[442]' \
#   --teacher_kwargs '{"n_components" : 2,"random_state":0,  "save_teacher_model": true, "save_teacher_path": "/shared/share_mala/irchang/drd/models/teacher/db_t_pca_seed=0.pkl"}' \
#   --student_kwargs '{"epochs": 6000, "batch_size": 128, "lambda_reg": 0.0, "lr": 0.1, "warmup": 0, "eta_min1": 1e-4, "eta_min2": 1e-10, "T_max":3000, "latent_dim": 2, "save_model": true, "save_model_path": "/shared/share_mala/irchang/drd/models/db_t_pca_seed=0.pt"}' \
#   --var_name '{"student": "lambda_d"}' \
#   --var_values '{"student": [5000]}' \
#   --seeds '[0]' \
#   --optimize "joint" \
#   -o results/diabetes_pca_drd_joint_n=442 \
#   --device "cuda:3"
python eval/run_drd.py \
  --dataset '["diabetes"]' \
  --teacher_method '["umap"]' \
  --hidden_layers '[50000, 10000, 5000, 2500, 1000]' \
  --teacher_kwargs '{"n_components" : 2,"random_state":0, "n_neighbors":10}' \
  --student_kwargs '{"epochs": 100000, "batch_size": 512, "lambda_reg": 0.0, "lr": 0.0253689, "warmup": 0, "eta_min1": 2e-4, "eta_min2": 1.2637e-6, "T_max":50000, "latent_dim": 2, "activation": "SELU"}' \
  --var_name '{"student": "lambda_d"}' \
  --var_values '{"student": [5000]}' \
  --seeds '[0]' \
  --optimize "joint" \
  -o results/diabetes_umap10_drd_n=442 \
  --device "cuda:6"