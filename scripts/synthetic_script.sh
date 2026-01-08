python eval/run_drd.py \
  --dataset "synthetic" \
  --teacher_method "tsne" \
  --hidden_layers '[700,700,700,700,700,700,700,700]' \
  --teacher_kwargs '{"n_components" : 2, "perplexity": 30, "learning_rate": "auto", "save_teacher_model": false, "save_teacher_path": null}' \
  --student_kwargs '{"epochs": 10000, "batch_size": 1024, "lr": 7.54e-03, "lambda_d": 100000, "warmup": 0, "eta_min1":1.38811e-05, "eta_min2": 1.75169e-06, "T_max": 7000, "save_model": true, "activation": "SELU", "bottleneck_activation": null, "save_model_path": "/shared/share_mala/irchang/drd/results/chkpt/synthetic"}' \
  --seeds '[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]' \
  --device "cuda:4" \
  --verbose \
  --test_size 0
