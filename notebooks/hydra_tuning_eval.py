from structure_preservation_metrics import evaluate_embedding
from medal.eval_utils import load_and_split, get_teacher_embeddings
from sklearn.model_selection import train_test_split
from medal.core import AutoEncoder
import numpy as np, pandas as pd
from pathlib import Path
import torch

PATH_PREFIX = '/share/ctn/users/bnc2119'

X_og, X_test, y_og, y_test = load_and_split("hydra", test_size=0.2, seed=0, labels=True)
X, X_val, y, y_val = train_test_split(X_og, y_og, test_size = 0.2, random_state = 0)
seed_list = [2]
scdeed_best = {
    "tsne": {0: 4999},
    "umap": {2: 2000}
}
embedr_best = {
    "tsne": {0: 4999,  },
    "umap": {2: 271, }
}
pcs_best = {
    "tsne": {0: 5, },
}
medal_best = {
    "tsne": {0: 1077,},
    "umap": {2: 5,  }
}
def load_trained_ae(ckpt_path, input_dim, hidden_dims,activation=torch.nn.SELU, latent_dim=2, batchnorm=False):
    import sys, os
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    model = AutoEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,            
        bottleneck_activation=None,
        use_batchnorm = batchnorm
    )
    sys.stdout = old_stdout

    sd = torch.load(ckpt_path, map_location="cpu")

    # unwrap to the real state dict
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # remove DDP prefixes if any
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # load STRICTLY so we fail fast if something doesn’t match
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded params: {len(sd)} | missing={len(missing)} | unexpected={len(unexpected)}")
    if missing or unexpected:
        print("⚠️ Check architecture/keys. Example missing:", missing[:3], "unexpected:", unexpected[:3])

    model.eval()
    return model


teacher_embed = {}
for s in [0]:
    teacher_embed[f'scdeed'] = np.load(Path(PATH_PREFIX) / f"MEDAL/comparisons/data/hydra_train_tsne_{scdeed_best['tsne'][s]}_0_train_pc4.npy")
    teacher_embed[f'embedr'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/hydra_tsne_{embedr_best['tsne'][s]}_0_train_embedr.npy").squeeze(0)
    teacher_embed[f'pcs'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/hydra_tsne_{pcs_best['tsne'][s]}_0_train_pcs.npy")

X_scdeed = pd.read_csv(Path(PATH_PREFIX) / f"MEDAL/comparisons/data/hydra_train_pc4.csv", index_col=0).to_numpy()

by_k_tsne = pd.DataFrame([])
for s in [0]:
    print("Eval MEDAL")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = get_teacher_embeddings("tsne", X_og, perplexity = medal_best["tsne"][s], learning_rate="auto", random_state=s),
        k_values = range(10, 201, 20)
    )
    tw_df.update({"method": f'MEDAL ({medal_best["tsne"][s]})'})
    by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
    print("Eval EMBEDR")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = teacher_embed[f"embedr"],
        k_values = range(10, 201, 20)
    )
    tw_df.update({"method": f'EMBEDR ({embedr_best["tsne"][s]})'})
    by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
    print("Eval PCS")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = teacher_embed[f"pcs"],
        k_values = range(10, 201, 20)
    )
    tw_df.update({"method": f'PCS ({pcs_best["tsne"][s]})'})
    by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
    print("Eval SCDEED")
    tw_df = evaluate_embedding(
        X_high = X_scdeed,
        X_low = teacher_embed[f"scdeed"],
        k_values = range(10, 201, 20)
    )
    tw_df.update({"method": f'SCDEED ({scdeed_best["tsne"][s]})'})
    by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
by_k_tsne.to_csv('hydra_by_k_tsne_all_methods.csv')


# by_k_tsne = pd.DataFrame([])
# for perp in np.unique(np.logspace(np.log10(5), np.log10(5000), 10).astype(int)):
#     teacher_embed = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/hydra_tsne_{perp}_0_train_embedr.npy").squeeze(0)
#     tw_df = evaluate_embedding(
#         X_high = X,
#         X_low = teacher_embed,
#         k_values = range(10, 201, 20)
#     )
#     tw_df.update({"method": f'MEDAL_{perp}', "split": "Train"})
#     by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
    
#     student = load_trained_ae(Path(PATH_PREFIX) / f"drd_data/tmp_results/compare_embedr/hydra/medal_tsne2_{perp}_tc0_0_ckpts/final.pt",
#                 input_dim=500,
#                 hidden_dims=[256, 1024, 1024, 1024],
#                 latent_dim=2, batchnorm=True)
        
#     student_recon, student_embed = student(torch.tensor(X, dtype=torch.float32))
#     train_distill = np.mean((teacher_embed - student_embed.detach().numpy()) ** 2)
#     if train_distill < 1e-5:
#         _, x_test_embed = student(torch.tensor(X_test, dtype=torch.float32))
#         tw_df = evaluate_embedding(
#             X_high = X_test,
#             X_low = x_test_embed.detach().numpy(),
#             k_values = range(10, 201, 20)
#         )
#         tw_df.update({"method": f'MEDAL_{perp}', "split": "Test"})
#         by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
            
# by_k_tsne.reset_index(names='k', inplace=True)
# by_k_tsne.to_csv('hydra_by_k_tsne_embedr.csv')


# UMAP

teacher_embed = {}

teacher_embed[f'scdeed'] = np.load(Path(PATH_PREFIX) / f"MEDAL/comparisons/data/hydra_train_umap_2000_0.1_0_train_pc4.npy")
# teacher_embed[f'medal'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings2/hydra_umap_5_0.1_0_train.npy")
teacher_embed[f'embedr'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/hydra_umap_271_0.1_2_train_embedr.npy").squeeze(0)

by_k_umap = pd.DataFrame([])
for s in seed_list:
    print("Eval MEDAL")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = get_teacher_embeddings("umap", X_og, n_neighbors = medal_best["umap"][s], min_dist = 0.1, random_state=s),
        k_values = range(10, 201, 20)
    )
    tw_df.update({"method": f'MEDAL (5)'})
    by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
    
    print("Eval EMBEDR")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = teacher_embed[f"embedr"],
        k_values = range(10, 201, 20)
    )
    tw_df.update({"method": f'EMBEDR (271)'})
    by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
    
    print("Eval SCDEED")
    tw_df = evaluate_embedding(
        X_high = X_scdeed,
        X_low = teacher_embed[f"scdeed"],
        k_values = range(10, 201, 20)
    )
    tw_df.update({"method": f'SCDEED (2000)'})
    by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
    
by_k_umap.to_csv('hydra_by_k_umap_all_methods.csv')


# by_k_umap = pd.DataFrame([])
# for n in np.unique(np.logspace(np.log10(5), np.log10(2000), 10).astype(int)):
#     teacher_embed = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/hydra_umap_{n}_0.1_2_train_embedr.npy").squeeze(0)
#     tw_df = evaluate_embedding(
#         X_high = X,
#         X_low = teacher_embed,
#         k_values = range(10, 201, 20)
#     )
#     tw_df.update({"method": f'MEDAL_{n}', "split": "Train"})
#     by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
    
    
#     student = load_trained_ae(Path(PATH_PREFIX) / f"drd_data/tmp_results/compare_embedr/hydra/medal_umap2_{n}_0.1_tc2_0_ckpts/final.pt",
#                 input_dim=500,
#                 hidden_dims=[256, 1024, 1024, 1024],
#                 latent_dim=2, batchnorm=True)
        
#     student_recon, student_embed = student(torch.tensor(X, dtype=torch.float32))
#     train_distill = np.mean((teacher_embed - student_embed.detach().numpy()) ** 2)
#     if train_distill < 1e-5:
#         _, x_test_embed = student(torch.tensor(X_test, dtype=torch.float32))
#         tw_df = evaluate_embedding(
#             X_high = X_test,
#             X_low = x_test_embed.detach().numpy(),
#             k_values = range(10, 201, 20)
#         )
#         tw_df.update({"method": f'MEDAL_{n}', "split": "Test"})
#         by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
            
# by_k_umap.reset_index(names='k', inplace=True)
# by_k_umap.to_csv('hydra_by_k_umap_embedr.csv')