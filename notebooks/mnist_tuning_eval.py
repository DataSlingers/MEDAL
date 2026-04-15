from structure_preservation_metrics import evaluate_embedding
from medal.eval_utils import load_and_split, get_teacher_embeddings
from sklearn.model_selection import train_test_split
from medal.core import AutoEncoder
import numpy as np, pandas as pd
from pathlib import Path
import torch

PATH_PREFIX = '/share/ctn/users/bnc2119'

X_og, X_test, y_og, y_test = load_and_split("mnist", test_size=0.2, seed=0, labels=True)
X, X_val, y, y_val = train_test_split(X_og, y_og, test_size = 0.2, random_state = 0)
seed_list = [0]
scdeed_best = {
    "tsne": {0: 341, 2: 1846, 10: 793},
    "umap": {0: 499, 2: 5,    10: 5}
}
embedr_best = {
    "tsne": {0: 5,   2: 5,   10: 5},
    "umap": {0: 499, 2: 499, 10: 499}
}
pcs_best = {
    "tsne": {0: 5,   2: 1846,   10: 793},
}
medal_best = {
    "tsne": {0: 793, 2: 793, 10: 793},
    "umap": {0: 49,  2: 18,  10: 18}
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
for s in seed_list:
    teacher_embed[f'scdeed_{s}'] = np.load(Path(PATH_PREFIX) / f"MEDAL/comparisons/data/mnist_train_tsne_{scdeed_best['tsne'][s]}_{s}_train_pc6.npy")
    teacher_embed[f'embedr_{s}'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/mnist_tsne_{embedr_best['tsne'][s]}_{s}_train_embedr.npy").squeeze(0)
    teacher_embed[f'pcs_{s}'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/mnist_tsne_{pcs_best['tsne'][s]}_{s}_train_pcs.npy")
    
X_scdeed = pd.read_csv(Path(PATH_PREFIX) / f"MEDAL/comparisons/data/mnist_train_pc6.csv", index_col=0).to_numpy()

by_k_tsne = pd.DataFrame([])
for s in seed_list:
    print("Eval MEDAL")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = get_teacher_embeddings("tsne", X_og, perplexity = medal_best["tsne"][s], learning_rate="auto", random_state=s),
        k_values = range(5, 51, 5)
    )
    tw_df.update({"method": f'MEDAL ({medal_best["tsne"][s]})'})
    by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
    print("Eval EMBEDR")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = teacher_embed[f"embedr_{s}"],
        k_values = range(5, 51, 5)
    )
    tw_df.update({"method": f'EMBEDR_SEED{s} ({embedr_best["tsne"][s]})'})
    by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
    print("Eval PCS")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = teacher_embed[f"pcs_{s}"],
        k_values = range(5, 51, 5)
    )
    tw_df.update({"method": f'PCS_SEED{s} ({pcs_best["tsne"][s]})'})
    by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
    print("Eval SCDEED")
    tw_df = evaluate_embedding(
        X_high = X_scdeed,
        X_low = teacher_embed[f"scdeed_{s}"],
        k_values = range(5, 51, 5)
    )
    tw_df.update({"method": f'SCDEED_SEED{s} ({scdeed_best["tsne"][s]})'})
    by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
by_k_tsne.to_csv('mnist_by_k_tsne_all_methods.csv')


# by_k_tsne = pd.DataFrame([])
# for perp in [    5,    11,    27,    62,   146,   341,   793,  1846]:
#     teacher_embed = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/mnist_tsne_{perp}_0_train_pcs.npy")
#     tw_df = evaluate_embedding(
#         X_high = X,
#         X_low = teacher_embed,
#         k_values = range(5, 51, 5)
#     )
#     tw_df.update({"method": f'MEDAL_{perp}', "split": "Train"})
#     by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])
    
    
#     student = load_trained_ae(Path(PATH_PREFIX) / f"drd_data/tmp_results/compare_pcs/mnist/medal_tsne2_{perp}_tc0_0_ckpts/final.pt",
#                 input_dim=784,
#                 hidden_dims=[512, 512, 512, 512],
#                 latent_dim=2, activation=torch.nn.SELU, batchnorm=False)
        
#     student_recon, student_embed = student(torch.tensor(X, dtype=torch.float32))
#     train_distill = np.mean((teacher_embed - student_embed.detach().numpy()) ** 2)
#     if train_distill < 1e-5:
#         _, x_test_embed = student(torch.tensor(X_test, dtype=torch.float32))
#         tw_df = evaluate_embedding(
#             X_high = X_test,
#             X_low = x_test_embed.detach().numpy(),
#             k_values = range(5, 51, 5)
#         )
#         tw_df.update({"method": f'MEDAL_{perp}', "split": "Test"})
#         by_k_tsne = pd.concat([by_k_tsne, pd.DataFrame(tw_df)])

# UMAP

teacher_embed = {}
for s in seed_list:
    teacher_embed[f'scdeed_{s}'] = np.load(Path(PATH_PREFIX) / f"MEDAL/comparisons/data/mnist_train_umap_{scdeed_best['umap'][s]}_0.1_{s}_train_pc6.npy")
    teacher_embed[f'embedr_{s}'] = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/mnist_umap_{embedr_best['umap'][s]}_0.1_{s}_train_embedr.npy").squeeze(0)
        
by_k_umap = pd.DataFrame([])
for s in seed_list:
    print("Eval MEDAL")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = get_teacher_embeddings("umap", X_og, n_neighbors = medal_best["umap"][s], min_dist = 0.1, random_state=s),
        k_values = range(5, 51, 5)
    )
    tw_df.update({"method": f'MEDAL_SEED{s} ({medal_best["umap"][s]})'})
    by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
    
    print("Eval EMBEDR")
    tw_df = evaluate_embedding(
        X_high = X_og,
        X_low = teacher_embed[f"embedr_{s}"],
        k_values = range(5, 51, 5)
    )
    tw_df.update({"method": f'EMBEDR ({embedr_best["umap"][s]})'})
    by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
    
    print("Eval SCDEED")
    tw_df = evaluate_embedding(
        X_high = X_scdeed,
        X_low = teacher_embed[f"scdeed_{s}"],
        k_values = range(5, 51, 5)
    )
    tw_df.update({"method": f'SCDEED ({scdeed_best["umap"][s]})'})
    by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
    
by_k_umap.to_csv('mnist_by_k_umap_all_methods.csv')


# by_k_umap = pd.DataFrame([])
# for n in np.unique(np.logspace(np.log10(5), np.log10(500), 15).astype(int)):
#     teacher_embed = np.load(Path(PATH_PREFIX) / f"drd_data/embeddings/mnist_umap_{n}_0.1_0_train_embedr.npy").squeeze(0)
#     tw_df = evaluate_embedding(
#         X_high = X,
#         X_low = teacher_embed,
#         k_values = range(5, 1001, 5)
#     )
#     tw_df.update({"method": f'MEDAL_{n}', "split": "Train"})
#     by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
    
    
#     student = load_trained_ae(Path(PATH_PREFIX) / f"drd_data/tmp_results/compare_embedr/mnist/medal_umap2_{n}_0.1_tc0_0_ckpts/final.pt",
#                 input_dim=784,
#                 hidden_dims=[512, 512, 512, 512],
#                 latent_dim=2, activation=torch.nn.SELU, batchnorm=False)
        
#     student_recon, student_embed = student(torch.tensor(X, dtype=torch.float32))
#     train_distill = np.mean((teacher_embed - student_embed.detach().numpy()) ** 2)
#     if train_distill < 1e-5:
#         _, x_test_embed = student(torch.tensor(X_test, dtype=torch.float32))
#         tw_df = evaluate_embedding(
#             X_high = X_test,
#             X_low = x_test_embed.detach().numpy(),
#             k_values = range(5, 1001, 5)
#         )
#         tw_df.update({"method": f'MEDAL_{n}', "split": "Test"})
#         by_k_umap = pd.concat([by_k_umap, pd.DataFrame(tw_df)])
            
# by_k_umap.reset_index(names='k', inplace=True)
# by_k_umap.to_csv('mnist_by_k_umap_embedr.csv')