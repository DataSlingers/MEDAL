from openTSNE import TSNE
from openTSNE.affinity import PerplexityBasedNN
from umap import UMAP
import scipy.io
from scipy.io import mmwrite
import numpy as np
from pathlib import Path

from medal.eval_utils import load_and_split

PATH_PREFIX = '/share/ctn/users/bnc2119' 

X, X_test, y, y_test = load_and_split("hydra", test_size=0.2, seed=0, labels=True)
for n in np.unique(np.logspace(np.log10(5), np.log10(2000), 10).astype(int)):
    for i in range(1, 6):
        id_set = np.load(Path(PATH_PREFIX) / f"MEDAL/comparisons/data/hydra_og_idx{i}.npy")
        X_subset = X[id_set]

#         aff = PerplexityBasedNN(X_subset,perplexity=perp, metric="euclidean", n_jobs=1,random_state=0)
#         P = aff.P

#         tsne = TSNE(n_components=2,initialization="pca",negative_gradient_method="fft",n_jobs=1,random_state=0)

#         Y = tsne.fit(affinities=aff)
#         Y = np.asarray(Y, dtype=np.float64)
        
        umap_obj = UMAP(n_neighbors=n, min_dist=0.1, random_state=0).fit(X_subset)
        Y = umap_obj.embedding_
        Y = np.asarray(Y, dtype=np.float64)
        P = umap_obj.graph_

        # np.save(Path(PATH_PREFIX) / f'drd_data/embeddings/hydra_tsne_{perp}_0_train.npy', Y)
        # print(f"Saved {Path(PATH_PREFIX) / f'drd_data/embeddings/hydra_tsne_{perp}_0_train.npy'}")
        np.save(Path(PATH_PREFIX) / f"drd_data/embeddings/hydra_umap_{n}_0.1_0_train_{i}.npy", Y)
        print(f"Saved {Path(PATH_PREFIX) / f'drd_data/embeddings/hydra_umap_{n}_0.1_0_train_{i}.npy'}")
        mmwrite(Path(PATH_PREFIX) / f"drd_data/embeddings/hydra_umap_{n}_0.1_0_P_{i}.mtx", P)
        print(f"Saved {Path(PATH_PREFIX) / f'drd_data/embeddings/hydra_umap_{n}_0.1_0_P_{i}.mtx'}")