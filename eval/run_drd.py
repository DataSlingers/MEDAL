import numpy as np, pandas as pd
from pathlib import Path
if not hasattr(np, "product"):
    np.product = np.prod
from src.drd import DRD
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from umap.parametric_umap import ParametricUMAP
from sklearn.metrics import mean_squared_error
from keras.losses import MeanSquaredError

import tensorflow as tf, os

os.environ["CUDA_VISIBLE_DEVICES"] = "" 

# load data
wine_data = load_wine()
X_wine = wine_data.data
y_wine = wine_data.target
X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.5, random_state=42)

# teacher model PCA
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))
])
pca_pipeline.fit(X_train)
scaler = pca_pipeline.named_steps['scaler']
# transform data
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test) # use scales from training set

# get PCA embeddings
teacher_pca_embeddings = pca_pipeline.transform(X_train)
teacher_pca_test_embeddings = pca_pipeline.transform(X_test)

distill_drd_train, distill_pumap_train = [], []
recon_drd_train,   recon_pumap_train   = [], [] 
distill_drd_test, distill_pumap_test = [], []
recon_drd_test,   recon_pumap_test   = [], []
lambda_ls = np.linspace(0.1, 1, 5)
for lmda_val in lambda_ls:
    # run UMAP-AE
    p_umap = ParametricUMAP(
        batch_size=256,
        n_components=2,
        parametric_reconstruction=True,
        parametric_reconstruction_loss_weight=lmda_val,
        parametric_reconstruction_loss_fcn=MeanSquaredError(),
        autoencoder_loss=False,
        verbose=False,
        random_state=42,
        low_memory      = True
    )
    p_umap.fit(X_train_s, landmark_positions = teacher_pca_embeddings)

    # run AE whose distillation loss is MSE wrt UMAP
    drd = DRD(input_dim=X_train.shape[1], latent_dim=2, epochs=10, 
              batch_size=256, lambda_d = lmda_val)
    drd.fit(X_train_s, teacher_Z = teacher_pca_embeddings)

    # TRAIN
    # embedding/distillation loss on training
    student_drd_embedding = drd.transform(X_train_s)
    student_pumap_embedding = p_umap.transform(X_train_s)
    d_drd_train = mean_squared_error(student_drd_embedding, teacher_pca_embeddings)
    d_pumap_train = mean_squared_error(student_pumap_embedding, teacher_pca_embeddings)

    # reconstruction loss on training
    X_train_recon_drd = drd.inverse_transform(student_drd_embedding)
    X_train_recon_pumap = p_umap.inverse_transform(student_pumap_embedding)
    r_drd_train = mean_squared_error(X_train_recon_drd, X_train_s)
    r_pumap_train = mean_squared_error(X_train_recon_pumap, X_train_s)

    # TEST
    # embedding/distillation loss on test
    student_drd_test_embedding = drd.transform(X_test_s)
    student_pumap_test_embedding = p_umap.transform(X_test_s)
    d_drd_test = mean_squared_error(student_drd_test_embedding, teacher_pca_test_embeddings)
    d_pumap_test = mean_squared_error(student_pumap_test_embedding, teacher_pca_test_embeddings)

    # reconstruction loss on test
    X_test_recon_drd = drd.inverse_transform(student_drd_test_embedding)    
    X_test_recon_pumap = p_umap.inverse_transform(student_pumap_test_embedding)
    r_drd_test = mean_squared_error(X_test_recon_drd, X_test_s)
    r_pumap_test = mean_squared_error(X_test_recon_pumap, X_test_s)

    distill_drd_train.append(d_drd_train);   distill_pumap_train.append(d_pumap_train)
    recon_drd_train.append(r_drd_train);     recon_pumap_train.append(r_pumap_train)
    distill_drd_test.append(d_drd_test);   distill_pumap_test.append(d_pumap_test)
    recon_drd_test.append(r_drd_test);     recon_pumap_test.append(r_pumap_test)


results = pd.DataFrame({
    "lambda":         lambda_ls,
    "distill_drd_train":    distill_drd_train,
    "distill_pumap_train":  distill_pumap_train,
    "recon_drd_train":      recon_drd_train,
    "recon_pumap_train":    recon_pumap_train,
    "distill_drd_test":    distill_drd_test,
    "distill_pumap_test":  distill_pumap_test,
    "recon_drd_test":      recon_drd_test,
    "recon_pumap_test":    recon_pumap_test,
})
Path("results").mkdir(exist_ok=True)               
results.to_csv("results/umap_drd_losses.csv", index=False)