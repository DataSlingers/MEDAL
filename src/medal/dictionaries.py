import numpy as np

DISTILL_BANDS_DICT = {
    "gene_cancer": [(1e-12, 9e-6)],
    "mnist": [ (1e-12, 9e-6)], #(1e-1, 5), (1e-2, 1e-1), (1e-4, 1e-2), (1e-6,1e-4), (9e-8, 1e-6),
    "darmanis": [(1e-12, 9e-6)],
    "wine": [(1e-12, 9e-8)],
    "hydra": [(1e-12, 9e-6)],
    #[(1e-1, 1e8), (1e-2, 1e-1), (1e-4, 1e-2), (1e-6,1e-4), (9e-8, 1e-6), (1e-12, 9e-8)],
    "pbmc": [(1e-12, 9e-6)],
    "astro": [(1e-12, 9e-6)],
    "cortical": [(1e-12, 9e-6)],
    "macaque": [(1e-12, 9e-6)],
}

PATH_PREFIX = "/share/ctn/users/bnc2119/drd_data"
# PATH_PREFIX = "/shared/share_mala/irchang/drd"

TEACHER_SWEEP_SPECS = {
    "mnist": {
        "umap": {
            "t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(500), 15).astype(int)), 
            "min_dist": [0.1], 
            "n_components": [2]},
        "tsne": {
            "perplexity": [5, 11,  27, 62, 146,  341, 793, 1846, 4297], 
            "n_components": [2]},
        "spectral": {
            "n_components": [2],
            "t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(500), 15).astype(int))},
        "pca": {"n_components": [2]},
    },
    "hydra": {
        "umap": {"t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(2000), 10).astype(int)), "min_dist": [0.1]},
        "tsne": {"perplexity": np.unique(np.logspace(np.log10(5), np.log10(5000), 10).astype(int))},
        "spectral": {"t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(200), 15).astype(int))},
        "pca": {"n_components": [2]},
    },
    "cortical": {
        "umap": {"t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(2000), 10).astype(int)), "min_dist": [0.1]},
        "tsne": {"perplexity": np.unique(np.logspace(np.log10(5), np.log10(6000), 10).astype(int))},
        "spectral": {"t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(200), 15).astype(int))},
        "pca": {"n_components": [2]},
    },
    "astro": {
        "umap": {"t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(500), 15).astype(int)), "min_dist": [0.1]},
        "tsne": {"perplexity": np.unique(np.logspace(np.log10(3), np.log10(500), 15).astype(int))},
        "spectral": {"t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(500), 15).astype(int))},
        "pca": {"n_components": [2]},
    },
}

RANK_SWEEP_SPECS = {
    "mnist": {
        "umap": {"t_n_neighbors": [18], "min_dist": [0.1], 
                 "n_components": np.unique(np.logspace(np.log10(2), np.log10(100), 10).astype(int))},
        "spectral": {"t_n_neighbors": [18],
                     "n_components": np.unique(np.logspace(np.log10(2), np.log10(100), 10).astype(int))},
        "pca": {"n_components": np.unique(np.logspace(np.log10(2), np.log10(100), 10).astype(int))},
    },
    "gene_cancer": {
        "umap": {"t_n_neighbors": [5], "min_dist": [0.1],
                 "n_components": np.unique(np.logspace(np.log10(2), np.log10(100), 10).astype(int))},
        "spectral": {"t_n_neighbors": [5],
                     "n_components": np.unique(np.logspace(np.log10(2), np.log10(100), 10).astype(int))},
        "pca": {"n_components": np.unique(np.logspace(np.log10(2), np.log10(100), 10).astype(int))},
    },
}

INIT_CONFIG = {
    "gene_cancer": {
        "lr": 1e-3, #tune.grid_search([1e-5, 1e-4, 5e-4, 5e-5]), #1e-3,  
        "lambda_d": 10000,#30000, # 1500
        "eta_min": 1e-11, #1e-9-
        "hidden_dims":[1000, 1000, 1000, 1000],
        "activation": "SELU",
        "bottleneck_activation": None,
        "max_epochs": 7000,
        "warmup": 0, 
        "test_size": 0.2,
        "batch_size": 100,
        "t_factor": 0.95,
        "t_patience": 20,
        "use_batchnorm": True
    },
    "wine": {
        "lr": 0.02, #0.003233466538536306,
        "lambda_d": 20000,
        "eta_min": 1e-8, 
        "hidden_dims": [258, 258, 258, 258],
        "activation": "SELU",
        "bottleneck_activation":  None,
        'max_epochs': 130000, 
        "warmup": 0, 
        "batch_size": 100
    },
    "mnist": { 
        "lr": 1e-3, #1e-4 (vanillaAE), #1e-3 (ROP), # 2e-5, #0.000269 (tsne),	
        "lambda_d": 10000, # 3000
        "eta_min": 1e-7, #1e-5, # 7.256237e-10, 1e-10(spectral)
        "hidden_dims": [1000, 1000, 1000, 1000, 1000], 
        "activation": "SELU",
        "bottleneck_activation": None,
        "max_epochs":5000, 
        "batch_size": 256,
        "warmup": 0, 
        "test_size":0.2,
        "t_patience": 20,
        "use_batchnorm": False,
    },
    "darmanis": {
        "lr": 0.001,
        "lambda_d": 50000,
        "eta_min": 1e-7, 
        "hidden_dims": [1000] * 4,
        "activation": "SELU",
        "bottleneck_activation": None,
        'max_epochs': 20000, 
        "warmup": 0, 
        "batch_size": 10000,
        "t_patience":20,
        "t_factor": 0.95,
        "use_batchnorm": False,
        "test_size": 0.2,
        "t_patience": 20,
    },
    "hydra":{
        "lr": 0.005, #0.0005 (old lr),# 0.005 (new lr), 
        "lambda_d": 30000, #30000,
        # "lambda_d": 0,
        # "eta_min1":  9.10708e-06, 
        # "eta_min2": 8.51602e-10, 
        "eta_min":  1e-07, 
        "hidden_dims": [309, 1792, 1792, 1792],
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 20000,
        "warmup": 1500, 
        "batch_size": 51200,
        "test_size": 0.2,
        "t_patience":20,
        "t_factor": 0.95,
        "use_batchnorm": True
    },
    "astro":{
        "lr": 0.0005, #0.00139911,
        "lambda_d": 10000,
        "eta_min": 1e-07, #3.55767e-07,
        "hidden_dims": [700] * 15,
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 60000,
        "warmup": 0, 
        "batch_size": 25600,
        "test_size": 0.2,
        "t_patience":70,
        "t_factor": 0.9,
        "use_batchnorm": False
    },
    "cortical":{
        "lr": 0.00268681,
        "lambda_d": 50000, #30000,
        "eta_min": 1e-7,
        "hidden_dims": [309, 1792, 1792, 1792],
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 6000,
        "warmup": 100, 
        "batch_size": 50000,
        "test_size": 0.2,
        "t_patience":20,
        "t_factor": 0.95,
        "use_batchnorm": True
    },
    "macaque":{       
        "lr":0.000341178, 
        "lambda_d": 10000, # 50000, 200000
        "eta_min": 1e-7, #6.24882e-06 (tsne), 1e-7 (umap)
        "hidden_dims": [700] * 15,
        "activation": "SELU", 
        "bottleneck_activation": None,
        'max_epochs': 25000,
        "warmup": 0, 
        "batch_size": 1024,
        "test_size":0.2,
        "t_patience":70,
        "t_factor": 0.9,
        "use_batchnorm": False
    },
}
