## ============================================================
## scDEED runner (dataset-specific grids + separate output dirs)
## - Saves tSNE outputs to: results_scdeed_tsne/
## - Saves UMAP outputs to: results_scdeed_umap/
## - Uses dataset-specific hyperparameter grids (from your table)
## - More robust package install: uses pak if available; otherwise stops
##   with a clear message if Seurat (and deps) cannot be installed here.
## ============================================================

## --------------------------
## 0) CRAN mirror + user lib
## --------------------------
if (is.null(getOption("repos")) ||
    is.na(getOption("repos")[["CRAN"]]) ||
    getOption("repos")[["CRAN"]] %in% c("", "@CRAN@")) {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
}

user_lib <- Sys.getenv("R_LIBS_USER")
if (!nzchar(user_lib)) {
  user_lib <- file.path(Sys.getenv("HOME"), "R",
                        paste0(R.version$platform, "-library"),
                        paste(R.version$major, R.version$minor, sep = "."))
}
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))

## --------------------------
## 1) Packages
## --------------------------
pkgs <- c(
  "Seurat", "SeuratObject",
  "gridExtra", "dplyr", "patchwork", "VGAM",
  "gplots", "ggplot2", "pracma", "resample", "foreach",
  "distances", "doParallel"
)

## NOTE on your install error:
## On Longleaf/RHEL, Seurat often fails to install inside a batch job because
## system libs (hdf5, gdal/geos, etc.) and compilation toolchain may be missing
## from the compute node environment. So we:
##   - try to install via pak (more reliable dependency handling),
##   - then STOP if Seurat still isn’t available (instead of silently continuing).

install_missing_pkgs <- function(pkgs, lib = user_lib, ncpu = max(1, parallel::detectCores() - 1)) {
  missing <- setdiff(pkgs, rownames(installed.packages(lib.loc = .libPaths())))
  if (!length(missing)) return(invisible(TRUE))
  
  message("Missing packages: ", paste(missing, collapse = ", "))
  
  ## Prefer pak if possible
  if (!requireNamespace("pak", quietly = TRUE)) {
    message("Installing 'pak' first (to ", lib, ") ...")
    install.packages("pak", lib = lib, dependencies = TRUE, Ncpus = ncpu)
  }
  
  if (requireNamespace("pak", quietly = TRUE)) {
    message("Installing missing packages with pak ...")
    ## pak ignores lib.loc; set lib path via .libPaths already
    pak::pkg_install(missing, upgrade = FALSE)
  } else {
    stop(
      "Could not install/use 'pak'.\n",
      "On Longleaf, Seurat may require system libraries and a compiler toolchain.\n",
      "Install missing packages interactively (login node) or via a prepared environment, then re-run the job."
    )
  }
  
  invisible(TRUE)
}

install_missing_pkgs(pkgs)

## Load
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop(
      "Package '", p, "' is still missing after install attempt.\n",
      "Most important: Seurat must be installed successfully before running scDEED.\n",
      "Try installing on a login node (interactive) and ensure system deps are available."
    )
  }
  library(p, character.only = TRUE)
}

## Parallel
doParallel::registerDoParallel(cores = 30)

## --------------------------
## 2) Source scDEED (patched)
## --------------------------
## This expects that you already replaced slot->layer in scDEED/R/scDEED.R
source("scDEED/R/scDEED.R")
#library(scDEED)
## --------------------------
## 3) Dataset-specific grids (from your table)
## --------------------------
mnist_umap_n <- c(5, 6, 9, 13, 18, 25, 35, 49, 69, 96, 134, 186, 258, 359, 499)
mnist_tsne_p <- c(5, 11, 27, 62, 146, 341, 793, 1846)

hydra_umap_n <- c(5, 9, 18, 36, 71, 139, 271, 528, 1027, 2000)
hydra_tsne_p <- c(5, 10, 23, 49, 107, 232, 499, 1077, 2320, 4999)

tasic_umap_n <- hydra_umap_n
tasic_tsne_p <- c(5, 10, 24, 53, 116, 256, 564, 1241, 2729, 6000)

astro_umap_n <- mnist_umap_n
astro_tsne_p <- c(3, 4, 6, 8, 12, 18, 26, 55, 80, 115, 167, 240, 346, 499)

## min_dist is 0.1 in your table (MNIST + Hydra; and “same as” implies same)
default_min_dist <- 0.1

infer_dataset_key <- function(name_or_path) {
  x <- tolower(basename(name_or_path))
  if (grepl("mnist", x)) return("mnist")
  if (grepl("hydra", x)) return("hydra")
  if (grepl("tasic", x)) return("tasic")
  if (grepl("astro", x)) return("astro")
  return("unknown")
}

get_param_grid <- function(key) {
  if (key == "mnist") {
    return(list(
      umap_n_neighbors = mnist_umap_n,
      umap_min_dist    = default_min_dist,
      tsne_perplexity  = mnist_tsne_p
    ))
  }
  if (key == "hydra") {
    return(list(
      umap_n_neighbors = hydra_umap_n,
      umap_min_dist    = default_min_dist,
      tsne_perplexity  = hydra_tsne_p
    ))
  }
  if (key == "tasic") {
    return(list(
      umap_n_neighbors = tasic_umap_n,
      umap_min_dist    = default_min_dist,
      tsne_perplexity  = tasic_tsne_p
    ))
  }
  if (key == "astro") {
    return(list(
      umap_n_neighbors = astro_umap_n,
      umap_min_dist    = default_min_dist,
      tsne_perplexity  = astro_tsne_p
    ))
  }
  ## fallback (your original defaults)
  return(list(
    umap_n_neighbors = seq(from = 5, to = 200, by = 25),
    umap_min_dist    = default_min_dist,
    tsne_perplexity  = seq(from = 10, to = 410, by = 20)
  ))
}

## --------------------------
## 4) Runner
## --------------------------
run_scdeed_all_data <- function(path, name, K, load_this_seed) {
  message("==== Dataset: ", name, " ====")
  
  ## Read
  df <- read.csv(path, check.names = FALSE)
  
  ## Drop PC* and metadata columns
  drop_cols <- grepl("^PC", names(df)) | names(df) %in% c("split", "labels")
  df <- df[, !drop_cols, drop = FALSE]
    
  if ("label" %in% names(df)) {
        cell_ids <- df$label
        df <- df[, names(df) != "label", drop = FALSE]
    } else {
        cell_ids <- paste0("cell_", seq_len(nrow(df)))
    }


  ## Drop any columns with empty/NA names (prevents Seurat/Matrix errors)
  bad_names <- is.na(names(df)) | trimws(names(df)) == ""
  if (any(bad_names)) {
    df <- df[, !bad_names, drop = FALSE]
  }
  
  feature_names <- names(df)
  ## Coerce to numeric & finite
  df[] <- lapply(df, function(x) suppressWarnings(as.numeric(x)))
  #mat_cells_features <- as.matrix(df)          # cells x features
  #mat_cells_features[!is.finite(mat_cells_features)] <- 0
  mat <- t(as.matrix(df))
  mat[!is.finite(mat)] <- 0
  ## Seurat expects features x cells → transpose
  # mat <- t(mat_cells_features)                 # features x cells
  
  ## Ensure feature names exist + unique (avoid LogMap / empty rowname issues)
  #feat <- rownames(mat)
  #feat[is.na(feat) | trimws(feat) == ""] <- paste0("feature_", which(is.na(feat) | trimws(feat) == ""))
  #feat <- make.unique(feat)
  #rownames(mat) <- feat
  rownames(mat) <- make.unique(ifelse(
        is.na(feature_names) | trimws(feature_names) == "",
        paste0("feature_", seq_along(feature_names)),
        feature_names
    ))
  colnames(mat) <- make.unique(ifelse(
        is.na(cell_ids) | trimws(as.character(cell_ids)) == "",
        paste0("cell_", seq_along(cell_ids)),
        as.character(cell_ids)
    ))
  message("n = ", ncol(mat), ", p = ", nrow(mat))
  ## Build Seurat object and place your already-scaled data in scale.data (no extra norm/scale)
  data <- CreateSeuratObject(counts = mat, assay = "RNA")
  DefaultAssay(data) <- "RNA"
  data <- SetAssayData(data, assay = "RNA", slot = "scale.data", new.data = mat)
  
  ## Use ALL features as variable features (so PCA uses all columns you provide)
  Seurat::VariableFeatures(data) <- rownames(mat)
  
  ## Default scDEED example-style K
  #K <- 8
  
  ## Run PCA (creates the default 'pca' reduction that scDEED expects)
  data <- Seurat::RunPCA(
    data,
    npcs     = K,
    features = Seurat::VariableFeatures(data),
    verbose  = FALSE
  )
  pca_embed <- data@reductions$pca@cell.embeddings[, 1:K]
  write.csv(pca_embed, file.path("comparisons/data", paste0(name, "_pc", K, ".csv")))
  
  ## Dataset-specific grids
  key   <- infer_dataset_key(name)
  grid  <- get_param_grid(key)
  
  ## Output dirs
  out_tsne <- paste0("results_scdeed_tsne/seed", load_this_seed)
  out_umap <- paste0("results_scdeed_umap/seed", load_this_seed)
  dir.create(out_tsne, showWarnings = FALSE, recursive = TRUE)
  dir.create(out_umap, showWarnings = FALSE, recursive = TRUE)
                 
  reticulate::use_python("~/.conda/envs/medalcomp/bin/python", required = TRUE)
  np <- reticulate::import("numpy")
  openTSNE <- reticulate::import("openTSNE")
  umap_lib <- reticulate::import("umap")
  
  run_and_save_tsne <- function(pca_mat, perplexity, save) {
    tsne <- openTSNE$TSNE(
      perplexity   = as.integer(perplexity),
      random_state = as.integer(load_this_seed),
      initialization = "random"
    )
    embed <- reticulate::py_to_r(tsne$fit(pca_mat))
    if (save){
        np$save(
          file.path("comparisons/data", sprintf("%s_tsne_%d_%d_train_pc%d.npy", name, as.integer(perplexity), load_this_seed, K)),
          reticulate::r_to_py(embed)
        )
    }
    embed
  }
  
  run_and_save_umap <- function(pca_mat, n_neighbors, min_dist, save) {
      reducer <- umap_lib$UMAP(
        n_neighbors  = as.integer(n_neighbors),
        min_dist     = min_dist,
        metric       = "euclidean",
        random_state = as.integer(load_this_seed)
      )
      embed <- reticulate::py_to_r(reducer$fit_transform(pca_mat))
      if (save) {
        np$save(
          file.path("comparisons/data",
                    sprintf("%s_umap_%d_%.1f_%d_train_pc%d.npy", name, as.integer(n_neighbors), min_dist,load_this_seed, K)),
          reticulate::r_to_py(embed)
        )
      }
      embed
    }
  
  ## --- t-SNE ---
  cat("Starting scDEED t-SNE...\n")
  start <- Sys.time()
  n_cells <- ncol(data)
  max_perp <- floor((n_cells - 1) / 3)
  cat("filtered grid:", grid$tsne_perplexity[grid$tsne_perplexity < floor((ncol(data)-1)/3)], "\n")
  result_tsne <- scDEED(
    data,
    K                     = K,
    reduction.method      = "tsne",
    perplexity            = grid$tsne_perplexity[grid$tsne_perplexity < max_perp],
    check_duplicates      = FALSE,
    embedding_fn = function(perplexity) {
      run_and_save_tsne(pca_embed, perplexity, save = TRUE)
    },
    permuted_embedding_fn = function(seurat_obj, perplexity) {
      pca_mat <- seurat_obj@reductions$pca@cell.embeddings[, 1:K]
      run_and_save_tsne(pca_mat, perplexity, save = FALSE)
    }
  )
  end <- Sys.time()
  time_tsne <- end - start
  print(time_tsne)
                 
  saveRDS(result_tsne, file.path(out_tsne, paste0("tsne_best_", name, ".Rds")))
  saveRDS(data.frame(time = time_tsne, method = "tSNE"),
          file.path(out_tsne, paste0("timeelapsed_", name, ".Rds")))
  
  ## --- UMAP ---
  cat("Starting scDEED UMAP...\n")
  start <- Sys.time()
  
  result_umap <- scDEED(
    data,
    K                     = K,
    reduction.method      = "umap",
    n_neighbors           = grid$umap_n_neighbors,
    min.dist              = grid$umap_min_dist,
    check_duplicates      = FALSE,
    embedding_fn = function(n_neighbors, min_dist) {
      run_and_save_umap(pca_embed, n_neighbors, min_dist, save = TRUE)
    },
    permuted_embedding_fn = function(seurat_obj, n_neighbors, min_dist) {
      pca_mat <- seurat_obj@reductions$pca@cell.embeddings[, 1:K]
      run_and_save_umap(pca_mat, n_neighbors, min_dist, save = FALSE)
    }
  )
  end <- Sys.time()
  time_umap <- end - start
  print(time_umap)
  
  # Save (separate folders)
  
  saveRDS(result_umap, file.path(out_umap, paste0("umap_best_", name, ".Rds")))
  saveRDS(data.frame(time = time_umap, method = "UMAP"),
          file.path(out_umap, paste0("timeelapsed_", name, ".Rds")))
  
  invisible(TRUE)
}

## --------------------------
## 5) Datasets list
## --------------------------
datasets <- list.files("data_tmp/", full.names = TRUE)

## If you want to exclude MNIST + Astro, uncomment:
datasets <- datasets[!grepl("mnist|tasic|hydra", datasets, ignore.case = TRUE)]

for (lts in c(2,10)){
    for (dataset in datasets) {
      name <- tools::file_path_sans_ext(basename(dataset))
      run_scdeed_all_data(path = dataset, name = name, K=5, load_this_seed = lts)
      
    }
}
