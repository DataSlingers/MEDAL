# ============================================================
# scDEED_modified.R
#
# Key changes vs original:
#
# 1. PRE-COMPUTED EMBEDDINGS
#    - scDEED() gains `embedding_fn` and `permuted_embedding_fn`,
#      both functions, so each grid point loads its own .npy file.
#
#    embedding_fn signatures:
#      tSNE:  function(perplexity)           -> n_cells x 2 matrix
#      UMAP:  function(n_neighbors, min_dist) -> n_cells x 2 matrix
#
#    permuted_embedding_fn signatures:
#      tSNE:  function(seurat_obj, perplexity)            -> matrix
#      UMAP:  function(seurat_obj, n_neighbors, min_dist) -> matrix
#
# 2. PARAMETER ALIGNMENT with OpenTSNE / Python UMAP
#    - align_with_python = TRUE adjusts RunTSNE/RunUMAP defaults
#      to match OpenTSNE / umap-learn as closely as possible.
#
# HOW TO USE (pre-computed embeddings, one file per perplexity):
#
#   np <- reticulate::import("numpy")
#
#   result <- scDEED(
#     input_data       = seurat_obj,
#     K                = 30,
#     reduction.method = "tsne",
#     perplexity       = c(30, 50, 100),
#     embedding_fn = function(perplexity) {
#       mat <- np$load(sprintf(".../tsne_%d.npy", as.integer(perplexity)))
#       reticulate::py_to_r(mat)
#     },
#     permuted_embedding_fn = function(seurat_obj, perplexity) {
#       # run OpenTSNE on permuted PCA coords via reticulate, return matrix
#     }
#   )
#
# HOW TO USE (R-internal but aligned with Python defaults):
#
#   result <- scDEED(..., align_with_python = TRUE)
# ============================================================

# -----------------------------------------------------------------------
# Helper: inject a pre-computed embedding matrix into a Seurat object
# so downstream code that reads @reductions$tsne / $umap still works.
# -----------------------------------------------------------------------
.inject_embedding <- function(seurat_obj, mat, reduction_name) {
  stopifnot(is.matrix(mat), is.numeric(mat))
  colnames(mat) <- paste0(toupper(reduction_name), "_", seq_len(ncol(mat)))
  rownames(mat) <- colnames(seurat_obj)

  dim_reduc <- Seurat::CreateDimReducObject(
    embeddings = mat,
    key        = paste0(toupper(reduction_name), "_"),
    assay      = Seurat::DefaultAssay(seurat_obj)
  )
  seurat_obj@reductions[[reduction_name]] <- dim_reduc
  seurat_obj
}


# -----------------------------------------------------------------------
# Permuted  (unchanged from original except slot->layer compat)
# -----------------------------------------------------------------------
Permuted <- function(pbmc,
                     default_assay = "active.assay",
                     layer = NULL,
                     slot = "scale.data",
                     run_pca = TRUE,
                     K) {

  if (is.null(layer) || !nzchar(layer)) layer <- slot

  if (default_assay == "active.assay") {
    default_assay <- pbmc@active.assay
  }

  pb <- txtProgressBar(min = 0, max = 1, initial = 0, char = "=",
                       width = NA, style = 3, file = "")
  setTxtProgressBar(pb, 0.1)

  Seurat::DefaultAssay(object = pbmc) <- default_assay
  pbmc.permuted <- pbmc

  X          <- Seurat::GetAssayData(pbmc,          slot = slot)
  X_permuted <- Seurat::GetAssayData(pbmc.permuted, slot = slot)

  setTxtProgressBar(pb, 0.4)

  set.seed(999999)
  n_features <- dim(X)[1]
  curr_pb    <- 0.4

  for (i in 1:n_features) {
    row <- pracma::randperm(dim(X)[2])
    X_permuted[i, ] <- X[i, row]
    curr_pb <- 0.4 + 0.6 * (i / n_features)   # linearly 0.4 -> 1.0, never overshoots
    setTxtProgressBar(pb, min(curr_pb, 1))
  }

  pbmc.permuted <- Seurat::SetAssayData(
    pbmc.permuted,
    slot     = slot,
    new.data = X_permuted,
    assay    = default_assay
  )

  if (isTRUE(run_pca)) {
    pbmc.permuted <- Seurat::RunPCA(
      pbmc.permuted,
      npcs     = K,
      features = Seurat::VariableFeatures(object = pbmc.permuted)
    )
  }

  setTxtProgressBar(pb, 1)
  return(pbmc.permuted)
}


# -----------------------------------------------------------------------
# Distances.pre_embedding  (unchanged)
# -----------------------------------------------------------------------
Distances.pre_embedding <- function(pbmc, pbmc_permuted, K, pre_embedding = "pca") {
  distances <- distances::distances

  M                            <- pbmc@reductions[[pre_embedding]]@cell.embeddings
  pre_embedding_distances      <- distances(M[, 1:K])

  M_permuted                        <- pbmc_permuted@reductions[[pre_embedding]]@cell.embeddings
  pre_embedding_distances_permuted  <- distances(M_permuted[, 1:K])

  list(
    pre_embedding_distances          = pre_embedding_distances,
    pre_embedding_distances_permuted = pre_embedding_distances_permuted
  )
}


# -----------------------------------------------------------------------
# Distances.tSNE
#
# NEW arguments:
#   embedding_fn          – function(perplexity) -> n_cells x 2 matrix.
#                           Called for the ORIGINAL data at each grid point.
#                           If NULL, RunTSNE is used instead.
#   permuted_embedding_fn – function(seurat_obj, perplexity) -> matrix.
#                           Called for the PERMUTED data at each grid point.
#                           If NULL, RunTSNE is used instead.
#   align_with_python     – logical. When TRUE, tweaks RunTSNE to match
#                           OpenTSNE defaults (init=pca, lr=N/12).
# -----------------------------------------------------------------------
Distances.tSNE <- function(pbmc, pbmc.permuted, K,
                            perplexity_score      = 40,
                            pre_embedding         = "pca",
                            check_duplicates      = TRUE,
                            rerun                 = TRUE,
                            # --- new ---
                            embedding_fn          = NULL,   # fn(perplexity) -> matrix
                            permuted_embedding_fn = NULL,   # fn(obj, perplexity) -> matrix
                            align_with_python     = FALSE) {

  distances <- distances::distances
  N         <- ncol(pbmc)

  .tsne_args <- function() {
    args <- list(
      seed.use         = 100,
      perplexity       = perplexity_score,
      reduction        = pre_embedding,
      do.fast          = TRUE,
      check_duplicates = check_duplicates,
      dim.embed        = 2
    )
    if (align_with_python) {
      args$initialization <- "pca"
      args$learning.rate  <- N / 12
      args$max_iter       <- 1000
    }
    args
  }

  # ---- Original data embedding ----
  if (!is.null(embedding_fn)) {
    mat  <- embedding_fn(perplexity_score)
    pbmc <- .inject_embedding(pbmc, mat, "tsne")
  } else if (rerun) {
    pbmc <- do.call(Seurat::RunTSNE, c(list(object = pbmc), .tsne_args()))
  }
  tSNE_distances <- distances(pbmc@reductions$tsne@cell.embeddings)

  # ---- Permuted data embedding ----
  if (!is.null(permuted_embedding_fn)) {
    perm_mat      <- permuted_embedding_fn(pbmc.permuted, perplexity_score)
    pbmc.permuted <- .inject_embedding(pbmc.permuted, perm_mat, "tsne")
  } else {
    pbmc.permuted <- do.call(Seurat::RunTSNE,
                             c(list(object = pbmc.permuted), .tsne_args()))
  }
  tSNE_distances_permuted <- distances(pbmc.permuted@reductions$tsne@cell.embeddings)

  list(
    reduced_dim_distances          = tSNE_distances,
    reduced_dim_distances_permuted = tSNE_distances_permuted
  )
}


# -----------------------------------------------------------------------
# Distances.UMAP
#
# NEW arguments:
#   embedding_fn          – function(n_neighbors, min_dist) -> n_cells x 2 matrix.
#                           Called for the ORIGINAL data at each grid point.
#   permuted_embedding_fn – function(seurat_obj, n_neighbors, min_dist) -> matrix.
#                           Called for the PERMUTED data at each grid point.
#   align_with_python     – logical. Forces metric="euclidean" to match
#                           Python umap-learn default (Seurat defaults to cosine).
# -----------------------------------------------------------------------
Distances.UMAP <- function(pbmc, pbmc.permuted, K,
                            pre_embedding         = "pca",
                            n                     = 30,
                            m                     = 0.3,
                            rerun                 = TRUE,
                            # --- new ---
                            embedding_fn          = NULL,
                            permuted_embedding_fn = NULL,
                            align_with_python     = FALSE) {

  distances <- distances::distances

  .umap_args <- function() {
    args <- list(
      dims        = 1:K,
      seed.use    = 100,
      reduction   = pre_embedding,
      n.neighbors = n,
      min.dist    = m
    )
    if (align_with_python) args$metric <- "euclidean"
    args
  }

  # ---- Original data embedding ----
  if (!is.null(embedding_fn)) {
    mat  <- embedding_fn(n, m)
    pbmc <- .inject_embedding(pbmc, mat, "umap")
  } else if (rerun) {
    pbmc <- do.call(Seurat::RunUMAP, c(list(object = pbmc), .umap_args()))
  }
  UMAP_distances <- distances(pbmc@reductions$umap@cell.embeddings)

  # ---- Permuted data embedding ----
  if (!is.null(permuted_embedding_fn)) {
    perm_mat      <- permuted_embedding_fn(pbmc.permuted, n, m)
    pbmc.permuted <- .inject_embedding(pbmc.permuted, perm_mat, "umap")
  } else {
    pbmc.permuted <- do.call(Seurat::RunUMAP,
                             c(list(object = pbmc.permuted), .umap_args()))
  }
  UMAP_distances_permuted <- distances(pbmc.permuted@reductions$umap@cell.embeddings)

  list(
    reduced_dim_distances          = UMAP_distances,
    reduced_dim_distances_permuted = UMAP_distances_permuted
  )
}


# -----------------------------------------------------------------------
# Cell.Similarity  (unchanged)
# -----------------------------------------------------------------------
Cell.Similarity <- function(pre_embedding_distances, pre_embedding_distances_permuted,
                             reduced_dim_distances, reduced_dim_distances_permuted,
                             similarity_percent = 0.50) {

  numberselected <- floor((dim(pre_embedding_distances)[2]) * similarity_percent)

  rho_original <- foreach::`%do%`(
    foreach::foreach(i = 1:(dim(pre_embedding_distances)[2])),
    cor(
      (reduced_dim_distances[i, order(pre_embedding_distances[i, ])][2:(numberselected + 1)]),
      (sort(reduced_dim_distances[i, ])[2:(numberselected + 1)])
    )
  )
  rho_original <- as.numeric(rho_original)

  rho_permuted <- foreach::`%do%`(
    foreach::foreach(i = 1:(dim(pre_embedding_distances)[2])),
    cor(
      (reduced_dim_distances_permuted[i, order(pre_embedding_distances_permuted[i, ])][2:(numberselected + 1)]),
      (sort(reduced_dim_distances_permuted[i, ])[2:(numberselected + 1)])
    )
  )
  rho_permuted <- as.numeric(rho_permuted)

  list(rho_original = rho_original, rho_permuted = rho_permuted)
}


# -----------------------------------------------------------------------
# Cell.Classify  (unchanged)
# -----------------------------------------------------------------------
Cell.Classify <- function(rho_original, rho_permuted,
                           dubious_cutoff = 0.05, trustworthy_cutoff = 0.95) {

  rho_trustworthy   <- quantile(rho_permuted, trustworthy_cutoff)
  rho_dubious       <- quantile(rho_permuted, dubious_cutoff)

  dubious_cells     <- which(rho_original < rho_dubious)
  trustworthy_cells <- which(rho_original > rho_trustworthy)
  intermediate      <- setdiff(seq_along(rho_original), c(dubious_cells, trustworthy_cells))

  list(
    dubious_cells      = dubious_cells,
    trustworthy_cells  = trustworthy_cells,
    intermediate_cells = intermediate
  )
}


# -----------------------------------------------------------------------
# optimize  (internal wrapper)
#   Passes new embedding arguments down to Distances.tSNE / Distances.UMAP.
#   NOTE: embedding_matrix is only meaningful for the ORIGINAL data here;
#   permuted embedding always uses permuted_embedding_fn (or RunTSNE/UMAP).
# -----------------------------------------------------------------------
optimize <- function(input_data, input_data.permuted, pre_embedding, reduction.method, K,
                     n, m, perplexity, results.PCA, similarity_percent,
                     dubious_cutoff, trustworthy_cutoff,
                     check_duplicates      = TRUE,
                     rerun                 = TRUE,
                     embedding_fn          = NULL,
                     permuted_embedding_fn = NULL,
                     align_with_python     = FALSE) {

  if (reduction.method == "umap") {
    results <- Distances.UMAP(
      pbmc                  = input_data,
      pbmc.permuted         = input_data.permuted,
      K                     = K,
      pre_embedding         = pre_embedding,
      n                     = n,
      m                     = m,
      rerun                 = rerun,
      embedding_fn          = embedding_fn,
      permuted_embedding_fn = permuted_embedding_fn,
      align_with_python     = align_with_python
    )
  } else if (reduction.method == "tsne") {
    results <- Distances.tSNE(
      pbmc                  = input_data,
      pbmc.permuted         = input_data.permuted,
      K                     = K,
      perplexity_score      = perplexity,
      pre_embedding         = pre_embedding,
      check_duplicates      = check_duplicates,
      rerun                 = rerun,
      embedding_fn          = embedding_fn,
      permuted_embedding_fn = permuted_embedding_fn,
      align_with_python     = align_with_python
    )
  } else {
    stop("Unknown reduction.method: ", reduction.method)
  }

  similarity_score <- Cell.Similarity(
    results.PCA$pre_embedding_distances,
    results.PCA$pre_embedding_distances_permuted,
    results$reduced_dim_distances,
    results$reduced_dim_distances_permuted,
    similarity_percent
  )

  ClassifiedCells <- Cell.Classify(
    similarity_score$rho_original,
    similarity_score$rho_permuted,
    dubious_cutoff     = dubious_cutoff,
    trustworthy_cutoff = trustworthy_cutoff
  )

  dub   <- ifelse(length(ClassifiedCells$dubious_cells)      != 0,
                  paste(ClassifiedCells$dubious_cells,      collapse = ","), "none")
  int   <- ifelse(length(ClassifiedCells$intermediate_cells) != 0,
                  paste(ClassifiedCells$intermediate_cells, collapse = ","), "none")
  trust <- ifelse(length(ClassifiedCells$trustworthy_cells)  != 0,
                  paste(ClassifiedCells$trustworthy_cells,  collapse = ","), "none")

  c(length(ClassifiedCells$dubious_cells), dub, trust, int)
}


# -----------------------------------------------------------------------
# scDEED  (main entry point)
#
# NEW arguments:
#   embedding_matrix      – cells x 2 numeric matrix.
#                           When supplied, this is used as-is for the
#                           ORIGINAL data's embedding (no RunTSNE/RunUMAP).
#                           The hyperparameter grid (perplexity / n_neighbors)
#                           still drives the PERMUTED null distribution.
#
#   permuted_embedding_fn – A function with signature:
#                             for tSNE:  fn(seurat_obj, perplexity) -> matrix
#                             for UMAP:  fn(seurat_obj, n_neighbors, min_dist) -> matrix
#                           When supplied, the permuted data embedding is
#                           computed by calling this function rather than
#                           Seurat's RunTSNE / RunUMAP.
#                           Use this to plug in reticulate + OpenTSNE / umap-learn.
#
#   align_with_python     – logical (default FALSE).
#                           When TRUE and neither embedding_matrix nor
#                           permuted_embedding_fn is provided, adjusts
#                           RunTSNE / RunUMAP parameters to match the
#                           defaults used by OpenTSNE / Python umap-learn
#                           as closely as possible.
# -----------------------------------------------------------------------
scDEED <- function(input_data, K,
                   n_neighbors           = c(5, 20, 30, 40, 50),
                   min.dist              = c(0.1, 0.4),
                   similarity_percent    = 0.5,
                   reduction.method,
                   perplexity            = c(seq(from = 20, to = 410, by = 30),
                                             seq(from = 450, to = 800, by = 50)),
                   pre_embedding         = "pca",
                   layer                 = NULL,
                   slot                  = "scale.data",
                   dubious_cutoff        = 0.05,
                   trustworthy_cutoff    = 0.95,
                   permuted              = NA,
                   check_duplicates      = TRUE,
                   rerun                 = TRUE,
                   default_assay         = "active.assay",
                   # ---- new ----
                   embedding_fn          = NULL,   # fn(perplexity) or fn(n_neighbors, min_dist) -> matrix
                   permuted_embedding_fn = NULL,
                   align_with_python     = FALSE) {

  if (is.null(layer) || !nzchar(layer)) layer <- slot

  # ---- Validate embedding_fn ----
  if (!is.null(embedding_fn) && !is.function(embedding_fn)) {
    stop("`embedding_fn` must be a function, e.g. function(perplexity) -> matrix")
  }

  # ---- Permuted object ----
  if (pre_embedding != "pca" & is.na(permuted)) {
    stop(paste0(
      "scDEED does not know how to calculate ", pre_embedding, ". ",
      "Please calculate this pre-embedding on the permuted data and provide ",
      "the object in the `permuted` argument."
    ))
  }

  if (is.na(permuted)) {
    print("Permuting data")
    input_data.permuted <- suppressMessages(
      Permuted(input_data, K = K, slot = slot, default_assay = default_assay)
    )
    print("Permutation finished")
  } else {
    input_data.permuted <- permuted
  }

  # ---- Pre-embedding distances ----
  results.PCA <- suppressMessages(
    Distances.pre_embedding(input_data, input_data.permuted,
                            K = K, pre_embedding = pre_embedding)
  )

  # ------------------------------------------------------------------
  # UMAP branch
  # ------------------------------------------------------------------
  if (tolower(reduction.method) == "umap") {
    all_pairs <- expand.grid(n_neighbors, min.dist)

    .run_one_umap <- function(n, m) {
      optimize(input_data, input_data.permuted, pre_embedding, "umap", K,
               n = n, m = m, perplexity = NA, results.PCA = results.PCA,
               similarity_percent    = similarity_percent,
               dubious_cutoff        = dubious_cutoff,
               trustworthy_cutoff    = trustworthy_cutoff,
               rerun                 = rerun,
               embedding_fn          = embedding_fn,
               permuted_embedding_fn = permuted_embedding_fn,
               align_with_python     = align_with_python)
    }

    if (nrow(all_pairs) > 1) {
      print("Estimating time for each hyperparameter setting...")
      start    <- Sys.time()
      original <- suppressMessages(
        foreach::`%do%`(
          foreach::foreach(n = all_pairs$Var1[1], m = all_pairs$Var2[1], .combine = "cbind"),
          .run_one_umap(n, m)
        )
      )
      end  <- Sys.time()
      time <- end - start
      print("Estimated time per hyperparameter setting:"); print(time)
      print("Estimated time of completion:"); print(Sys.time() + time * nrow(all_pairs))
      original <- as.matrix(original)

    }

    # Run all pairs in one loop (avoids [-1] negative-index bug)
    all_dub <- suppressMessages(
      foreach::`%do%`(
        foreach::foreach(n = all_pairs$Var1, m = all_pairs$Var2, .combine = "cbind"),
        .run_one_umap(n, m)
      )
    )
    all_dub <- as.matrix(all_dub)

    all_dub <- t(all_dub)
    colnames(all_dub)  <- c("number_dubious_cells", "dubious_cells",
                             "trustworthy_cells", "intermediate_cells")
    colnames(all_pairs) <- c("n_neighbors", "min.dist")

    dubious_number_UMAP <- cbind(all_pairs, all_dub)
    dub_para            <- data.frame(dubious_number_UMAP[, 1:3])
    dub_para_full       <- as.data.frame(dubious_number_UMAP)
  }

  # ------------------------------------------------------------------
  # tSNE branch
  # ------------------------------------------------------------------
  if (tolower(reduction.method) == "tsne") {

    # If no embedding_fn and tsne reduction is absent, seed one
    if (is.null(embedding_fn) && is.null(input_data@reductions$tsne)) {
      input_data <- Seurat::RunTSNE(input_data,
                                    do.fast          = TRUE,
                                    check_duplicates = check_duplicates)
    }

    .run_one_tsne <- function(p) {
      optimize(input_data, input_data.permuted, pre_embedding, "tsne", K,
               n = NA, m = NA, perplexity = p, results.PCA = results.PCA,
               similarity_percent    = similarity_percent,
               dubious_cutoff        = dubious_cutoff,
               trustworthy_cutoff    = trustworthy_cutoff,
               check_duplicates      = check_duplicates,
               rerun                 = rerun,
               embedding_fn          = embedding_fn,
               permuted_embedding_fn = permuted_embedding_fn,
               align_with_python     = align_with_python)
    }

    if (length(perplexity) > 1) {
      print("Estimating time for each hyperparameter setting...")
      start <- Sys.time()
      .run_one_tsne(perplexity[[1]])
      time  <- Sys.time() - start
      print("Estimated time per hyperparameter setting:"); print(time)
      print("Estimated time of completion:"); print(Sys.time() + time * length(perplexity))
    }

    # Single loop over all perplexities - avoids perplexity[-1] negative-index bug
    dubious_number_tSNE <- suppressMessages(
      foreach::`%do%`(
        foreach::foreach(p = perplexity, .combine = "cbind"),
        .run_one_tsne(p)
      )
    )
    dubious_number_tSNE <- as.matrix(dubious_number_tSNE)

    all_dub <- t(dubious_number_tSNE)
    colnames(all_dub) <- c("number_dubious_cells", "dubious_cells",
                            "trustworthy_cells", "intermediate_cells")

    dubious_number_tsne         <- cbind(perplexity, all_dub)
    colnames(dubious_number_tsne)[1] <- "perplexity"
    dub_para                    <- data.frame(dubious_number_tsne[, 1:2])
    if (length(dub_para) == 1) dub_para <- as.data.frame(t(dub_para))
    dub_para_full               <- as.data.frame(dubious_number_tsne)
  }

  dub_para$number_dubious_cells      <- as.numeric(dub_para$number_dubious_cells)
  dub_para_full$number_dubious_cells <- as.numeric(dub_para_full$number_dubious_cells)
  rownames(dub_para)      <- NULL
  rownames(dub_para_full) <- NULL

  list(num_dubious = dub_para, full_results = dub_para_full)
}
