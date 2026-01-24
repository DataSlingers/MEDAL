## ============================================================
## OOM-safer cluster driver (streaming writes)  [UPDATED AGAIN]
## Key changes to stop OOM on large n:
##   - FORCE BLAS/OpenMP threads = 1 (avoid hidden thread fan-out)
##   - pscore cores = 1 for n >= 10k (avoid forked-worker memory spikes)
##   - Barnes–Hut theta for large n
##   - checkpoint messages to pinpoint where OOM occurs
## ============================================================

#suppressPackageStartupMessages({
#  if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr", repos="https://cloud.r-project.org")
#  if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools", repos="https://cloud.r-project.org")
  
  #devtools::install_github("zhexuandliu/RtsneWithP", upgrade = "never")
  #devtools::install_github("zhexuandliu/neMDBD", upgrade = "never")
  
#  library(dplyr)
#  library(neMDBD)
#  library(RtsneWithP)
#})


suppressPackageStartupMessages({
  # Ensure installs go to a user-writable library
  userlib <- Sys.getenv("R_LIBS_USER")
  if (nzchar(userlib)) {
    dir.create(userlib, recursive = TRUE, showWarnings = FALSE)
    .libPaths(c(userlib, .libPaths()))
  }

  # CRAN deps
  if (!requireNamespace("dplyr", quietly = TRUE))
    install.packages("dplyr", repos="https://cloud.r-project.org")

  # Use remotes instead of devtools (much lighter; fewer system deps)
  if (!requireNamespace("remotes", quietly = TRUE))
    install.packages("remotes", repos="https://cloud.r-project.org")

  # Install GitHub packages only if missing
  if (!requireNamespace("RtsneWithP", quietly = TRUE))
    remotes::install_github("zhexuandliu/RtsneWithP", upgrade = "never")

  if (!requireNamespace("neMDBD", quietly = TRUE))
    remotes::install_github("zhexuandliu/neMDBD", upgrade = "never")

  library(dplyr)
  library(neMDBD)
  library(RtsneWithP)
})


set.seed(1453)

## ----------------------------
## 0) CPU/threads settings
## ----------------------------
# CPUs allocated to the job (informational for logging)
cores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "10"))
if (!is.finite(cores) || cores <= 0) cores <- 10
message("SLURM_CPUS_PER_TASK = ", cores)

# CRITICAL: prevent hidden multi-threading (BLAS/OpenMP) from exploding memory
# Use 1 here. Your explicit parallelism is inside perturbation_score_compute.
Sys.setenv(
  OMP_NUM_THREADS = 1,
  OPENBLAS_NUM_THREADS = 1,
  MKL_NUM_THREADS = 1,
  VECLIB_MAXIMUM_THREADS = 1,
  NUMEXPR_NUM_THREADS = 1
)
message("Set BLAS/OpenMP thread env vars to 1")

## ----------------------------
## 1) Helpers
## ----------------------------
get_perplexity_list <- function(df_name) {
  df_name <- toupper(trimws(df_name))
  if (df_name == "MNIST") return(c(5, 11, 27, 62, 146, 341, 793, 1846))
  if (df_name == "HYDRA") return(c(5, 10, 23, 49, 107, 232, 499, 1077, 2320, 4999))
  if (df_name == "TASIC") return(c(5, 10, 24, 53, 116, 256, 564, 1241, 2729, 6000))
  if (df_name == "ASTRO") return(c(3, 4, 6, 8, 12, 18, 26, 55, 80, 115, 167, 240, 346, 499))
  #if (df_name == "ASTRO") return(c(3, 4))
  stop("Unknown df_name: ", df_name)
}

top_tail_mean <- function(v, frac = 0.05) {
  v <- v[is.finite(v)]
  if (!length(v)) return(NA_real_)
  k <- max(1L, ceiling(frac * length(v)))
  mean(sort(v, decreasing = TRUE)[seq_len(k)])
}

find_elbow_triangle <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]; y <- y[ok]
  if (!length(x)) return(NA_real_)
  if (length(unique(x)) == 1 || length(unique(y)) == 1) return(x[1])
  
  o <- order(x); x <- x[o]; y <- y[o]
  xs <- (x - min(x)) / (max(x) - min(x))
  ys <- (y - min(y)) / (max(y) - min(y))
  
  x1 <- xs[1]; y1 <- ys[1]
  x2 <- xs[length(xs)]; y2 <- ys[length(ys)]
  denom <- sqrt((y2 - y1)^2 + (x2 - x1)^2)
  if (denom == 0) return(x[1])
  
  dist <- abs((y2 - y1) * xs - (x2 - x1) * ys + x2*y1 - y2*x1) / denom
  x[which.max(dist)]
}

sanitize_colnames <- function(nm) {
  nm <- as.character(nm)
  bad <- is.na(nm) | trimws(nm) == ""
  if (any(bad)) nm[bad] <- paste0("V", which(bad))
  make.unique(make.names(nm), sep = "_")
}

# Barnes–Hut for large n (lower memory than exact)
theta_for_n <- function(n) {
  if (n >= 5000) return(0.5)
  return(0.0)
}

# IMPORTANT: avoid parallel fork memory spikes for large n
pscore_cores_for_n <- function(n, cores) {
  if (n >= 10000) return(1L)         # <-- key OOM fix
  if (n >= 5000)  return(min(cores, 4L))
  return(min(cores, 8L))
}

## ----------------------------
## 2) Core runner for ONE dataset
## ----------------------------
run_one_dataset <- function(csv_path,
                            out_dir,
                            cores = 10,
                            max_iter = 1000,
                            top_frac = 0.05,
                            length_param = 0.5,
                            approx = 1) {
  
  base <- basename(csv_path)
  stem <- sub("\\.csv$", "", base)
  df_name <- toupper(sub("_.*$", "", stem))
  
  message("\n==============================")
  message("Dataset file: ", base)
  message("Dataset key : ", df_name)
  message("==============================")
  
  dataset_out <- file.path(out_dir, stem)
  dir.create(dataset_out, showWarnings = FALSE, recursive = TRUE)
  
  X <- read.csv(csv_path, check.names = FALSE)
  names(X) <- sanitize_colnames(names(X))
  
  nm <- names(X)
  keep <- !(tolower(nm) %in% c("label", "split"))
  dropped <- nm[!keep]
  if (length(dropped)) message("Dropped label/split columns: ", paste(dropped, collapse=", "))
  X <- X[, keep, drop=FALSE]
  
  is_num <- vapply(X, is.numeric, logical(1))
  if (any(!is_num)) {
    nonnum <- names(X)[!is_num]
    message("Dropping non-numeric columns: ", paste(nonnum, collapse=", "))
    X <- X[, is_num, drop=FALSE]
  }
  
  X_mat <- as.matrix(X)
  n <- nrow(X_mat)
  
  perplexity_list <- get_perplexity_list(df_name)
  max_safe <- (n - 1) / 3
  perplexity_list <- perplexity_list[perplexity_list < max_safe]
  if (!length(perplexity_list)) stop("No valid perplexities after filtering by (n-1)/3. n=", n)
  
  message("n = ", n, ", p = ", ncol(X_mat))
  message("Using perplexities: ", paste(perplexity_list, collapse=", "))
  
  theta_use <- theta_for_n(n)
  pscore_cores <- pscore_cores_for_n(n, cores)
  message("theta_use = ", theta_use, " | pscore_cores = ", pscore_cores)
  
  # Output file paths
  scores_path <- file.path(dataset_out, "scores_per_point_all_perplexities.csv")
  elbow_path  <- file.path(dataset_out, "elbow_df.csv")
  xout_path   <- file.path(dataset_out, "X_with_best_scores.csv")
  
  # (re)start scores file with header
  write.csv(
    data.frame(perplexity=integer(), point_id=integer(), pscore=double(), sscore=double()),
    scores_path, row.names=FALSE
  )
  
  summary_rows <- vector("list", length(perplexity_list))
  
  best_mean_pscore <- Inf; best_mean_pscore_perp <- NA; best_mean_pscore_vec <- NULL
  best_mean_sscore <- Inf; best_mean_sscore_perp <- NA; best_mean_sscore_vec <- NULL
  
  for (i in seq_along(perplexity_list)) {
    perp <- perplexity_list[i]
    message("  -> perplexity = ", perp, " (", i, "/", length(perplexity_list), ")")
    
    message("     [tSNE] starting")
    tsne_out <- RtsneWithP::Rtsne(
      X_mat,
      perplexity = perp,
      theta = theta_use,
      max_iter = max_iter
    )
    message("     [tSNE] done  | dim(Y) = ", paste(dim(tsne_out$Y), collapse="x"))
    
    message("     [pscore] starting (no.cores = ", pscore_cores, ")")
    t0 <- proc.time()
    pscore <- perturbation_score_compute(
      X_mat, tsne_out$Y, perp,
      length = length_param,
      approx = approx,
      no.cores = pscore_cores
    )
    #pscore <- rep(0, n)
    rt_p <- as.numeric((proc.time() - t0)[["elapsed"]])
    message("     [pscore] done  | elapsed = ", rt_p, " sec")
    
    message("     [sscore] starting")
    t1 <- proc.time()
    sscore <- singularity_score_compute(tsne_out$Y, tsne_out$P)
    rt_s <- as.numeric((proc.time() - t1)[["elapsed"]])
    message("     [sscore] done  | elapsed = ", rt_s, " sec")
    
    # write per-point immediately
    point_df <- data.frame(
      perplexity = perp,
      point_id   = seq_len(n),
      pscore     = as.numeric(pscore),
      sscore     = as.numeric(sscore)
    )
    write.table(point_df, scores_path, sep=",", row.names=FALSE, col.names=FALSE, append=TRUE)
    
    # summary stats
    mean_p <- mean(point_df$pscore, na.rm=TRUE)
    mean_s <- mean(point_df$sscore, na.rm=TRUE)
    top5_p <- top_tail_mean(point_df$pscore, frac=top_frac)
    top5_s <- top_tail_mean(point_df$sscore, frac=top_frac)
    
    summary_rows[[i]] <- data.frame(
      perplexity = perp,
      mean_pscore = mean_p,
      mean_sscore = mean_s,
      top5_mean_pscore = top5_p,
      top5_mean_sscore = top5_s,
      runtime_pscore_sec = rt_p,
      runtime_sscore_sec = rt_s
    )
    
    # update best-by-mean vectors (keep only best in memory)
    if (is.finite(mean_p) && mean_p < best_mean_pscore) {
      best_mean_pscore <- mean_p
      best_mean_pscore_perp <- perp
      best_mean_pscore_vec <- point_df$pscore
    }
    if (is.finite(mean_s) && mean_s < best_mean_sscore) {
      best_mean_sscore <- mean_s
      best_mean_sscore_perp <- perp
      best_mean_sscore_vec <- point_df$sscore
    }
    
    # free big objects ASAP
    rm(tsne_out, pscore, sscore, point_df)
    gc(verbose = FALSE)
  }
  
  elbow_df <- dplyr::bind_rows(summary_rows) %>% arrange(perplexity)
  
  # select elbow(top5) and meanbest
  p_elbow_top5 <- find_elbow_triangle(elbow_df$perplexity, elbow_df$top5_mean_pscore)
  s_elbow_top5 <- find_elbow_triangle(elbow_df$perplexity, elbow_df$top5_mean_sscore)
  
  p_best_mean <- best_mean_pscore_perp
  s_best_mean <- best_mean_sscore_perp
  
  message("Chosen pscore elbow(top5) = ", p_elbow_top5, " | pscore best(mean) = ", p_best_mean)
  message("Chosen sscore elbow(top5) = ", s_elbow_top5, " | pscore best(mean) = ", s_best_mean)
  
  # helper: read one vector back from scores CSV for a specific perplexity
  read_metric_vec <- function(perp_value, metric=c("pscore","sscore")) {
    metric <- match.arg(metric)
    df <- read.csv(scores_path)
    sub <- df[df$perplexity == perp_value, ]
    sub <- sub[order(sub$point_id), ]
    sub[[metric]]
  }
  
  # only read back elbow vectors (cheap)
  p_elbow_vec <- read_metric_vec(p_elbow_top5, "pscore")
  s_elbow_vec <- read_metric_vec(s_elbow_top5, "sscore")
  
  X_out <- as.data.frame(X_mat)
  X_out[[paste0("pscore_top5elbow_", p_elbow_top5)]] <- p_elbow_vec
  X_out[[paste0("sscore_top5elbow_", s_elbow_top5)]] <- s_elbow_vec
  X_out[[paste0("pscore_meanbest_", p_best_mean)]]   <- best_mean_pscore_vec
  X_out[[paste0("sscore_meanbest_", s_best_mean)]]   <- best_mean_sscore_vec
  
  X_out$pscore_top5elbow_perplexity <- p_elbow_top5
  X_out$sscore_top5elbow_perplexity <- s_elbow_top5
  X_out$pscore_meanbest_perplexity  <- p_best_mean
  X_out$sscore_meanbest_perplexity  <- s_best_mean
  
  write.csv(elbow_df, elbow_path, row.names=FALSE)
  write.csv(X_out, xout_path, row.names=FALSE)
  
  invisible(list(
    dataset = stem,
    df_name = df_name,
    out_dir = dataset_out,
    elbows = list(
      pscore_elbow_top5 = p_elbow_top5,
      sscore_elbow_top5 = s_elbow_top5,
      pscore_best_mean = p_best_mean,
      sscore_best_mean = s_best_mean
    )
  ))
}

## ----------------------------
## 3) Driver: loop over all *_train.csv
## ----------------------------
input_dir <- "../data2"

## CHANGED: write everything under results_pcs/
out_dir <- "results_pcs"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

csv_files <- list.files(input_dir, pattern = "_train\\.csv$", full.names = TRUE, ignore.case = TRUE)
if (!length(csv_files)) stop("No *_train.csv files found in input_dir: ", input_dir)

message("Found ", length(csv_files), " dataset(s) in ", normalizePath(input_dir))

results <- vector("list", length(csv_files))
for (i in seq_along(csv_files)) {
  results[[i]] <- run_one_dataset(csv_files[i], out_dir=out_dir, cores=cores)
}

run_log <- do.call(rbind, lapply(results, function(x) {
  data.frame(
    dataset = x$dataset,
    df_name = x$df_name,
    pscore_elbow_top5 = x$elbows$pscore_elbow_top5,
    sscore_elbow_top5 = x$elbows$sscore_elbow_top5,
    pscore_best_mean  = x$elbows$pscore_best_mean,
    sscore_best_mean  = x$elbows$sscore_best_mean,
    out_dir = x$out_dir,
    stringsAsFactors = FALSE
  )
}))
write.csv(run_log, file.path(out_dir, "run_log.csv"), row.names=FALSE)
message("Wrote run log: ", normalizePath(file.path(out_dir, "run_log.csv")))
