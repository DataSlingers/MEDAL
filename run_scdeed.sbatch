#!/bin/bash
#SBATCH -p general
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=160g
#SBATCH --job-name=scdeed
#SBATCH -t 96:00:00
#SBATCH --output=scdeed_%j.out
#SBATCH --error=scdeed_%j.err
#SBATCH --chdir=/nas/longleaf/home/tarek/MEDAL_Comparisons

module purge
module add r/4.4.0

# ------------------------------------------------------------
# Use a Seurat-v4-only library so scDEED's slot= calls work
# (SeuratObject >=5 makes slot= defunct -> your current error)
# ------------------------------------------------------------
export R_LIBS_USER="$HOME/R/seurat4_r44"
mkdir -p "$R_LIBS_USER"

# Optional but helps prevent BLAS/OpenMP thread explosions
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Always run from the submission directory (or the #SBATCH --chdir above)
cd "$SLURM_SUBMIT_DIR"

# ------------------------------------------------------------
# One-time (per library) install of Seurat/SeuratObject v4
# Will NO-OP after they are installed in $R_LIBS_USER
# ------------------------------------------------------------
Rscript --vanilla -e '
options(repos = c(CRAN="https://cloud.r-project.org"))
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))

if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes", quiet=TRUE)

need <- function(pkg, min_version=NULL, max_version=NULL) {
  if (!requireNamespace(pkg, quietly=TRUE)) return(TRUE)
  v <- as.character(utils::packageVersion(pkg))
  if (!is.null(min_version) && utils::packageVersion(pkg) < min_version) return(TRUE)
  if (!is.null(max_version) && utils::packageVersion(pkg) >= max_version) return(TRUE)
  FALSE
}

# Force SeuratObject < 5 and Seurat 4.x
if (need("SeuratObject", max_version="5.0.0")) {
  remotes::install_version("SeuratObject", version="4.1.4", dependencies=TRUE, quiet=TRUE)
}
if (need("Seurat", max_version="5.0.0")) {
  remotes::install_version("Seurat", version="4.4.0", dependencies=TRUE, quiet=TRUE)
}

cat("LIBPATHS:\n"); print(.libPaths())
cat("SeuratObject:", as.character(packageVersion("SeuratObject")), "\n")
cat("Seurat      :", as.character(packageVersion("Seurat")), "\n")
'

# ------------------------------------------------------------
# Run scDEED script (now under Seurat v4 API)
# ------------------------------------------------------------
Rscript --vanilla scDEED/scdeed_compare.R