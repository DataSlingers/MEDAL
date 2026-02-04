#!/bin/bash
#SBATCH -p general
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160g
#SBATCH --job-name=scdeed
#SBATCH -t 96:00:00
#SBATCH --output=logs/scdeed_%j.out
#SBATCH --error=logs/scdeed_%j.err
#SBATCH --chdir=/share/ctn/users/bnc2119/MEDAL/comparisons

module purge
#module add r/4.4.0
module load anaconda3-2023.07
conda activate medalcomp

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

check_ver <- function(pkg, want_prefix) {
  if (!requireNamespace(pkg, quietly=TRUE)) stop("Missing package: ", pkg)
  v <- as.character(packageVersion(pkg))
  if (!startsWith(v, want_prefix)) stop("Wrong version for ", pkg, ": ", v, " (wanted ", want_prefix, "x)")
  cat(pkg, "=", v, "\n")
}

cat("LIBPATHS:\n"); print(.libPaths())

check_ver("SeuratObject", "4.")
check_ver("Seurat", "4.")
'

# ------------------------------------------------------------
# Run scDEED script (now under Seurat v4 API)
# ------------------------------------------------------------
stdbuf -oL -eL Rscript --vanilla scDEED/scdeed_compare.R
