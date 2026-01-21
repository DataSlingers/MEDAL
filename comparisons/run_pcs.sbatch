#!/bin/bash
#SBATCH -p general
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=160g
#SBATCH --job-name=pcs
#SBATCH -t 96:00:00
#SBATCH --output=pcs_%j.out
#SBATCH --error=pcs_%j.err
#SBATCH --chdir=/nas/longleaf/home/tarek/MEDAL_Comparisons
module purge
module add r/4.4.0

export R_LIBS_USER="$HOME/R/x86_64-pc-linux-gnu-library/4.4"
mkdir -p "$R_LIBS_USER"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"
Rscript --vanilla pcs/pcs.R
