#!/bin/bash
#SBATCH -p general
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=160g
#SBATCH --job-name=pcs
#SBATCH -t 96:00:00
#SBATCH --output=logs/pcs_%j.out
#SBATCH --error=logs/pcs_%j.err
#SBATCH --chdir=/share/ctn/users/bnc2119/MEDAL/comparisons
module load anaconda3-2023.07
module purge
module load Miniforge-24.7.1-2
conda activate medalcomp
conda install -c conda-forge r-base=4.4 r-essentials -y
#module add r/4.4.0

export R_LIBS_USER="$HOME/R/x86_64-pc-linux-gnu-library/4.4"
mkdir -p "$R_LIBS_USER"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"
stdbuf -oL -eL Rscript --vanilla pcs/pcs.R
