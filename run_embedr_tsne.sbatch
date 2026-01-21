#!/bin/bash
#SBATCH -p general
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=160g
#SBATCH --job-name=embedr_tsne
#SBATCH -t 96:00:00
#SBATCH --output=embedr_tsne_%j.out
#SBATCH --error=embedr_tsne_%j.err
#SBATCH --chdir=/nas/longleaf/home/tarek/MEDAL_Comparisons
module purge
module load anaconda

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

conda run -n embedr python -c "import sys; print('PY:', sys.executable); import EMBEDR; print('EMBEDR:', EMBEDR.__file__)"
cd "$SLURM_SUBMIT_DIR"
conda run -n embedr python EMBEDR/examples/embedr_results_umap.py
