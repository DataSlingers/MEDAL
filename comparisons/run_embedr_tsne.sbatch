#!/bin/bash
#SBATCH -p general
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160g
#SBATCH --job-name=embedr_tsne
#SBATCH -t 96:00:00
#SBATCH --output=logs/embedr_%j.out
#SBATCH --error=logs/embedr_%j.err
module purge
module load anaconda3-2023.07
conda activate embedr_test

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -c "import sys; print('PY:', sys.executable); import EMBEDR; print('EMBEDR:', EMBEDR.__file__)"
cd "$SLURM_SUBMIT_DIR"
python -u EMBEDR/examples/embedr_results_tsne.py
