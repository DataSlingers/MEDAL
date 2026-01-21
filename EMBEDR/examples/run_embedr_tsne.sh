#!/bin/bash
#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem=160g
#SBATCH --cpus-per-task=64
#SBATCH -t 120:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

module purge
module load anaconda

# DEBUG: show which python is used + whether EMBEDR is importable
conda run -n embedr python -c "import sys; print('PY:', sys.executable); import EMBEDR; print('EMBEDR:', EMBEDR.__file__)"

# Run your script using the env explicitly
conda run -n embedr python /nas/longleaf/home/tarek/Irene/embedr/EMBEDR/examples/embedr_results_tsne.py
