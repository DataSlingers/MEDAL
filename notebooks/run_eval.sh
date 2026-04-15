#!/bin/bash
#SBATCH -J medal_eval
#SBATCH -p ctn
#SBATCH -N 1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err

module load cuda 
module load anaconda3-2023.07
conda activate medal


PYTHONPATH=/home/bnc2119/MEDAL:$PYTHONPATH python -u tasic_tuning_eval.py
