#!/bin/bash
set -euo pipefail

# Submit each method as its own Slurm job
sbatch run_pcs.sbatch
sbatch run_scdeed.sbatch
#sbatch run_embedr_umap.sbatch
#sbatch run_embedr_tsne.sbatch
