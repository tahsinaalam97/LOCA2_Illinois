#!/bin/bash
#SBATCH --job-name=st_pierre_thi
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --chdir=/data/keeling/a/tahsina2/b/Climate_Impact/Notebooks
#SBATCH --output=/data/keeling/a/tahsina2/b/Climate_Impact/Notebooks/st_pierre_%j.out
#SBATCH --error=/data/keeling/a/tahsina2/b/Climate_Impact/Notebooks/st_pierre_%j.err

set -eo pipefail

source /data/keeling/a/tahsina2/miniconda3/etc/profile.d/conda.sh
conda activate pyWBM
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SCRIPT_DIR=/data/keeling/a/tahsina2/b/Climate_Impact/Notebooks
cd "$SCRIPT_DIR"
test -f "$SCRIPT_DIR/St_Pierre.py"
test -f "$SCRIPT_DIR/loca2_common.py"

echo "Starting St Pierre at $(date) on $(hostname)"
echo "Python: $(command -v python)"

# OVERWRITE=False is defined in St_Pierre.py. Extra arguments are optional:
# sbatch St_Pierre.slurm --run-limit 1
# sbatch St_Pierre.slurm --export-csv
srun --ntasks=1 python -u "$SCRIPT_DIR/St_Pierre.py" "$@"

echo "Finished St Pierre at $(date)"
