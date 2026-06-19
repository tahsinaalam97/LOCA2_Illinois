#!/bin/bash
#SBATCH -p node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=LOCA2_IL
#SBATCH --output=/data/keeling/a/tahsina2/b/logs/LOCA2_IL_%A_%a.out
#SBATCH --error=/data/keeling/a/tahsina2/b/logs/LOCA2_IL_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tahsina2@illinois.edu

source /data/keeling/a/tahsina2/miniconda3/bin/activate
conda activate pyWBM

mkdir -p /data/keeling/a/tahsina2/b/logs

python /data/keeling/a/tahsina2/b/download_loca2_il.py \
    --task-id ${SLURM_ARRAY_TASK_ID}