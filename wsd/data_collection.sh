#!/bin/bash
# ====================
# Options for sbatch
# ====================

# Maximum number of nodes to use for the job
#SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:1

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=14000

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=1

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=9:00:00

source ~/.bashrc
conda activate pt

python translate_wsd.py --source wsd_train.tsv --pred wsd_train_pred.txt
wait

