#!/bin/bash

#SBATCH -p lovelace
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --job-name=GWG_erg_test
#SBATCH -o ebm_results.out
#SBATCH -e ebm_results.err

module load miniconda  # or module load miniconda


# Activate the conda environment
conda activate cert  # or conda activate scoreenv if your cluster supports it directly

# Navigate to the directory with your script
cd /u/f_giacomarra/repos/git_erg/torch-erg



python deep_ebm/exec_ebm.py