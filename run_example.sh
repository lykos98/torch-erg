#!/bin/bash

#SBATCH -p lovelace
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --job-name=GWG_erg_test
#SBATCH -o run_erg_GWGexample.out
#SBATCH -e run_erg_GWGexample.err

module load miniconda  # or module load miniconda


# Activate the conda environment
conda activate cert  # or conda activate scoreenv if your cluster supports it directly

# Navigate to the directory with your script
cd /u/f_giacomarra/repos/torch-erg



python example.py