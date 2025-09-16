#!/bin/bash

#SBATCH -p lovelace
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --job-name=erg_test
#SBATCH -o run_ensemble.out
#SBATCH -e run_ensemble.err

module load miniconda  # or module load miniconda


# Activate the conda environment
conda activate cert  # change to your environment

# Navigate to the directory with your script
cd /u/f_giacomarra/repos/git_erg/torch-erg



python ensembler.py