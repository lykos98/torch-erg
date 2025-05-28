#!/bin/bash

#SBATCH -p lovelace
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --job-name=erg_test
#SBATCH -o run_erg_GWGensemble.out
#SBATCH -e run_erg_GWGensemble.err

module load miniconda  # or module load miniconda


# Activate the conda environment
conda activate cert  # change to your environment

# Navigate to the directory with your script
cd /u/f_giacomarra/repos/torch-erg # change to your directory



python ensembler.py