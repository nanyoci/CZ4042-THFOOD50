#!/bin/bash
#SBATCH --job-name=CZ4042_Project
#SBATCH --output=out/NoPretrain_Output.out
#SBATCH --error=out/NoPretrain_Error.err
#SBATCH --nodes=1
#SBATCH --partition=SCSEGPU_UG
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

module load anaconda
source activate ass2_vit
python transfer_learning.py --epochs 10
