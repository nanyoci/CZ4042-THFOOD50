#!/bin/bash
#SBATCH --job-name=CZ4042_Project
#SBATCH --output=out/Original_Output.out
#SBATCH --error=out/Original_Error.err
#SBATCH --nodes=1
#SBATCH --partition=SCSEGPU_UG
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

module load anaconda
source activate ass2_vit
python transfer_learning.py --pretrain original --epochs 10
