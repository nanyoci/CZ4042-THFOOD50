#!/bin/bash
#SBATCH --job-name=cz4042
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=Output_v1.out 
#SBATCH --error=Error_v1.err 
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --mem=128000M
#SBATCH --gres=gpu:1

module load anaconda
source activate /apps/conda_env/CZ4042_v2
python modified_v1.py