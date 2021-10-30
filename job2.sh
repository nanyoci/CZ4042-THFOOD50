#!/bin/bash
#SBATCH --job-name=cz4042
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=Output2.out 
#SBATCH --error=Error2.err 
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --mem=128000M
#SBATCH --gres=gpu:1

python deep_thfood50.py

