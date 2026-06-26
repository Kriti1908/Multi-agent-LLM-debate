#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 16                  # total CPU cores
#SBATCH --gres=gpu:4           # request 4 GPUs
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.lavanya@students.iiit.ac.in  # Email for notifications

python3 analyse2.py