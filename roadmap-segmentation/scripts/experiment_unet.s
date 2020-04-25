#!/bin/bash
#SBATCH --job-name=ssl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:p40:1
module purge
source /scratch/bz1030/capstone_env/bin/activate
cd ..
python main.py -bs 4