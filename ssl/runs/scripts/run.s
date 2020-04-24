#!/bin/sh
#SBATCH --job-name=ssl
#SBATCH --nodes=1
#SBATCH --gres=gpu:k80:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB
#SBATCH --time=8:00:00
#SBATCH --mail-user=cc5048@nyu.edu
#SBATCH --mail-type=ALL

module purge

module load cuda/10.0.130
module load cudnn/9.0v7.0.5
module load anaconda3/4.3.1
source activate py36
cd ../cpc



python run.py

