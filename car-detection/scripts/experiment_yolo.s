#!/bin/bash
#SBATCH --job-name=ssl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
module purge
source /scratch/bz1030/capstone_env/bin/activate
cd ..
python main.py -mc ./config/yolov3.cfg