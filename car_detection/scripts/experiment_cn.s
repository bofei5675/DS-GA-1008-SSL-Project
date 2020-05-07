#!/bin/bash
#SBATCH --job-name=ssl_cn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
module purge
source /scratch/bz1030/capstone_env/bin/activate
cd ..
python main.py -mc center_net -bs 2 -pt no -det yes -seg no -ssl yes -a 6 -g 1



