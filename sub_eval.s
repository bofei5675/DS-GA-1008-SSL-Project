#!/bin/bash
#SBATCH --job-name=ssl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=./eval_%j.out
#SBATCH --error=./eval_%j.err

module purge
source /scratch/bz1030/capstone_env/bin/activate

python run_test.py --det_model /scratch/bz1030/data_ds_1008/detection/car_detection/runs/p2v_yolo_2020-04-30_12-20-49_det/best-model-1.pth \
--seg_model /scratch/bz1030/data_ds_1008/detection/roadmap_segmentation/runs/pix2vox_2020-04-26_00-29-58/best-model-9.pth