#!/bin/bash
#SBATCH --job-name=ssl_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:k80:1
#SBATCH --output=./eval_%j.out
#SBATCH --error=./eval_%j.err

module purge
source /scratch/bz1030/capstone_env/bin/activate

python run_test_submission.py --det_model /scratch/bz1030/data_ds_1008/detection/car_detection/runs/yolov3_p2v_yolo_2020-05-06_05-38-06_det_pt/best-model-15.pth \
--seg_model /scratch/bz1030/data_ds_1008/detection/car_detection/runs/yolov3_p2v_yolo_2020-05-05_20-52-38_seg_pt/best-model-7.pth