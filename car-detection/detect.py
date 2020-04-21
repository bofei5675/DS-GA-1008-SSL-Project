import argparse
import torch
import numpy as np
from models import *
from utils.utils import *
from utils.datasets import *
from train import setup

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import pandas as pd
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('-mc', '--model-config', dest='model_config', default='./config/yolov3_large.cfg', type=str)
parser.add_argument('-bs', '--batch-size', dest='batch_size', default=4, type=int)
args = parser.parse_args()
_, _, _, valloader = setup(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = args.model_config
weights = '/scratch/bz1030/data_ds_1008/detection/car-detection/runs/yolov3_2020-04-18_07-40-21/best-model-3.pth'
model = Darknet(model_config, img_size=832).to(device)
model.load_darknet_weights(weights)
model.eval()  # Set in evaluation mode
result = []
bar = tqdm(total=len(valloader), desc='Processing', ncols=90)

for idx, (sample, target, road_image, extra) in enumerate(valloader):
    sample = torch.stack(sample)
    sample = sample.to(device)
    with torch.no_grad():
        _, detections, _ = model(sample)
        detections = non_max_suppression(detections, 0.5, 0.3)
    for meta_info, detection in zip(extra, detections):
        if detection is None:
            continue
        detection = detection.numpy().reshape(1, -1)
        detection = ' '.join([str(i) for i in detection.tolist()])
        result.append((meta_info['file_path'], detection))
    bar.update()

result = pd.DataFrame(result, columns=['file', 'annotation'])
result.to_csv('prediction.csv', index=False)
