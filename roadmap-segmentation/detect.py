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
parser.add_argument('-lm', '--load-model', dest='load_model', default='/scratch/bz1030/data_ds_1008/detection/car-detection/runs/yolov3_large_2020-04-20_14-16-43/best-model-1.pth', type=str)

args = parser.parse_args()
args.demo = False
_, _, _, valloader = setup(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = args.model_config
weights = args.load_model
model = Darknet(model_config, img_size=832).to(device)
model.load_darknet_weights(weights)
model.eval()  # Set in evaluation mode
result = []
bar = tqdm(total=len(valloader), desc='Processing', ncols=90)
yolo = model.yolo_layers
for idx, (sample, target, road_image, extra) in enumerate(valloader):
    sample = torch.stack(sample)
    target = target.to(device)
    sample = sample.to(device)
    with torch.no_grad():
        yolo_outputs = []
        output = model(sample)
        for out, yolo_layer in zip(output, yolo):
            detection, loss, metrics = yolo_layer(out, target, model.img_size)
            yolo_outputs.append(detection)
            break
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        print('Conf score', yolo_outputs[:, :, 4].mean())
        detections = non_max_suppression(yolo_outputs, 0.4, 0.4)
    for meta_info, detection in zip(extra, detections):
        if detection is None:
            continue
        detection = detection.numpy().reshape(1, -1)
        detection = ' '.join([str(i) for i in detection.tolist()])
        result.append((meta_info['file_path'], detection))
    bar.update(1)

result = pd.DataFrame(result, columns=['file', 'annotation'])
result.to_csv('prediction.csv', index=False)
