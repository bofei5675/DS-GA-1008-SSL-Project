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

parser = argparse.ArgumentParser()
parser.add_argument('-mc', '--model-config', dest='model_config', default='./config/yolov3.cfg', type=str)
parser.add_argument('-bs', '--batch-size', dest='batch_size', default=4, type=int)
args = parser.parse_args()
_, _, _, valloader = setup(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = args.model_config
weights = '/scratch/bz1030/data_ds_1008/detection/runs/yolov3_2020-04-16_12-01-17/best-model-1.pth'
model = Darknet(model_config, img_size=256).to(device)
model.load_darknet_weights(weights)
model.eval()  # Set in evaluation mode

for idx, (sample, target, road_image, extra) in enumerate(valloader):
    sample = torch.stack(sample)
    sample = sample.to(device)
    with torch.no_grad():
        _, detections, _ = model(sample)
        detections = non_max_suppression(detections, 0.5, 0.3)
    print(detections)