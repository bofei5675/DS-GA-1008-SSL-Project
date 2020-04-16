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


_, _, _, valloader = setup()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = './config/yolov3.cfg'
weights = '/scratch/bz1030/data_ds_1008/detection/runs/yolov3_2020-04-16_09-26-14/best-model-3.pth'
model = Darknet(model_config, img_size=256).to(device)
model.load_darknet_weights(weights)
model.eval()  # Set in evaluation mode

for idx, (sample, target, road_image, extra) in enumerate(valloader):
    sample = torch.stack(sample)
    sample = sample.to(device)
    with torch.no_grad():
        _, detections, _ = model(sample)
        print(detections.shape)
        detections = non_max_suppression(detections, 0.4, 0.3)
    print(detections)