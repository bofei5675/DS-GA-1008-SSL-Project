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
parser.add_argument('-mc', '--model-config', dest='model_config', default='./config/yolov3.cfg', type=str)
parser.add_argument('-bs', '--batch-size', dest='batch_size', default=4, type=int)
parser.add_argument('-lm', '--load-model', dest='load_model', default='/scratch/bz1030/data_ds_1008/detection/car-detection/runs/yolov3_2020-04-21_08-49-56/best-model-1.pth', type=str)
parser.add_argument('-th', '--threshold', dest='threshold', default=0.5, type=float)

args = parser.parse_args()
args.demo = False
args.pre_train = True
_, _, _, valloader = setup(args)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if  use_cuda else "cpu")
model_config = args.model_config
model = Darknet(model_config, img_size=416).to(device)
if use_cuda:
    state_dict = torch.load(args.load_model)
else:
    state_dict = torch.load(args.load_model, map_location='cpu')
model.load_state_dict(state_dict)
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
        outputs = {0: None, 1: None, 2: None}
        for image_idx in range(6):
            image = sample[:, image_idx].squeeze()
            yolo_output = model(image)

            for output_idx in range(3):
                if outputs[output_idx] is None:
                    outputs[output_idx] = yolo_output[output_idx]
                else:
                    outputs[output_idx] = torch.cat((outputs[output_idx],
                                                     yolo_output[output_idx]),
                                                    dim=1)
        # compute loss
        loss = 0
        detections = []
        for output_idx in range(3):
            output = outputs[output_idx]
            output = model.module_list[-3 + output_idx](output)
            output, layer_loss, metrics = yolo[output_idx](output, target, 416)
            detections.append(output)
            loss += layer_loss
        detections = to_cpu(torch.cat(detections, 1))
        print('average/max conf:', detections[:, :, 4].mean().item(), detections[:, :, 4].max().item())
        print('loss:', loss.item(), target.shape)
        detections = non_max_suppression(detections, args.threshold, 0)
    for meta_info, detection in zip(extra, detections):
        if detection is None:
            print('No  detection')
            continue
        detection = detection.numpy().reshape(1, -1)
        detection = ' '.join([str(i) for i in detection.tolist()])
        result.append((meta_info['file_path'], detection))
    bar.update(1)

result = pd.DataFrame(result, columns=['file', 'annotation'])
result.to_csv('prediction_thres_{}.csv'.format(args.threshold), index=False)