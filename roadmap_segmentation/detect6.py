'''
This scripts is for output for evaluation
Refer to car-detection/detect6.py for what we did for bounding box.
'''


import argparse
import torch
import numpy as np
from models import *
from train import setup
import  matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import pandas as pd
from tqdm import tqdm
import os
from pix2vox import pix2vox_seg

parser = argparse.ArgumentParser()
parser.add_argument('-bs', '--batch-size', dest='batch_size', default=4, type=int)
parser.add_argument('-lm', '--load-model',
                    dest='load_model',
                    default='/scratch/bz1030/data_ds_1008/detection/roadmap_segmentation/runs/pix2vox_2020-04-26_01-18-51/best-model-15.pth',
                    type=str)
parser.add_argument('-th', '--threshold', dest='threshold', default=0.5, type=float)
parser.add_argument('-m', '--model', dest='model', default='pix2vox', type=str,
                    choices=['unet', 'pix2vox'])

args = parser.parse_args()
args.demo = False
args.pre_train = False
_, _, _, valloader = setup(args)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if  use_cuda else "cpu")

if args.model == 'unet':
    model = UNet(3, 1).to(device)
elif args.model == 'pix2vox':
    model = pix2vox_seg().to(device)
save_dir = args.load_model.split('/')
save_dir = '/'.join(save_dir[:-1])
save_dir += '/eval_fig'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if use_cuda:
    state_dict = torch.load(args.load_model)
else:
    state_dict = torch.load(args.load_model, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()  # Set in evaluation mode
result = []
bar = tqdm(total=len(valloader), desc='Processing', ncols=90)
criterion = nn.BCEWithLogitsLoss()

for idx, (sample, target, road_image, extra) in enumerate(valloader):
    sample = torch.stack(sample)
    road_image = torch.stack(road_image)
    target = road_image.to(device)
    sample = sample.to(device)
    with torch.no_grad():
        if args.model == 'unet':
            outputs = []
            for image_idx in range(6):
                image = sample[:, image_idx].squeeze(dim=1)
                output = model(image)
                outputs.append(output)

            # combine outputs
            outputs = torch.cat(outputs, dim=1)
            outputs = model.mapping(outputs).squeeze(dim=1)
        elif args.model == 'pix2vox':
            outputs = model(sample)
        # compute loss
        loss = criterion(outputs, target)
        prob = torch.sigmoid(outputs)
        pred_mask = prob > 0.5
        pred_mask = pred_mask.cpu().numpy()
        target = target.cpu().numpy()
        accuracy = (pred_mask == target).sum() / (args.batch_size * 800 * 800)

        print(accuracy)
        cnt = 0
        for pred,  t_mask, info in zip(pred_mask, target, extra):
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(t_mask)
            ax[1].imshow(pred)
            plt.tight_layout()
            plt.title('Accuracy: {:.3f}'.format(accuracy))
            plt.savefig(save_dir + '/{}_{}.png'.format(info['scene_id'], info['sample_id']))
            plt.close()
            cnt += 1

    bar.update(1)
