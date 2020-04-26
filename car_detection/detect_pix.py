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
from pix2vox import pix2vox
parser = argparse.ArgumentParser()
parser.add_argument('-mc', '--model-config', dest='model_config', default='pix2vox', type=str)
parser.add_argument('-bs', '--batch-size', dest='batch_size', default=4, type=int)
parser.add_argument('-lm', '--load-model', dest='load_model', default='/scratch/bz1030/data_ds_1008/detection/car-detection/runs/yolov3_2020-04-26_06-41-09/best-model-1.pth', type=str)
parser.add_argument('-th', '--threshold', dest='threshold', default=0.5, type=float)

args = parser.parse_args()
save_dir = args.load_model.split('/')
save_dir = '/'.join(save_dir[:-1])
save_dir += '/eval_fig'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
args.demo = False
args.pre_train = True
_, _, _, valloader = setup(args)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if  use_cuda else "cpu")
model = pix2vox().to(device)
if use_cuda:
    state_dict = torch.load(args.load_model)
else:
    state_dict = torch.load(args.load_model, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()  # Set in evaluation mode
result = []
bar = tqdm(total=len(valloader), desc='Processing', ncols=90)
yolo_criterion = build_yolo()
lanemap_criterion = torch.nn.BCEWithLogitsLoss()
for idx, (sample, target, road_image, extra) in enumerate(valloader):
    sample = torch.stack(sample)
    road_image = torch.stack(road_image)
    target = target.to(device)
    sample = sample.to(device)
    road_image = road_image.to(device)
    with torch.no_grad():
        road_outputs, yolo_outputs = model(sample)
        road_outputs = road_outputs.squeeze(dim=1)
        # compute loss
        output, yolo_loss, metrics = yolo_criterion(yolo_outputs, target, 416)
        lane_map_loss = lanemap_criterion(road_outputs, road_image)
        lane_map_pred = torch.sigmoid(road_outputs)
        lane_map_mask = (lane_map_pred > 0.5).detach().cpu().numpy()
        road_image = road_image.cpu().numpy()
        road_image_acc = (lane_map_mask == road_image).sum() / (args.batch_size * 800 * 800)
        loss = yolo_loss + lane_map_loss
        detections = [output]
        detections = to_cpu(torch.cat(detections, 1))
        print('average/max conf:', detections[:, :, 6].mean().item(), detections[:, :, 6].max().item())
        print('road map acc:', road_image_acc)
        print('loss:', loss.item())
        detections = non_max_suppression(detections, args.threshold, 0)
    for meta_info, detection, lane_map, mask_pred, info in zip(extra, detections, road_image, lane_map_mask, extra):
        if detection is None:
            print('No  detection')
        else:
            detection = detection.numpy().reshape(1, -1)
            detection = ' '.join([str(i) for i in detection.tolist()])
            result.append((meta_info['file_path'], detection))
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(lane_map)
        ax[1].imshow(mask_pred)
        plt.tight_layout()
        plt.title('Accuracy: {:.3f}'.format(road_image_acc))
        plt.savefig(save_dir + '/{}_{}.png'.format(info['scene_id'], info['sample_id']))
        plt.close()

    bar.update(1)

result = pd.DataFrame(result, columns=['file', 'annotation'])
result.to_csv('prediction_thres_{}.csv'.format(args.threshold), index=False)