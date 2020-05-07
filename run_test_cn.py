import os
import random
import argparse

import numpy as np
import sys

sys.path.append('./car_detection')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from data_helper import LabeledDataset
from helper import compute_ats_bounding_boxes, compute_ts_road_map, draw_box, draw_box_no_scale

from model_loader import get_transform, ModelLoader, ModelLoader2
from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../DLSP20Dataset/data')
parser.add_argument('--det_model', type=str, default=None)
parser.add_argument('--seg_model', type=str, default=None)
parser.add_argument('--testset', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--debug', action='store_true')
opt = parser.parse_args()

image_folder = opt.data_dir
annotation_csv = f'{opt.data_dir}/annotation.csv'

if opt.testset:
    labeled_scene_index = np.arange(134, 148)
else:
    labeled_scene_index = np.arange(130, 134)

labeled_trainset = LabeledDataset(
    image_folder=image_folder,
    annotation_file=annotation_csv,
    scene_index=labeled_scene_index,
    transform=get_transform(),
    extra_info=True
)
dataloader = torch.utils.data.DataLoader(
    labeled_trainset,
    batch_size=1,
    shuffle=False,
    num_workers=4
)
bar = tqdm(total=len(dataloader), desc='Processing', ncols=90)

model_loader = ModelLoader2()

total = 0
total_ats_bounding_boxes = 0
total_ts_road_map = 0
bar = tqdm(total=len(dataloader), desc='Processing', ncols=90)

for i, data in enumerate(dataloader):
    total += 1
    sample, target, road_image, extra = data
    if torch.cuda.is_available():
        sample = sample.cuda()
    # print(sample.shape)
    # only works for batch size = 1 ?
    predicted_bounding_boxes = model_loader.get_bounding_boxes(sample)[0].cpu()
    predicted_road_map = model_loader.get_binary_road_map(sample).cpu()
    ats_bounding_boxes = compute_ats_bounding_boxes(predicted_bounding_boxes, target['bounding_box'][0])
    ts_road_map = compute_ts_road_map(predicted_road_map, road_image)

    total_ats_bounding_boxes += ats_bounding_boxes
    total_ts_road_map += ts_road_map
    bar.update(1)
    if opt.verbose:
        print(f'{i} - Bounding Box Score: {ats_bounding_boxes:.4} - Road Map Score: {ts_road_map:.4}')
    if opt.debug:
        # draw bbox
        fig, ax = plt.subplots()

        # The ego car position
        ax.grid()
        ax.plot(400, 400, 'x', color="red")
        for i, bb in enumerate(target['bounding_box'][0]):
            # print(bb)
            # print(bb.shape)
            point_squence = torch.stack([bb[:, 0], bb[:, 1], bb[:, 3], bb[:, 2], bb[:, 0]])

            ax.plot(point_squence.T[0] * 10 + 400, -point_squence.T[1] * 10 + 400, color='blue')

        for i, bb in enumerate(predicted_bounding_boxes):
            draw_box_no_scale(ax, bb, color='orange')

        ax.set_xlim(0, 800)
        ax.set_ylim(0, 800)
        ax.set_title('blue=Target; orange=Prediction')
        plt.savefig(
            '{}/debug_det/{}_{}.png'.format(model_loader.debug_det, extra['scene_id'].item(),
                                            extra['sample_id'].item()))
        plt.close()

        # draw lanemap
        fig, ax = plt.subplots(1, 2)
        # print(road_image.cpu().numpy().shape, predicted_road_map.cpu().numpy().shape)
        target_mask, pred_mask = road_image.cpu().numpy()[0], predicted_road_map.cpu().numpy()[0]
        ax[0].imshow(target_mask)
        ax[1].imshow(pred_mask)
        # The ego car position
        ax[1].set_title('Pred Score:{}'.format(ts_road_map))
        plt.tight_layout()
        plt.savefig(
            '{}/debug_seg/{}_{}.png'.format(model_loader.debug_seg, extra['scene_id'].item(),
                                            extra['sample_id'].item()))
        plt.close()

print(
    f'{model_loader.team_name} - {model_loader.round_number} - Bounding Box Score: {total_ats_bounding_boxes / total:.4} - Road Map Score: {total_ts_road_map / total:.4}')

avg_det_score = total_ats_bounding_boxes / total
avg_seg_score = total_ts_road_map / total

with open(model_loader.debug_det + '/eval.txt', 'w') as f:
    f.write('Detection threat score: {}'.format(avg_det_score))

with open(model_loader.debug_seg + '/eval.txt', 'w') as f:
    f.write('Detection threat score: {}'.format(avg_seg_score))





