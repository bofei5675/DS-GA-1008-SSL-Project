from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from data import UnlabeledDataset, LabeledDataset, LabeledDatasetLarge
from terminaltables import AsciiTable
from tqdm import tqdm

import os
import sys
import time
import datetime
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from helper import collate_fn, draw_box, collate_fn2

def setup(args = None):
    model_config = args.model_config
    # Initiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(model_config).to(device)
    model.apply(weights_init_normal)
    print('Use cuda?', torch.cuda.is_available())
    print('Device name:', torch.cuda.get_device_name(0))
    image_folder = '/scratch/bz1030/data_ds_1008/DLSP20Dataset/data'
    annotation_csv = '/scratch/bz1030/data_ds_1008/DLSP20Dataset/data/annotation.csv'
    labeled_scene_index_train = np.arange(106, 130)
    labeled_scene_index_val = np.arange(130, 134)
    if 'large' not in args.model_config:
        transform = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.ToTensor()])

        labeled_trainset = LabeledDataset(image_folder=image_folder,
                                          annotation_file=annotation_csv,
                                          scene_index=labeled_scene_index_train,
                                          transform=transform,
                                          extra_info=True
                                          )

        trainloader = torch.utils.data.DataLoader(labeled_trainset,
                                                  batch_size=8,
                                                  shuffle=True,
                                                  num_workers=2,
                                                  collate_fn=collate_fn2)

        labeled_valset = LabeledDataset(image_folder=image_folder,
                                          annotation_file=annotation_csv,
                                          scene_index=labeled_scene_index_val,
                                          transform=transform,
                                          extra_info=True
                                          )

        valloader = torch.utils.data.DataLoader(labeled_valset,
                                                  batch_size=8,
                                                  shuffle=True,
                                                  num_workers=2,
                                                  collate_fn=collate_fn2)
    else:
        transform = transforms.Compose([transforms.Resize((832, 832)),
                                        transforms.ToTensor()])

        labeled_trainset = LabeledDatasetLarge(image_folder=image_folder,
                                          annotation_file=annotation_csv,
                                          scene_index=labeled_scene_index_train,
                                          transform=transform,
                                          extra_info=True
                                          )

        trainloader = torch.utils.data.DataLoader(labeled_trainset,
                                                  batch_size=8,
                                                  shuffle=True,
                                                  num_workers=2,
                                                  collate_fn=collate_fn2)

        labeled_valset = LabeledDatasetLarge(image_folder=image_folder,
                                        annotation_file=annotation_csv,
                                        scene_index=labeled_scene_index_val,
                                        transform=transform,
                                        extra_info=True
                                        )

        valloader = torch.utils.data.DataLoader(labeled_valset,
                                                batch_size=8,
                                                shuffle=True,
                                                num_workers=2,
                                                collate_fn=collate_fn2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer, trainloader, valloader


def train_yolov3(model, optimizer, trainloader, valloader, args):
    if not os.path.exists('runs'):
        os.mkdir('runs')
    model_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    model_name = 'yolov3' + '_' + model_name

    if not os.path.exists('runs/' + model_name):
        os.mkdir('runs/' + model_name)
        with open('runs/' + model_name + '/config.txt', 'w') as f:
            f.write(str(args))
        print(f'New directory {model_name} created')

    save_dir = os.path.join('./runs', model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    dataloader = {'train': trainloader, 'val': valloader}
    best_loss = [1e+6]
    f = open(save_dir + '/log.txt', 'a+')
    for e in range(30):
        for phase in ['train', 'val']:
            print('Stage', phase)
            total_loss = 0
            num_batch = len(dataloader[phase])
            bar = tqdm(total=num_batch, desc='Processing', ncols=90)

            if phase == 'train':
                model.train()
            else:
                model.eval()
            for idx, (sample, target, road_image, extra) in enumerate(dataloader[phase]):
                sample = torch.stack(sample)
                sample = sample.to(device)
                target = target.to(device)
                loss, yolo_output, metrics = model(sample, target)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                bar.update(1)
                if idx % 100 == 0 and phase == 'train':
                    #print(metrics)
                    output = '{}/{}, {} Loss:{:.4f}' \
                          .format(idx + 1, num_batch, phase, total_loss / (idx + 1))
                    print(output)
                    write_to_log(output, save_dir)
            if phase == 'val':
                #print(metrics)
                output = 'Epoch {}, Val Loss: {:.4f}' \
                      .format(e + 1, total_loss / num_batch)
                print(output)
                write_to_log(output, save_dir)
                if np.min(best_loss) > (total_loss / num_batch):
                    model.save_darknet_weights(save_dir + '/best-model-{}.pth'.format(e))
                best_loss.append(total_loss / num_batch)
    f.close()



def save_model(model, path, epoch):
    torch.save(model.state_dict(),
               os.path.join(path, 'best-model-{}.pth'.format(epoch)))

def write_to_log(text, save_dir):
    f = open(save_dir + '/log.txt', 'a+')
    f.write(str(text) + '\n')
    f.close()


