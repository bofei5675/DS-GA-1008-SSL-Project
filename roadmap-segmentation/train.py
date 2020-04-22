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
import torchsummary
def setup(args = None):
    model_config = args.model_config
    # Initiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Use cuda?', torch.cuda.is_available())
    print('Device name:', torch.cuda.get_device_name(0))
    image_folder = '/scratch/bz1030/data_ds_1008/DLSP20Dataset/data'
    annotation_csv = '/scratch/bz1030/data_ds_1008/DLSP20Dataset/data/annotation.csv'
    if args.demo:
        labeled_scene_index_train = np.arange(106, 107)
        labeled_scene_index_val = np.arange(130, 131)
    else:
        labeled_scene_index_train = np.arange(106, 130)
        labeled_scene_index_val = np.arange(130, 134)
    if 'large' not in args.model_config:

        model = Darknet(model_config, 416).to(device)
        model.apply(weights_init_normal)
        if args.pre_train:
            model.load_darknet_weights('/scratch/bz1030/data_ds_1008/PyTorch-YOLOv3/weights/yolov3.weights')
        transform = transforms.Compose([transforms.Resize((416, 416)),
                                        transforms.ToTensor()])

        labeled_trainset = LabeledDataset(image_folder=image_folder,
                                          annotation_file=annotation_csv,
                                          scene_index=labeled_scene_index_train,
                                          transform=transform,
                                          extra_info=True
                                          )

        trainloader = torch.utils.data.DataLoader(labeled_trainset,
                                                  batch_size=args.batch_size,
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
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=2,
                                                  collate_fn=collate_fn2)
'''

Refer to this function, we might want to change the architecture of UNet
so we forward pass 6 times then find way combine 6 feature maps togehter 
to compute loss
Current method: 1x1 conv inchannel=6 out_channel=1
'''
def train_yolov3_pass_6(model, optimizer, trainloader, valloader, args):
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
    model_file = open(save_dir + '/model_arch.txt', 'w')
    print(torchsummary.summary(model, input_size=(3, 416, 416)), file=model_file)
    model_file.close()
    model.train()
    dataloader = {'train': trainloader, 'val': valloader}
    best_loss = [1e+6]
    yolo = model.yolo_layers
    f = open(save_dir + '/log.txt', 'a+')
    for e in range(30):
        for phase in ['train', 'val']:
            print('Stage', phase)
            total_loss = 0
            num_batch = len(dataloader[phase])
            bar = tqdm(total=num_batch, desc='Processing', ncols=90)
            metrics_epoch = {
                "loss": 0,
                "x":0,
                "y":0,
                "w":0,
                "h":0,
                "conf": 0,
                'conf_obj':0,
                'conf_noobj':0,
                "cls": 0,
                "cls_acc": 0,
                'precision':0,
                'recall50':0,
                'recall75':0,
                'rotation':0
            }
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for idx, (sample, target, road_image, extra) in enumerate(dataloader[phase]):
                sample = torch.stack(sample)
                sample = sample.to(device)
                target = target.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = {0: None, 1:None, 2: None}
                    # 这里要把6张图依次PASS， 保存他们的FEATURE MAP然后合成一张
                    # 怎么合成： 1） 一个1X1 CONV2D？ 2）直接加一起？
                    outputs = []
                    for image_idx in range(6):
                        image  = sample[:, image_idx].squeeze()
                        output = model(image)
                        outputs.append(output)
                    # combine outputs
                    outputs = torch.sum(outputs) or model.one_by_one_conv(torch.stack(outputs))
                    # compute loss
                    loss = pixel_wise_bce(outputs, road_image)
                    if args.demo:
                        output = '{}/{}:' \
                            .format(idx + 1, num_batch)
                        output += ';'.join(
                            ['{}: {:4.2f}'.format(key, value / (idx + 1) / 3) for key, value in metrics_epoch.items()])
                        print(output)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    total_loss += loss.item()
                    bar.update(1)
                    if idx % 100 == 0 and phase == 'train':
                        #print(metrics)
                        output = '{}/{}:' \
                              .format(idx + 1, num_batch)
                        output += ';'.join(['{}: {:4.2f}'.format(key, value/(idx + 1)/3) for key, value in metrics_epoch.items()])
                        print(output)
                        write_to_log(output, save_dir)
                    if phase == 'val' and idx + 1 == num_batch:
                        #print(metrics)
                        output = 'Epoch {}:' \
                            .format(e)
                        output += ';'.join(
                            ['{}: {:4.2f}'.format(key, value / num_batch / 3) for key, value in metrics_epoch.items()])
                        print(output)
                        loss_epoch = metrics_epoch['loss'] / num_batch / 3
                        write_to_log(output, save_dir)
                        if np.min(best_loss) > loss_epoch:
                            torch.save(model.state_dict(), save_dir + '/best-model-{}.pth'.format(e))
                        best_loss.append(loss_epoch)
    f.close()



def save_model(model, path, epoch):
    torch.save(model.state_dict(),
               os.path.join(path, 'best-model-{}.pth'.format(epoch)))

def write_to_log(text, save_dir):
    f = open(save_dir + '/log.txt', 'a+')
    f.write(str(text) + '\n')
    f.close()


