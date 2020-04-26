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
from pix2vox  import pix2vox
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
    if args.model_config  == './config/yolov3.cfg':

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
    elif args.model_config == './config/yolov3_large.cfg':
        model = Darknet(model_config, 832).to(device)

        transform = transforms.Compose([transforms.Resize((832, 832)),
                                        transforms.ToTensor()])

        labeled_trainset = LabeledDatasetLarge(image_folder=image_folder,
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

        labeled_valset = LabeledDatasetLarge(image_folder=image_folder,
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
    elif args.model_config == 'pix2vox':
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
        model = pix2vox(args.pre_train).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer, trainloader, valloader


def train_yolov3(model, optimizer, trainloader, valloader, args):
    if not os.path.exists('runs'):
        os.mkdir('runs')
    model_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    model_name = 'yolov3_large' + '_' + model_name

    if not os.path.exists('runs/' + model_name):
        os.mkdir('runs/' + model_name)
        with open('runs/' + model_name + '/config.txt', 'w') as f:
            f.write(str(args))
        print(f'New directory {model_name} created')

    save_dir = os.path.join('./runs', model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file = open(save_dir + '/model_arch.txt', 'w')
    #print(torchsummary.summary(model, input_size=(6, 3, 416, 416)), file=model_file)
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

            if phase == 'train':
                model.train()
            else:
                model.eval()
            for idx, (sample, target, road_image, extra) in enumerate(dataloader[phase]):
                sample = torch.stack(sample)
                sample = sample.to(device)
                target = target.to(device)
                yolo_output = model(sample)
                loss = 0
                for yolo_layer, output in zip(yolo, yolo_output):
                    output, layer_loss, metrics = yolo_layer(output, target, 832)
                    loss += layer_loss
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
    yolo = build_yolo()
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
                    for image_idx in range(6):
                        image  = sample[:, image_idx].squeeze()
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
                    for output_idx in range(3):
                        output = outputs[output_idx]
                        print('output', output_idx, output.shape)
                        output = model.module_list[-3 + output_idx](output)
                        output, layer_loss, metrics = yolo[output_idx](output, target, 416)
                        for key in metrics_epoch:
                            metrics_epoch[key] += metrics[key]
                        loss += layer_loss
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



def train_pix2vox_yolo(model, optimizer, trainloader, valloader, args):
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
    #print(torchsummary.summary(model, input_size=(6, 3, 416, 416)), file=model_file)
    model_file.close()
    model.train()
    dataloader = {'train': trainloader, 'val': valloader}
    best_loss = [1e+6]
    yolo_criterion = build_yolo()
    lanemap_criterion = torch.nn.BCEWithLogitsLoss()
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
                'rotation':0,
                'road_map_acc': 0
            }
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for idx, (sample, target, road_image, extra) in enumerate(dataloader[phase]):
                sample = torch.stack(sample)
                road_image = torch.stack(road_image)
                sample = sample.to(device)
                target = target.to(device)
                road_image = road_image.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    road_outputs, yolo_outputs  = model(sample)
                    road_outputs = road_outputs.squeeze(dim=1)
                    # compute loss
                    output, yolo_loss, metrics = yolo_criterion(yolo_outputs, target, 416)
                    lane_map_loss = lanemap_criterion(road_outputs, road_image)
                    lane_map_pred = torch.sigmoid(road_outputs)
                    lane_map_mask = (lane_map_pred > 0.5).detach().cpu().numpy()
                    road_image = road_image.cpu().numpy()
                    road_image_acc = (lane_map_mask == road_image).sum() / (args.batch_size * 800 * 800)
                    for key in metrics_epoch:
                        if key != 'road_map_acc':
                            metrics_epoch[key] += metrics[key]
                        else:
                            metrics_epoch[key] += road_image_acc
                    loss = yolo_loss + lane_map_loss
                    if args.demo:
                        output = '{}/{}:' \
                            .format(idx + 1, num_batch)
                        output += ';'.join(
                            ['{}: {:4.2f}'.format(key, value / (idx + 1)) for key, value in metrics_epoch.items()])
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
                        output += ';'.join(['{}: {:4.2f}'.format(key, value/(idx + 1)) for key, value in metrics_epoch.items()])
                        print(output)
                        write_to_log(output, save_dir)
                    if phase == 'val' and idx + 1 == num_batch:
                        #print(metrics)
                        output = 'Epoch {}:' \
                            .format(e)
                        output += ';'.join(
                            ['{}: {:4.2f}'.format(key, value / num_batch) for key, value in metrics_epoch.items()])
                        print(output)
                        loss_epoch = metrics_epoch['loss'] / num_batch
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


