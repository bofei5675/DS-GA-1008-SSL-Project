from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from data import UnlabeledDataset, LabeledDataset, LabeledDatasetCenterNet, LabeledDatasetLarge
from terminaltables import AsciiTable
from tqdm import tqdm
from load_ssl import ssl_loader
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
from helper import collate_fn, draw_box, collate_fn2, collate_fn_cn
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
        train_transform = transforms.Compose([transforms.Resize((416, 416)),
                                              transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                                                                     hue=0.5),
                                              transforms.ToTensor()])
        val_transform = transforms.Compose([transforms.Resize((416, 416)),
                                              transforms.ToTensor()])

        labeled_trainset = LabeledDataset(image_folder=image_folder,
                                          annotation_file=annotation_csv,
                                          scene_index=labeled_scene_index_train,
                                          transform=train_transform,
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
                                        transform=val_transform,
                                        extra_info=True
                                        )

        valloader = torch.utils.data.DataLoader(labeled_valset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=2,
                                                collate_fn=collate_fn2)
        if not args.ssl:
            model = pix2vox(args, args.pre_train, args.det, args.seg).to(device)
        else:
            model = ssl_loader(args)
    elif args.model_config == 'center_net':
        train_transform = transforms.Compose([transforms.Resize((416, 416)),
                                              transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                                                                     hue=0.5),
                                              transforms.ToTensor()])
        val_transform = transforms.Compose([transforms.Resize((416, 416)),
                                            transforms.ToTensor()])


        labeled_trainset = LabeledDatasetCenterNet(image_folder=image_folder,
                                               annotation_file=annotation_csv,
                                               scene_index=labeled_scene_index_train,
                                               transform=train_transform,
                                               extra_info=True
                                               )

        trainloader = torch.utils.data.DataLoader(labeled_trainset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=2,
                                                  collate_fn=collate_fn_cn)

        labeled_valset = LabeledDatasetCenterNet(image_folder=image_folder,
                                             annotation_file=annotation_csv,
                                             scene_index=labeled_scene_index_val,
                                             transform=val_transform,
                                             extra_info=True
                                             )

        valloader = torch.utils.data.DataLoader(labeled_valset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=2,
                                                collate_fn=collate_fn_cn)
        if not args.ssl:
            model = pix2vox(args, args.pre_train, False, True).to(device)
        else:
            model = ssl_loader(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
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
                        output += ';'.join(['{}: {:4.3f}'.format(key, value/(idx + 1)/3) for key, value in metrics_epoch.items()])
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


def build_model_dir(args):
    if not os.path.exists('./runs'):
        os.mkdir('./runs')
    model_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    model_name += '_demo' if args.demo else ''
    model_name += '_det' if args.det else ''
    model_name += '_seg' if args.seg else ''
    model_name += '_pt' if args.pre_train else ''
    model_name += '_ssl' if args.ssl else ''
    model_name = 'p2v_yolo' + '_' + model_name if args.model_config == 'pix2vox' else model_name
    model_name = 'center_net' + '_' + model_name if args.model_config == 'center_net' else 'yolov3'  + '_' + model_name

    if not os.path.exists('./runs/' + model_name):
        os.mkdir('./runs/' + model_name)
        with open('./runs/' + model_name + '/config.txt', 'w') as f:
            f.write(str(args))
        print(f'New directory {model_name} created')
    return model_name

def get_metrics_holder(args):
    if args.model_config == 'center_net':
        metrics = {
            "loss": 0,
            'clf_loss': 0,
            'regr_loss': 0,
            'pos_conf':0,
            'neg_conf':0
        }
        return metrics
    if args.seg:
        metrics = {
            "loss": 0,
            'road_map_acc': 0,
            'road_ts': 0
        }
    elif args.det:
        metrics = {
            "loss": 0,
            "x": 0,
            "y": 0,
            "w": 0,
            "h": 0,
            "conf": 0,
            'conf_obj': 0,
            'conf_noobj': 0,
            'rotation': 0,
        }
    return metrics
def train_pix2vox_yolo(model, optimizer, trainloader, valloader, args):
    model_name = build_model_dir(args)
    save_dir = os.path.join('./runs', model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file = open(save_dir + '/model_arch.txt', 'w')
    #print(torchsummary.summary(model, input_size=(6, 3, 416, 416)), file=model_file)
    model_file.close()
    model.train()
    dataloader = {'train': trainloader, 'val': valloader}
    best_metrics = {'loss':1e+6, 'road_map_acc':0, 'road_ts':0}
    yolo_criterion = build_yolo()
    lanemap_criterion = torch.nn.BCEWithLogitsLoss()
    f = open(save_dir + '/log.txt', 'a+')
    for e in range(30):
        for phase in ['train', 'val']:
            print('Stage', phase)
            total_loss = 0
            num_batch = len(dataloader[phase])
            bar = tqdm(total=num_batch, desc='Processing', ncols=90)
            metrics_epoch = get_metrics_holder(args)
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
                    # compute loss
                    if args.det:
                        output, yolo_loss, metrics = yolo_criterion(yolo_outputs, target, 416)
                    else:
                        output, yolo_loss, metrics = None, torch.tensor(0, device=device), {}
                    if args.seg:
                        lane_map_loss = lanemap_criterion(road_outputs, road_image)
                        lane_map_pred = torch.sigmoid(road_outputs)
                        lane_map_mask = (lane_map_pred > 0.5).detach().cpu().numpy()
                        road_image = road_image.cpu().numpy()
                        road_image_acc = (lane_map_mask == road_image).sum() / (args.batch_size * 800 * 800)
                        road_image_ts_score = compute_ts_road_map(lane_map_mask, road_image)

                    else:
                        road_image_acc = 0
                        lane_map_loss = torch.tensor(0, device=device)
                        road_image_ts_score = 0


                    loss = yolo_loss + lane_map_loss
                    iteration_stats = {'loss': loss.item(), 'road_map_acc': road_image_acc,
                                       'road_ts': road_image_ts_score}
                    for key in metrics_epoch:
                        if key not in ['road_map_acc', 'loss'] and key in metrics:
                            metrics_epoch[key] += metrics[key]
                        elif key in ['road_map_acc', 'loss', 'road_ts']:
                            metrics_epoch[key] += iteration_stats[key]

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
                        for key in metrics_epoch:
                            metrics_epoch[key] = metrics_epoch[key] / num_batch
                        write_to_log(output, save_dir)
                        best_metrics = save_model(model.state_dict(), save_dir + '/best-model-{}.pth'.format(e),
                                                  metrics_epoch, best_metrics, args)
    f.close()


def train_center_net(model, optimizer, trainloader, valloader, args):
    model_name = build_model_dir(args)
    save_dir = os.path.join('./runs', model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    dataloader = {'train': trainloader, 'val': valloader}
    best_metrics = {'loss':1e+6, 'road_map_acc':0, 'road_ts':0}
    centernet_criterion = CenterNetLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    f = open(save_dir + '/log.txt', 'a+')
    for e in range(30):
        for phase in ['train', 'val']:
            print('Stage', phase)
            total_loss = 0
            num_batch = len(dataloader[phase])
            bar = tqdm(total=num_batch, desc='Processing', ncols=90)
            metrics_epoch = {'loss':0, 'car_ts':0,  'car_acc':0, 'pos_loss': 0, 'neg_loss': 0}
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for idx, (sample, target, road_image, extra) in enumerate(dataloader[phase]):
                sample = sample.to(device)
                target = target.to(device).float().squeeze(dim=1)
                with torch.set_grad_enabled(phase == 'train'):
                    yolo_outputs, _  = model(sample)
                    # compute loss
                    #print('outputs', yolo_outputs.mean().item(), yolo_outputs.max().item(), yolo_outputs.min().item())
                    prob = torch.sigmoid(yolo_outputs)#.detach().cpu().numpy()
                    # print('prob', prob.mean().item(), prob.max().item(), prob.min().item())
                    target_mask = target.cpu().numpy()
                    #print('target', target_mask.mean(), target_mask.max(), target_mask.min())

                    pred = prob.detach().cpu().numpy() > 0.5
                    target_mask = target_mask > 0.5
                    #yolo_loss = criterion(yolo_outputs, target)
                    mask = (target == 1).float()
                    yolo_loss, pos_loss, neg_loss = focal_loss_cn(prob, target, mask, alpha=args.alpha,
                                              beta=4,
                                              pos_weights=args.gamma, neg_weights=1)
                    loss = yolo_loss  # + lane_map_loss
                    car_acc = (target_mask  == pred).sum() / (args.batch_size * 800 * 800)
                    iteration_stats = {'loss': loss.item(),
                                       'car_ts': compute_ts_road_map(target_mask, pred),
                                       'car_acc': car_acc,
                                       'pos_loss': pos_loss,
                                       'neg_loss': neg_loss}
                    for key in metrics_epoch:
                        metrics_epoch[key] += iteration_stats[key]
                    if args.demo:
                        output = '{}/{}:' \
                            .format(idx + 1, num_batch)
                        output += ';'.join(
                            ['{}: {:4.4f}'.format(key, value / (idx + 1)) for key, value in metrics_epoch.items()])
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
                        output += ';'.join(['{}: {:4.4f}'.format(key, value/(idx + 1)) for key, value in metrics_epoch.items()])
                        print(output)
                        write_to_log(output, save_dir)
                    if phase == 'val' and idx + 1 == num_batch:
                        #print(metrics)
                        output = 'Epoch {}:' \
                            .format(e)
                        output += ';'.join(
                            ['{}: {:4.4f}'.format(key, value / num_batch) for key, value in metrics_epoch.items()])
                        print(output)
                        for key in metrics_epoch:
                            metrics_epoch[key] = metrics_epoch[key] / num_batch
                        write_to_log(output, save_dir)
                        save_model(model.state_dict(), save_dir + '/best-model-{}.pth'.format(e), metrics_epoch, best_metrics, args)
    f.close()

def save_model(model, path, metrics_epoch, best_metrics, args):
    if args.det or (args.det and args.seg):
        if metrics_epoch['loss'] < best_metrics['loss']:
            torch.save(model,
                    path)
            best_metrics['loss'] = metrics_epoch['loss']
    elif args.seg:
        if metrics_epoch['road_ts'] > best_metrics['road_ts']:
            torch.save(model,
                    path)
            best_metrics['road_ts'] = metrics_epoch['road_ts']
    return best_metrics


def write_to_log(text, save_dir):
    f = open(save_dir + '/log.txt', 'a+')
    f.write(str(text) + '\n')
    f.close()

def compute_ts_road_map(road_map1, road_map2):
    tp = (road_map1 * road_map2).sum()
    print(road_map1.sum(), road_map2.sum())
    print(tp,'/', road_map1.sum() + road_map2.sum() - tp)
    return tp * 1.0 / (road_map1.sum() + road_map2.sum() - tp)

