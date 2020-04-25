from __future__ import division

from models import *
from data import LabeledDataset
from terminaltables import AsciiTable
from tqdm import tqdm

import os
import sys
import time
import datetime
import argparse
import numpy as np
from models import UNet
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from helper import collate_fn, draw_box, collate_fn2
import torchsummary
from pix2vox import pix2vox

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def setup(args = None):
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

    # model = UNet(3, 1).to(device)
    model = pix2vox().to(device)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    return model, optimizer, trainloader, valloader

'''

Refer to this function, we might want to change the architecture of UNet
so we forward pass 6 times then find way combine 6 feature maps togehter 
to compute loss
Current method: 1x1 conv inchannel=6 out_channel=1
'''
def bce_loss(outputs, targets):
    outputs = outputs.squeeze(dim=1)
    prob = torch.sigmoid(outputs)
    pos_loss = targets * -torch.log(prob + 1e-8)
    neg_loss = (1 - targets) * -torch.log(1 - prob + 1e-8)
    loss = pos_loss + neg_loss
    loss = loss.sum(dim=(1, 2))
    return loss.mean(), prob

def train_unet(model, optimizer, trainloader, valloader, args):
    if not os.path.exists('runs'):
        os.mkdir('runs')
    model_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    model_name = 'unet' + '_' + model_name
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
    f = open(save_dir + '/log.txt', 'a+')
    criterion = nn.BCEWithLogitsLoss()

    for e in range(30):
        for phase in ['train', 'val']:
            print('Stage', phase)
            num_batch = len(dataloader[phase])
            bar = tqdm(total=num_batch, desc='Processing', ncols=90)
            metrics_epoch = {
                "loss": 0,
                'accuracy': 0
            }
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for idx, (sample, target, road_image, extra) in enumerate(dataloader[phase]):
                sample = torch.stack(sample)
                target = torch.stack(road_image)
                sample = sample.to(device)
                target = target.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = []
                    for image_idx in range(6):
                        image  = sample[:, image_idx].squeeze(dim=1)
                        output = model(image)
                        outputs.append(output)
                    # combine outputs
                    outputs = torch.cat(outputs, dim=1)
                    outputs = model.mapping(outputs).squeeze(dim=1)
                    # compute loss
                    loss = criterion(outputs, target)
                    prob = torch.sigmoid(outputs)
                    pred_mask = prob > 0.5
                    pred_mask = pred_mask.cpu().numpy()
                    accuracy = (pred_mask == target.cpu().numpy()).sum() / (args.batch_size * 800  * 800)

                    metrics_epoch['loss'] += loss.item()
                    metrics_epoch['accuracy'] += accuracy
                    if args.demo:
                        output = '{}/{}:' \
                            .format(idx + 1, num_batch)
                        output += ';'.join(
                            ['{}: {:4.4f}'.format(key, value / (idx + 1)) for key, value in metrics_epoch.items()])
                        output += '; Running loss:{:.4f}; Running acc:{:.4f}'.format(loss.item(), accuracy)
                        print(output)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
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
                            ['{}: {:4.4f}'.format(key, value / num_batch ) for key, value in metrics_epoch.items()])
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


