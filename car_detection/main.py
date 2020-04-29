import argparse

from train import setup, train_yolov3, train_yolov3_pass_6, train_pix2vox_yolo
import torch
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser()
parser.add_argument('-mc', '--model-conifg', dest='model_config',
                    type=str, default='pix2vox',
                    choices=['./config/yolov3_large.cfg', './config/yolov3.cfg', 'pix2vox'])
parser.add_argument('-bs', '--batch-size', dest='batch_size',
                    type=int, default=4)
parser.add_argument('-pt', '--pre-train', dest='pre_train',
                    type=str2bool, default='no')
parser.add_argument('-dm', '--demo', dest='demo',
                    type=str2bool, default='no')
args = parser.parse_args()

if __name__ == '__main__':
    model, optimizer, trainloader, valloader = setup(args)
    #if 'large' in args.model_config:
     #   train_yolov3(model, optimizer, trainloader, valloader, args)
    #else:
    #train_yolov3_pass_6(model, optimizer, trainloader, valloader, args)
    train_pix2vox_yolo(model, optimizer, trainloader, valloader, args)
