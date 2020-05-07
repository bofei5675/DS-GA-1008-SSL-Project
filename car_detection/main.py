import argparse

from train import setup, train_center_net, train_yolov3, train_yolov3_pass_6, train_pix2vox_yolo
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
                    choices=['./config/yolov3_large.cfg', './config/yolov3.cfg', 'pix2vox', 'center_net'])
parser.add_argument('-bs', '--batch-size', dest='batch_size',
                    type=int, default=4)
parser.add_argument('-pt', '--pre-train', dest='pre_train',
                    type=str2bool, default='no')
parser.add_argument('-dm', '--demo', dest='demo',
                    type=str2bool, default='no')
parser.add_argument('-det', '--detection', dest='det',
                    type=str2bool, default='yes')
parser.add_argument('-seg', '--segmentation', dest='seg',
                    type=str2bool, default='yes')
parser.add_argument('-ssl', '--ssl', dest='ssl',
                    type=str2bool, default='no')
parser.add_argument('-a', '--alpha', dest='alpha',
                    type=float, default=2)
parser.add_argument('-g', '--gamma', dest='gamma',
                    type=float, default=1)

args = parser.parse_args()

if __name__ == '__main__':
    model, optimizer, trainloader, valloader = setup(args)
    if args.model_config == './config/yolov3_large.cfg':
        train_yolov3(model, optimizer, trainloader, valloader, args)
    elif args.model_config == './config/yolov3.cfg':
        train_yolov3_pass_6(model, optimizer, trainloader, valloader, args)
    elif args.model_config == 'pix2vox':
        train_pix2vox_yolo(model, optimizer, trainloader, valloader, args)
    elif args.model_config == 'center_net':
        train_center_net(model, optimizer, trainloader, valloader, args)