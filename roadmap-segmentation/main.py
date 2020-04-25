import argparse
from train import setup, train_unet
import torch

# 应该不用改
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
parser.add_argument('-bs', '--batch-size', dest='batch_size',
                    type=int, default=1)
parser.add_argument('-dm', '--demo', dest='demo',
                    type=str2bool, default='no')
args = parser.parse_args()

if __name__ == '__main__':
    model, optimizer, trainloader, valloader = setup(args)
    train_unet(model, optimizer, trainloader, valloader, args)
