import argparse
from train import setup, train_yolov3
import torch
parser = argparse.ArgumentParser()
parser.add_argument('-mc', '--model-conifg', dest='model_config',
                    type=str, default='./config/yolov3.cfg')
parser.add_argument('-bs', '--batch-size', dest='batch_size',
                    type=int, default=4)
args = parser.parse_args()

if __name__ == '__main__':
    model, optimizer, trainloader, valloader = setup(args)

    train_yolov3(model, optimizer, trainloader, valloader, args)