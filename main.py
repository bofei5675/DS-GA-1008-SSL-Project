import argparse
from train import setup, train_yolov3
import torch
parser = argparse.ArgumentParser()
args = parser.parse_args()

if __name__ == '__main__':
    model, optimizer, trainloader, valloader = setup()

    train_yolov3(model, optimizer, trainloader, valloader, args)