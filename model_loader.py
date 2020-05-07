"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from car_detection.models import build_yolo, CenterNetLoss
from car_detection.utils.utils import to_cpu, nms_with_rot, extract_coords
from car_detection.pix2vox import pix2vox
from roadmap_segmentation.pix2vox import pix2vox_seg
import os
import argparse
# import your model class
# import ...

# Put your transform function here, we will use it for our dataloader
def get_transform():
    transform = transforms.Compose([transforms.Resize((416, 416)),
                                    transforms.ToTensor()])
    return transform

def get_transform_task1():
    transform = transforms.Compose([transforms.Resize((416, 416)),
                                    transforms.ToTensor()])
    return transform

def get_transform_task2():
    transform = transforms.Compose([transforms.Resize((416, 416)),
                                    transforms.ToTensor()])
    return transform
def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

class ModelLoader():
    # Fill the information for your team
    team_name = 'some name'
    round_number = 1
    team_member = ['Bofei Zhang', 'Cui Can', 'Yuanxi Sun']
    contact_email = 'bz1030@nyu.edu'

    def __init__(self, detection_model='/scratch/bz1030/data_ds_1008/detection/car_detection/runs/yolov3_p2v_yolo_2020-05-05_04-51-31_det_pt/best-model-6.pth',
                 segmentation_model='/scratch/bz1030/data_ds_1008/detection/car_detection/runs/yolov3_p2v_yolo_2020-05-06_05-38-14_seg_ssl/best-model-13.pth'):
        self.model_detection = pix2vox(None, False, seg=False, det=True)
        self.model_segmentation = pix2vox(None, False, seg=True, det=False)
        self.yolo = build_yolo()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.debug_det = detection_model.split('/')[:-1]
        self.debug_det = '/'.join(self.debug_det)
        #self.debug_det += '/debug_det'
        self.debug_seg = segmentation_model.split('/')[:-1]
        self.debug_seg = '/'.join(self.debug_seg)
        #self.debug_seg += '/debug_seg'
        create_dir(self.debug_det)
        create_dir(self.debug_seg)
        create_dir(self.debug_det + '/debug_det')
        create_dir(self.debug_seg + '/debug_seg')
        if self.use_cuda:
            state_dict1 = torch.load(detection_model)
            state_dict2 = torch.load(segmentation_model)
            self.model_detection.cuda()
            self.model_detection.load_state_dict(state_dict1)
            self.model_segmentation.cuda()
            self.model_segmentation.load_state_dict(state_dict2)
        else:
            state_dict1 = torch.load(detection_model)
            state_dict2 = torch.load(segmentation_model)
            self.model_detection.load_state_dict(state_dict1, map_location='cpu')
            self.model_segmentation.load_state_dict(state_dict2, map_location='cpu')
        self.model_segmentation.eval()
        self.model_detection.eval()

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        if type(samples) is tuple:
            samples = torch.stack(samples).to(self.device)
        else:
            samples = samples.to(self.device)
        with torch.no_grad():
            road_outputs, yolo_outputs = self.model_detection(samples)
            # compute loss
            output, yolo_loss, metrics = self.yolo(yolo_outputs, None, 416)
        # detections = to_cpu(torch.cat(output, 1))
        detections = output.cpu()
        #print('average/max conf:', detections[:, :, 6].mean().item(), detections[:, :, 6].max().item())
        threshold = detections[:, :, 6].max().item() * 0.9 # dynamics cut-off
        detections = nms_with_rot(detections, threshold, 0)
        return torch.stack(detections)

    def get_binary_road_map(self, samples):
        if type(samples) is tuple:
            samples = torch.stack(samples).to(self.device)
        else:
            samples = samples.to(self.device)
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        with torch.no_grad():
            road_outputs, _ = self.model_segmentation(samples)
            prob = torch.sigmoid(road_outputs)
        return prob > 0.5


def convert_coord(coords):
    coords_convert = torch.zeros(coords.shape)
    coords_convert[:, :, 0] = (coords[:, :, 0] - 400) / 10
    coords_convert[:, :, 1] = (coords[:, :, 1] - 400) / 10 * (-1)
    return coords_convert



class ModelLoaderSubmission():
    # Fill the information for your team
    team_name = 'some name'
    round_number = 1
    team_member = ['Bofei Zhang', 'Cui Can', 'Yuanxi Sun']
    contact_email = 'bz1030@nyu.edu'

    def __init__(self, detection_model='/scratch/bz1030/data_ds_1008/detection/car_detection/runs/yolov3_p2v_yolo_2020-05-05_04-51-31_det_pt/best-model-6.pth',
                 segmentation_model='/scratch/bz1030/data_ds_1008/detection/car_detection/runs/yolov3_p2v_yolo_2020-05-06_05-38-14_seg_ssl/best-model-13.pth'):
        self.model_detection = pix2vox(None, False, seg=False, det=True)
        self.model_segmentation = pix2vox(None, False, seg=True, det=False)
        self.yolo = build_yolo()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.debug_det = detection_model.split('/')[:-1]
        self.debug_det = '/'.join(self.debug_det)
        #self.debug_det += '/debug_det'
        self.debug_seg = segmentation_model.split('/')[:-1]
        self.debug_seg = '/'.join(self.debug_seg)
        #self.debug_seg += '/debug_seg'
        create_dir(self.debug_det)
        create_dir(self.debug_seg)
        create_dir(self.debug_det + '/debug_det')
        create_dir(self.debug_seg + '/debug_seg')
        self.threshold = 0.6
        if self.use_cuda:
            state_dict1 = torch.load(detection_model)
            state_dict2 = torch.load(segmentation_model)
            self.model_detection.cuda()
            self.model_detection.load_state_dict(state_dict1)
            self.model_segmentation.cuda()
            self.model_segmentation.load_state_dict(state_dict2)
        else:
            state_dict1 = torch.load(detection_model)
            state_dict2 = torch.load(segmentation_model)
            self.model_detection.load_state_dict(state_dict1, map_location='cpu')
            self.model_segmentation.load_state_dict(state_dict2, map_location='cpu')
        self.model_segmentation.eval()
        self.model_detection.eval()

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        if type(samples) is tuple:
            samples = torch.stack(samples).to(self.device)
        else:
            samples = samples.to(self.device)
        with torch.no_grad():
            road_outputs, yolo_outputs = self.model_detection(samples)
            # compute loss
            output, yolo_loss, metrics = self.yolo(yolo_outputs, None, 416)
        # detections = to_cpu(torch.cat(output, 1))
        detections = output.cpu()
        #print('average/max conf:', detections[:, :, 6].mean().item(), detections[:, :, 6].max().item())
        threshold = detections[:, :, 6].max().item() * self.threshold # dynamics cut-off
        detections = nms_with_rot(detections, threshold, 0)
        detections = torch.stack(detections)
        detections = convert_coord(detections)
        return detections

    def get_binary_road_map(self, samples):
        if type(samples) is tuple:
            samples = torch.stack(samples).to(self.device)
        else:
            samples = samples.to(self.device)
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        with torch.no_grad():
            road_outputs, _ = self.model_segmentation(samples)
            prob = torch.sigmoid(road_outputs)
        return prob > 0.5

class ModelLoader2():
    # Fill the information for your team
    team_name = 'some name'
    round_number = 1
    team_member = ['Bofei Zhang', 'Cui Can', 'Yuanxi Sun']
    contact_email = 'bz1030@nyu.edu'

    def __init__(self, detection_model='/scratch/bz1030/data_ds_1008/detection/car_detection/runs/center_net_2020-05-07_03-40-17_det_pt/best-model-1.pth',
                 segmentation_model='/scratch/bz1030/data_ds_1008/detection/model_weights/best-model-pix2vox.pth'):
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.model_config = 'center_net'
        self.model_detection = pix2vox(None, False, seg=True, det=False)
        self.model_segmentation = pix2vox_seg()
        self.yolo = CenterNetLoss()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.debug_det = detection_model.split('/')[:-1]
        self.debug_det = '/'.join(self.debug_det)
        #self.debug_det += '/debug_det'
        self.debug_seg = segmentation_model.split('/')[:-1]
        self.debug_seg = '/'.join(self.debug_seg)
        #self.debug_seg += '/debug_seg'
        create_dir(self.debug_det)
        create_dir(self.debug_seg)
        if self.use_cuda:
            state_dict1 = torch.load(detection_model)
            state_dict2 = torch.load(segmentation_model)
            self.model_detection.cuda()
            self.model_detection.load_state_dict(state_dict1)
            self.model_segmentation.cuda()
            self.model_segmentation.load_state_dict(state_dict2)
        else:
            state_dict1 = torch.load(detection_model, map_location='cpu')
            state_dict2 = torch.load(segmentation_model, map_location='cpu')
            self.model_detection.load_state_dict(state_dict1)
            self.model_segmentation.load_state_dict(state_dict2)
        self.model_segmentation.eval()
        self.model_detection.eval()

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        if type(samples) is tuple:
            samples = torch.stack(samples).to(self.device)
        else:
            samples = samples.to(self.device)
        with torch.no_grad():
            outputs, _ = self.model_detection(samples)
            # compute loss
            # output, yolo_loss, metrics = self.yolo(yolo_outputs, None)
        print(outputs.shape, outputs[:].mean(), outputs[:].max())
        # detections = to_cpu(torch.cat(output, 1))
        detections = output.squeeze().cpu().numpy()
        coords = extract_coords(detections, threshold = output[:, 0].max().item() * 0.999)
        #print('average/max conf:', detections[:, :, 6].mean().item(), detections[:, :, 6].max().item())
        return torch.stack(detections)

    def get_binary_road_map(self, samples):
        if type(samples) is tuple:
            samples = torch.stack(samples).to(self.device)
        else:
            samples = samples.to(self.device)
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        with torch.no_grad():
            road_outputs = self.model_segmentation(samples)
            prob = torch.sigmoid(road_outputs)
        return prob > 0.5
