"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from car_detection.models import build_yolo
from car_detection.utils.utils import non_max_suppression, to_cpu, nms_with_rot
from car_detection.pix2vox import pix2vox
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


class ModelLoader():
    # Fill the information for your team
    team_name = 'some name'
    round_number = 1
    team_member = ['Bofei Zhang', 'Can Cui', 'Yuanxi Sun']
    contact_email = 'bz1030@nyu.edu'

    def __init__(self, detection_model='./car_detection/runs/p2v_yolo_2020-04-30_11-53-15_det_pt/best-model-3.pth',
                 segmentation_model='./model_weights/best-model-pix2vox.pth'):
        self.model_detection = pix2vox(det=True, seg=False)
        self.model_segmentation = pix2vox(det=False, seg=True)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        if self.use_cuda:
            state_dict1 = torch.load(detection_model)
            state_dict2 = torch.load(segmentation_model)
            self.model_detection.cuda()
            self.model_detection.load_state_dict(state_dict1)
            self.model_segmentation.cuda()
            self.model_segmentation.load_state_dict(state_dict2)
            self.yolo = build_yolo()
        else:
            state_dict1 = torch.load(detection_model)
            state_dict2 = torch.load(segmentation_model)
            self.model_detection.load_state_dict(state_dict1, map_location='cpu')
            self.model_segmentation.load_state_dict(state_dict2, map_location='cpu')
            self.yolo = build_yolo()
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
        yolo = self.model_detection.yolo_layers
        with torch.no_grad():
            outputs = {0: None, 1: None, 2: None}
            for image_idx in range(6):
                image = samples[:, image_idx].squeeze(dim=1)
                yolo_output = self.model_detection(image)

                for output_idx in range(3):
                    if outputs[output_idx] is None:
                        outputs[output_idx] = yolo_output[output_idx]
                    else:
                        outputs[output_idx] = torch.cat((outputs[output_idx],
                                                         yolo_output[output_idx]),
                                                        dim=1)
            # compute loss
            loss = 0
            detections = []
            for output_idx in range(3):
                output = outputs[output_idx]
                output = self.model_detection.module_list[-3 + output_idx](output)
                output, layer_loss, metrics = yolo[output_idx](output, None, 416)
                detections.append(output)
                loss += layer_loss
            detections = to_cpu(torch.cat(detections, 1))
            #print('average/max conf:', detections[:, :, 6].mean().item(), detections[:, :, 6].max().item())
            detections = nms_with_rot(detections, 0.7, 0)
        return torch.stack(detections)

    def get_binary_road_map(self, samples):
        if type(samples) is tuple:
            samples = torch.stack(samples).to(self.device)
        else:
            samples = samples.to(self.device)
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        with torch.no_grad():
            outputs = self.model_segmentation(samples).squeeze(dim=1)
            # compute loss
            prob = torch.sigmoid(outputs)
        return prob > 0.5