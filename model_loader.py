"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from car_detection.models import Darknet
from car_detection.utils.utils import non_max_suppression
from roadmap_segmentation.pix2vox import pix2vox
# import your model class
# import ...

# Put your transform function here, we will use it for our dataloader
def get_transform():
    transform = transforms.Compose([transforms.Resize((416, 416)),
                                    transforms.ToTensor()])
    return transform


class ModelLoader():
    # Fill the information for your team
    team_name = 'some name'
    round_number = 1
    team_member = ['Bofei Zhang', 'Cui Can', 'Yuanxi Sun']
    contact_email = 'bz1030@nyu.edu'

    def __init__(self, detection_model='./best-model-1.pth',
                 segmentation_model='./roadmap_segmentation/runs/pix2vox_2020-04-26_01-18-51/best-model-15.pth'):
        self.model_detection = Darknet('./car-detection/config/yolov3.cfg', 416)
        self.model_segmentation = pix2vox()
        self.use_cuda = torch.cuda.is_available()

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

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        sample = torch.stack(sample)
        if self.cuda:
            sample = sample.cuda()
        with torch.no_grad():
            yolo_outputs = []
            outputs = {0: None, 1: None, 2: None}
            for image_idx in range(6):
                image = sample[:, image_idx].squeeze()
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
                output, layer_loss, metrics = yolo[output_idx](output, target, 416)
                detections.append(output)
                loss += layer_loss
            detections = to_cpu(torch.cat(detections, 1))
            print('average/max conf:', detections[:, :, 6].mean().item(), detections[:, :, 6].max().item())
            print('loss:', loss.item(), target.shape)
            detections = non_max_suppression(detections, 0.5, 0)
        return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        outputs = model(samples).squeeze(dim=1)
        # compute loss
        prob = torch.sigmoid(outputs)
        return prob > 0.5


ModelLoader()