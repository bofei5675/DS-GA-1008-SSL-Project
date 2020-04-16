import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from helper import convert_map_to_lane_map, convert_map_to_road_map

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
]


# The dataset class for unlabeled data.
class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index, first_dim, transform):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data
            first_dim ({'sample', 'image'}):
                'sample' will return [batch_size, NUM_IMAGE_PER_SAMPLE, 3, H, W]
                'image' will return [batch_size, 3, H, W] and the index of the camera [0 - 5]
                    CAM_FRONT_LEFT: 0
                    CAM_FRONT: 1
                    CAM_FRONT_RIGHT: 2
                    CAM_BACK_LEFT: 3
                    CAM_BACK.jpeg: 4
                    CAM_BACK_RIGHT: 5
            transform (Transform): The function to process the image
        """

        self.image_folder = image_folder
        self.scene_index = scene_index
        self.transform = transform

        assert first_dim in ['sample', 'image']
        self.first_dim = first_dim

    def __len__(self):
        if self.first_dim == 'sample':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE
        elif self.first_dim == 'image':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE

    def __getitem__(self, index):
        if self.first_dim == 'sample':
            scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
            sample_id = index % NUM_SAMPLE_PER_SCENE
            sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')

            images = []
            for image_name in image_names:
                image_path = os.path.join(sample_path, image_name)
                image = Image.open(image_path)
                images.append(self.transform(image))
            image_tensor = torch.stack(images)

            return image_tensor

        elif self.first_dim == 'image':
            scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
            sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
            image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

            image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name)

            image = Image.open(image_path)

            return self.transform(image), index % NUM_IMAGE_PER_SAMPLE


# The dataset class for labeled data.
class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, annotation_file, scene_index, transform, extra_info=True):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """

        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')

        images = []
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            images.append(self.transform(image))
        image_tensor = torch.cat(images, dim=0)
        data_entries = self.annotation_dataframe[
            (self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y', 'bl_y', 'br_y']].to_numpy()
        categories = data_entries.category_id.to_numpy()

        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)

        bounding_box = corners.reshape((-1, 2, 4))
        # convert to the grid final 800 x 800 grid
        bounding_box = bounding_box * 10 + 400
        num_bbs = bounding_box.shape[0]
        # leave the first column as an identifier
        labels = np.zeros((num_bbs, 5 + 1))
        # compute width by (x_fl, y_fl) - (x_fr, y_fr)
        width = (bounding_box[:, 0, 0] - bounding_box[:, 0, 1]) ** 2 + \
                (bounding_box[:, 1, 0] - bounding_box[:, 1, 1]) ** 2
        width = np.sqrt(width)
        # compute height in a same way
        height = (bounding_box[:, 0, 1] - bounding_box[:, 0, 3]) ** 2 + \
                (bounding_box[:, 1, 1] - bounding_box[:, 1, 3]) ** 2
        height = np.sqrt(height)
        # get x center
        x_center = (bounding_box[:, 0, 0] + bounding_box[:, 0, 1]) / 2
        y_center = (bounding_box[:, 1, 0] + bounding_box[:, 1, 1]) / 2
        # normalize by 800
        x_center /= 800
        y_center /= 800
        width /= 800
        height /= 800
        # column order: category, x, y, width, height
        labels[:, 1] = categories
        labels[:, 2] = x_center
        labels[:, 3] = y_center
        labels[:, 4] = width
        labels[:, 5] = height
        labels = torch.as_tensor(labels)
        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)

            extra = {}
            extra['action'] = torch.as_tensor(actions)
            extra['ego_image'] = ego_image
            extra['lane_image'] = lane_image

            return image_tensor, labels, road_image, extra

        else:
            return image_tensor, labels, road_image

