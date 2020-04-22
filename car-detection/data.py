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

WIDTH_ORIGIN = 306
HEIGHT_ORIGIN = 256
RESIZE_DIM = 256 # make sure it's square
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
            image = self.transform(image)
            image = image.unsqueeze(0)
            images.append(image)
        image_tensor = torch.cat(images, dim=0)
        data_entries = self.annotation_dataframe[
            (self.annotation_dataframe['scene'] == scene_id) & (
                    self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y', 'bl_y', 'br_y']] \
            .to_numpy().reshape(-1, 2, 4)
        labels = self.build_labels(corners)

        categories = data_entries.category_id.to_numpy()
        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)
        labels[:, 1] = categories
        # column order: category, x, y, width, height; Plan to transpose the label
        # reserve first column for idx of each instance.
        labels = torch.as_tensor(labels)
        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)

            extra = {}
            extra['action'] = torch.as_tensor(actions)
            extra['ego_image'] = ego_image
            extra['lane_image'] = lane_image
            extra['file_path'] = sample_path

            return image_tensor, labels, road_image, extra

        else:
            return image_tensor, labels, road_image


    def build_labels(self, corners):
        labels = np.zeros((corners.shape[0], 6 + 2))
        for idx, corner in enumerate(corners):
            point_squence = np.stack([corner[:, 0], corner[:, 1], corner[:, 3], corner[:, 2], corner[:, 0]])
            x = point_squence.T[0] * 10 + 400
            y = - point_squence.T[1] * 10 + 400
            xc = (x.min().item() + x.max().item()) / 2
            yc = (y.min().item() + y.max().item()) / 2
            w = np.abs(x.min().item() - x.max().item())
            h = np.abs(y.min().item() - y.max().item())
            # normalize to 0-1
            labels[idx, 2] = xc / 800
            labels[idx, 3] = yc / 800
            labels[idx, 4] = w / 800
            labels[idx, 5] = h / 800
            # build directions
            # compute angle
            vector1 = np.array([x[0], y[0]])
            vector2 = np.array([x[1], y[1]])
            cxy = np.array([xc, yc])
            direction = (vector1 - cxy) + (vector2 - cxy)
            direction = direction / np.linalg.norm(direction)
            labels[idx, 6:] = direction
        return labels

# The dataset class for labeled data.
class LabeledDatasetLarge(torch.utils.data.Dataset):
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

        total_width = RESIZE_DIM * 3
        total_height = RESIZE_DIM * 2
        new_img = Image.new('RGB', (total_width, total_height))
        x_offset = 0
        for image_name in image_names[: 3]:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            image = image.resize((RESIZE_DIM, RESIZE_DIM))
            new_img.paste(image, (x_offset, 0))
            x_offset += RESIZE_DIM
        x_offset = 0
        y_offset = RESIZE_DIM
        for image_name in image_names[3:]:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            image = image.resize((RESIZE_DIM, RESIZE_DIM))
            # keep spatial information
            if 'CAM_BACK.jpeg' == image_name:
                image = image.rotate(-180)
            elif 'CAM_BACK_LEFT.jpeg' == image_name:
                image = image.rotate(180)
            else:
                image = image.rotate(-180)
            new_img.paste(image, (x_offset, y_offset))
            x_offset += RESIZE_DIM
        #new_img.save(f'../foo/{scene_id}_{sample_id}.jpg')
        image_tensor = torch.as_tensor(self.transform(new_img))
        # print(image_tensor.shape)
        data_entries = self.annotation_dataframe[
            (self.annotation_dataframe['scene'] == scene_id) & (
                        self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y', 'bl_y', 'br_y']]\
            .to_numpy().reshape(-1, 2, 4)
        labels = np.zeros((corners.shape[0], 6))
        for idx, corner in  enumerate(corners):
            point_squence = np.stack([corner[:, 0], corner[:, 1], corner[:, 3], corner[:, 2], corner[:, 0]])
            x = point_squence.T[0] * 10 + 400
            y = - point_squence.T[1] * 10 + 400
            xc = (x[0] + x[2]) / 2
            yc = (y[0] + y[1]) / 2
            w = np.abs(x[0] - x[2]) / 2
            h = np.abs(y[0] - y[1]) / 2
            # normalize to 0-1
            labels[idx, 2] = xc / 800
            labels[idx, 3] = yc / 800
            labels[idx, 4] = w / 800
            labels[idx, 5] = h / 800

        categories = data_entries.category_id.to_numpy()
        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)
        labels[:, 1] = categories
        # column order: category, x, y, width, height; Plan to transpose the label
        # reserve first column for idx of each instance.
        labels = torch.as_tensor(labels)
        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)

            extra = {}
            extra['action'] = torch.as_tensor(actions)
            extra['ego_image'] = ego_image
            extra['lane_image'] = lane_image
            extra['file_path'] = sample_path


            return image_tensor, labels, road_image, extra

        else:
            return image_tensor, labels, road_image

