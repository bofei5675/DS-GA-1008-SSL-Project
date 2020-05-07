import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def gaussian_kernel(inp_size, xc, yc, w, h, direction):
    center = np.array([xc, yc])
    direction_perp = np.array([-direction[1], direction[0]])
    u = np.stack([direction, direction_perp]).T
    radius1 = max(w,h) / 2
    radius2 = min(w,h) / 2
    s = np.diag([ radius1 ** 2, radius2 ** 2])
    cov = u.dot(s).dot(u.T)
    cov_inv = np.linalg.inv(cov)
    x = np.arange(0, inp_size, 1)
    y = np.arange(0, inp_size, 1)
    xx, yy = np.meshgrid(x, y)
    pos = np.stack([xx, yy]).reshape(2, -1).T
    diff = pos - center
    outputs = np.tensordot(diff, cov_inv, axes=(-1,-1)) * diff
    outputs = outputs.sum(axis=1)
    outputs = np.exp(-0.5 * outputs).reshape(inp_size, inp_size)
    return outputs


def convert_map_to_lane_map(ego_map, binary_lane):
    mask = (ego_map[0,:,:] == ego_map[1,:,:]) * (ego_map[1,:,:] == ego_map[2,:,:]) + (ego_map[0,:,:] == 250 / 255)

    if binary_lane:
        return (~ mask)
    return ego_map * (~ mask.view(1, ego_map.shape[1], ego_map.shape[2]))

def convert_map_to_road_map(ego_map):
    mask = (ego_map[0,:,:] == 1) * (ego_map[1,:,:] == 1) * (ego_map[2,:,:] == 1)

    return (~mask)


def collate_fn2(batch):
    try:
        image_tensor, labels, road_image = list(zip(*batch))
        # Add sample index to labels
        for i, boxes in enumerate(labels):
            boxes[:, 0] = i
        labels = torch.cat(labels, 0)
        return image_tensor, labels, road_image
    except Exception as e:
        image_tensor, labels, road_image, extra = list(zip(*batch))
        # Add sample index to labels
        for i, boxes in enumerate(labels):
            boxes[:, 0] = i
        labels = torch.cat(labels, 0)
        return image_tensor, labels, road_image, extra

def collate_fn_cn(batch):
    try:
        image_tensor, labels, road_image = list(zip(*batch))
        # Add sample index to labels
        for i, boxes in enumerate(labels):
            boxes[:, 0] = i
        labels = torch.cat(labels, 0)
        return image_tensor, labels, road_image
    except Exception as e:
        image_tensor, labels, road_image, extra = list(zip(*batch))
        # Add sample index to labels
        labels = torch.stack(labels)
        image_tensor = torch.stack(image_tensor)
        road_image = torch.stack(road_image)
        return image_tensor, labels, road_image, extra


def collate_fn(batch):
    return tuple(zip(*batch))

def draw_box(ax, corners, color):
    point_squence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])
    
    # the corners are in meter and time 10 will convert them in pixels
    # Add 400, since the center of the image is at pixel (400, 400)
    # The negative sign is because the y axis is reversed for matplotlib
    ax.plot(point_squence.T[0] * 10 + 400, -point_squence.T[1] * 10 + 400, color=color)





