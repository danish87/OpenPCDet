import numpy as np
import torch

from ...utils import  common_utils


def random_flip_along_x(gt_boxes, points, enable_=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5]) if enable_ is None else enable_
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points, enable


def random_flip_along_y(gt_boxes, points, enable_=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5]) if enable_ is None else enable_
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points, enable


def global_rotation(gt_boxes, points, rot_range, rot_angle_=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1]) if rot_angle_ is None else rot_angle_
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points, noise_rotation


def global_scaling(gt_boxes, points, scale_range, scale_=None):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1]) if scale_ is None else scale_
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points, noise_scale


def random_flip_along_x_bbox(gt_boxes, enables):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    for i in range(len(enables)):
        if enables[i]:
            valid_mask = torch.logical_not(torch.all(gt_boxes[i] == 0, dim=-1))
            gt_boxes[i, valid_mask, 1] = -gt_boxes[i, valid_mask, 1]
            gt_boxes[i, valid_mask, 6] = -gt_boxes[i, valid_mask, 6]

            # if gt_boxes.shape[2] > 7:
            #     gt_boxes[i, :, 8] = -gt_boxes[i, :, 8]

    return gt_boxes


def random_flip_along_y_bbox(gt_boxes, enables):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    for i in range(len(enables)):
        if enables[i]:
            valid_mask = torch.logical_not(torch.all(gt_boxes[i] == 0, dim=-1))
            gt_boxes[i, valid_mask, 0] = -gt_boxes[i, valid_mask, 0]
            gt_boxes[i, valid_mask, 6] = -(gt_boxes[i, valid_mask, 6] + np.pi)

            # if gt_boxes.shape[2] > 7:
            #     gt_boxes[i, :, 7] = -gt_boxes[i, :, 7]

    return gt_boxes


def global_rotation_bbox(gt_boxes, rotations):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    for i in range(len(rotations)):
        rotation = rotations[i:i+1]
        valid_mask = torch.logical_not(torch.all(gt_boxes[i] == 0, dim=-1))
        gt_boxes[i, valid_mask, 0:3] = common_utils.rotate_points_along_z(gt_boxes[i:i+1, valid_mask, 0:3], rotation)[0]
        gt_boxes[i, valid_mask, 6] += rotation
        # if gt_boxes.shape[2] > 7:
        #     gt_boxes[i, :, 7:9] = common_utils.rotate_points_along_z(
        #         np.hstack((gt_boxes[i, :, 7:9], np.zeros((gt_boxes.shape[1], 1))))[np.newaxis, :, :],
        #         rotation)
        #     )[0][:, 0:2]

    return gt_boxes


def global_scaling_bbox(gt_boxes, scales):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    for i in range(len(scales)):
        valid_mask = torch.logical_not(torch.all(gt_boxes[i] == 0, dim=-1))
        gt_boxes[i, valid_mask, :6] *= scales[i]
    return gt_boxes
