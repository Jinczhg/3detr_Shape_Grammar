# Copyright (c) Facebook, Inc. and its affiliates.


"""
Modified from https://github.com/facebookresearch/votenet
Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Jincheng Zhang
Date: 2022

Modified from code written by
Author: Charles R. Qi
Date: 2019

"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio  # to load .mat files for depth points

import utils.pc_util as pc_util
from utils.random_cuboid import RandomCuboid
from utils.pc_util import shift_scale_points, scale_points
from utils.box_util import (
    flip_axis_to_camera_tensor,
    get_3d_box_batch_tensor,
    flip_axis_to_camera_np,
    get_3d_box_batch_np,
)

MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1
DATA_PATH = "/home/jzhang72/PycharmProjects/3detr_Shape_Grammar/datasets/SG/sg_pc_bbox"  ## Replace with path to dataset


# def sg_flip_axis_to_camera_tensor(pc):
#     """Flip X-left,Y-up,Z-forward to X-right,Y-down,Z-forward
#    Input and output are both (N,3) array
#    """
#     pc2 = torch.clone(pc)
#     pc2[..., 0] *= -1
#     pc2[..., 1] *= -1
#     return pc2
#
#
# def sg_flip_axis_to_camera_np(pc):
#     """Flip X-left,Y-up,Z-forward to X-right,Y-down,Z-forward
#    Input and output are both (N,3) array
#    """
#     pc2 = pc.copy()
#     pc2[..., 0] *= -1
#     pc2[..., 1] *= -1
#     return pc2


class SGDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 3
        self.num_angle_bin = 12
        self.max_num_obj = 64
        self.sg_para_number = 7  # number of shape grammar parameters to be estimated
        self.type2class = {
            'table': 0, 'chair': 1, 'bookshelf': 2  # , 'bed': 3, 'sofa': 4, 'dresser': 5,
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.type2onehotclass = {
            'table': 0, 'chair': 1, 'bookshelf': 2  # , 'bed': 3, 'sofa': 4, 'dresser': 5,
        }

    def angle2class(self, angle):
        """Convert continuous angle to discrete class
       [optinal] also small regression number from
       class center angle to current angle.

       angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
       returns class [0,1,...,N-1] and a residual number such that
           class*(2pi/N) + number = angle
       """
        num_class = self.num_angle_bin
        angle = angle % (2 * np.pi)
        assert angle >= 0 and angle <= 2 * np.pi
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (
                class_id * angle_per_class + angle_per_class / 2
        )
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        """Inverse function to angle2class"""
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return self.class2angle_batch(pred_cls, residual, to_label_format)

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle_roll, box_angle_pitch, box_angle_yaw):
        # SG dataset doesn't need the flipping step. Original data is already in the cam coordinate
        # box_center_upright = sg_flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle_roll, box_angle_pitch, box_angle_yaw, box_center_unnorm)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle_roll, box_angle_pitch, box_angle_yaw):
        # SG dataset doesn't need the flipping step. Original data is already in the cam coordinate
        # box_center_upright = sg_flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle_roll, box_angle_pitch, box_angle_yaw, box_center_unnorm)
        return boxes

    def my_compute_box_3d(self, center, size, rotation):
        R = np.matmul(pc_util.rotz(rotation[2]), pc_util.roty(rotation[1]))
        R = np.matmul(R, pc_util.rotx(rotation[0]))
        l, w, h = size
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)


class SGDetectionDataset(Dataset):
    def __init__(
            self,
            dataset_config,
            split_set="train",
            root_dir=None,
            num_points=20000,
            use_color=False,
            use_height=False,
            augment=False,
            use_random_cuboid=True,
            random_cuboid_min_points=30000,
    ):
        assert num_points <= 50000
        assert split_set in ["train", "val", "trainval"]
        self.dataset_config = dataset_config

        if root_dir is None:
            root_dir = DATA_PATH

        self.data_path = root_dir + "_%s" % (split_set)

        if split_set in ["train", "val"]:
            self.scan_names = sorted(
                list(
                    set([os.path.basename(x)[0:6] for x in os.listdir(self.data_path)])
                )
            )
        elif split_set in ["trainval"]:
            # combine names from both
            sub_splits = ["train", "val"]
            all_paths = []
            for sub_split in sub_splits:
                data_path = self.data_path.replace("trainval", sub_split)
                basenames = sorted(
                    list(set([os.path.basename(x)[0:6] for x in os.listdir(data_path)]))
                )
                basenames = [os.path.join(data_path, x) for x in basenames]
                all_paths.extend(basenames)
            all_paths.sort()
            self.scan_names = all_paths

        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(
            min_points=random_cuboid_min_points,
            aspect=0.75,
            min_crop=0.75,
            max_crop=1.0,
        )
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.max_num_obj = 64

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        if scan_name.startswith("/"):
            scan_path = scan_name
        else:
            scan_path = os.path.join(self.data_path, scan_name)
        point_cloud = np.load(scan_path + "_pc.npz")["pc"]  # Nx6
        bboxes = np.load(scan_path + "_bbox.npy")  # K,8

        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]
        else:
            assert point_cloud.shape[1] == 6
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = point_cloud[:, 3:] - MEAN_COLOR_RGB

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate(
                [point_cloud, np.expand_dims(height, 1)], 1
            )  # (N,4) or (N,7)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                bboxes[:, 0] = -1 * bboxes[:, 0]
                bboxes[:, 7] = np.pi - bboxes[:, 7]

            # Rotation along up-axis/Y-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = pc_util.roty(rot_angle)

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 7] += rot_angle  # pitch

            # Augment RGB color
            if self.use_color:
                rgb_color = point_cloud[:, 3:6] + MEAN_COLOR_RGB
                rgb_color *= (
                        1 + 0.4 * np.random.random(3) - 0.2
                )  # brightness change for each channel
                rgb_color += (
                        0.1 * np.random.random(3) - 0.05
                )  # color shift for each channel
                rgb_color += np.expand_dims(
                    (0.05 * np.random.random(point_cloud.shape[0]) - 0.025), -1
                )  # jittering on each pixel
                rgb_color = np.clip(rgb_color, 0, 1)
                # randomly drop out 30% of the points' colors
                rgb_color *= np.expand_dims(
                    np.random.random(point_cloud.shape[0]) > 0.3, -1
                )
                point_cloud[:, 3:6] = rgb_color - MEAN_COLOR_RGB

            # Augment point cloud scale: 0.85x-1.15x
            # Not scaling the point cloud because the changes that scaling brings to the shape grammar parameters is indeterminate
            # scale_ratio = np.random.random() * 0.3 + 0.85
            # scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            # point_cloud[:, 0:3] *= scale_ratio
            # bboxes[:, 0:3] *= scale_ratio
            # bboxes[:, 3:6] *= scale_ratio

            # if self.use_height:
            #     point_cloud[:, -1] *= scale_ratio[0, 0]

            if self.use_random_cuboid:
                point_cloud, bboxes, _ = self.random_cuboid_augmentor(
                    point_cloud, bboxes
                )

        # ------------------------------- LABELS ------------------------------
        roll_angle_classes = np.zeros((self.max_num_obj,), dtype=np.float32)
        roll_angle_residuals = np.zeros((self.max_num_obj,), dtype=np.float32)
        pitch_angle_classes = np.zeros((self.max_num_obj,), dtype=np.float32)
        pitch_angle_residuals = np.zeros((self.max_num_obj,), dtype=np.float32)
        yaw_angle_classes = np.zeros((self.max_num_obj,), dtype=np.float32)
        yaw_angle_residuals = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_sizes = np.zeros((self.max_num_obj, 3), dtype=np.float32)
        label_mask = np.zeros((self.max_num_obj))
        label_mask[0: bboxes.shape[0]] = 1
        # max_bboxes = np.zeros((self.max_num_obj, 10))
        # max_bboxes[0 : bboxes.shape[0], :] = bboxes
        sg_paras = np.zeros((self.max_num_obj, 7), dtype=np.float32)        # this number is SGDatasetConfig.sg_para_number

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((self.max_num_obj, 6))

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[16]
            box3d_size = bbox[3:6] * 2
            sg_para = bbox[9:16]
            sg_paras[i, :] = sg_para
            raw_sizes[i, :] = box3d_size
            roll_angle_class, roll_angle_residual = self.dataset_config.angle2class(bbox[6])
            roll_angle_classes[i] = roll_angle_class
            roll_angle_residuals[i] = roll_angle_residual
            pitch_angle_class, pitch_angle_residual = self.dataset_config.angle2class(bbox[7])
            pitch_angle_classes[i] = pitch_angle_class
            pitch_angle_residuals[i] = pitch_angle_residual
            yaw_angle_class, yaw_angle_residual = self.dataset_config.angle2class(bbox[8])
            yaw_angle_classes[i] = yaw_angle_class
            yaw_angle_residuals[i] = yaw_angle_residual
            corners_3d = self.dataset_config.my_compute_box_3d(
                bbox[0:3], bbox[3:6], bbox[6:9]
            )
            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            target_bbox = np.array(
                [
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    (zmin + zmax) / 2,
                    xmax - xmin,
                    ymax - ymin,
                    zmax - zmin,
                ]
            )
            target_bboxes[i, :] = target_bbox

        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )

        point_cloud_dims_min = point_cloud.min(axis=0)
        point_cloud_dims_max = point_cloud.max(axis=0)

        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        def scale_points_sg(pred_sg, factor_sg):
            if pred_sg.ndim == 7:                       # this number is SGDatasetConfig.sg_para_number
                factor_sg = factor_sg[:, None]
            scaled_sg = pred_sg * factor_sg[:, None, :]
            return scaled_sg

        mult_factor_sg = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        sg_paras_normalized = scale_points_sg(
            sg_paras.astype(np.float32)[None, ...],
            factor_sg=1.0 / mult_factor_sg[None, ...],
        )
        sg_paras_normalized = sg_paras_normalized.squeeze(0)

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]

        # re-encode angles to be consistent with VoteNet eval
        roll_angle_classes = roll_angle_classes.astype(np.int64)
        roll_angle_residuals = roll_angle_residuals.astype(np.float32)
        pitch_angle_classes = pitch_angle_classes.astype(np.int64)
        pitch_angle_residuals = pitch_angle_residuals.astype(np.float32)
        yaw_angle_classes = yaw_angle_classes.astype(np.int64)
        yaw_angle_residuals = yaw_angle_residuals.astype(np.float32)
        roll_raw_angles = self.dataset_config.class2angle_batch(
            roll_angle_classes, roll_angle_residuals
        )
        pitch_raw_angles = self.dataset_config.class2angle_batch(
            pitch_angle_classes, pitch_angle_residuals
        )
        yaw_raw_angles = self.dataset_config.class2angle_batch(
            yaw_angle_classes, yaw_angle_residuals
        )

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            roll_raw_angles.astype(np.float32)[None, ...],
            pitch_raw_angles.astype(np.float32)[None, ...],
            yaw_raw_angles.astype(np.float32)[None, ...]
        )
        box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(np.float32)
        target_bboxes_semcls = np.zeros((self.max_num_obj))
        target_bboxes_semcls[0: bboxes.shape[0]] = bboxes[:, -1]  # from 0 to 9
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles_roll"] = roll_raw_angles.astype(np.float32)
        ret_dict["gt_angle_class_label_roll"] = roll_angle_classes
        ret_dict["gt_angle_residual_label_roll"] = roll_angle_residuals
        ret_dict["gt_box_angles_pitch"] = pitch_raw_angles.astype(np.float32)
        ret_dict["gt_angle_class_label_pitch"] = pitch_angle_classes
        ret_dict["gt_angle_residual_label_pitch"] = pitch_angle_residuals
        ret_dict["gt_box_angles_yaw"] = yaw_raw_angles.astype(np.float32)
        ret_dict["gt_angle_class_label_yaw"] = yaw_angle_classes
        ret_dict["gt_angle_residual_label_yaw"] = yaw_angle_residuals
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(np.float32)
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(np.float32)
        ret_dict["sg_para"] = sg_paras_normalized.astype(np.float32)
        return ret_dict
