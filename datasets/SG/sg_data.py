# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper class and functions for loading Shape Grammar objects

Author: Jincheng Zhang
Date: July 2022

Modified from code written by
Author: Charles R. Qi
Date: December 2018

Note: removed unused code for frustum preparation.
Changed a way for data visualization (removed dependency on mayavi).
Load depth with scipy.io
"""

import os
import sys
import numpy as np
import sys
import cv2
import argparse
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../'))
from utils import pc_util
from datasets.SUNRGBD import sunrgbd_utils

CLASS_WHITELIST = ['bed', 'table', 'sofa', 'chair', 'bookshelf']
class2id = {'table': 0, 'chair': 1, 'bookshelf': 2, 'bed': 3, 'sofa': 4, 'dresser': 5}
id2class = {class2id[t]: t for t in class2id}


class sg_object(object):
    """ Load and parse object data """

    def __init__(self, root_dir, split='training'):
        self.root_dir = root_dir
        self.split = split
        assert (self.split == 'training')
        self.split_dir = os.path.join(root_dir)

        if split == 'training':
            self.num_samples = 500
        elif split == 'testing':
            self.num_samples = 500
        else:
            print('Unknown split: %s' % split)
            exit(-1)

    def __len__(self):
        return self.num_samples

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.split_dir, '%d/bb3d.txt' % idx)
        return load_label(label_filename)

    def get_pc(self, idx):
        pc_filename = os.path.join(self.split_dir, '%d/pointcloud.txt' % idx)
        return load_points(pc_filename)


class SGObject3d(object):
    def __init__(self, line):
        data = line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        self.class_id = data[0]
        self.centroid = np.array([data[1], data[2], data[3]])
        self.l = data[4]
        self.w = data[5]
        self.h = data[6]
        self.roll = data[7]
        self.pitch = data[8]
        self.yaw = data[9]
        self.sg_para = np.array([data[10], data[11], data[12], data[13], data[14], data[15], data[16]])


def data_viz(data_dir, dump_dir=os.path.join(BASE_DIR, 'data_viz_dump')):
    """ Examine and visualize SUN RGB-D data. """
    sg = sg_object(data_dir)
    idxs = np.array(range(1, len(sg) + 1))
    np.random.seed(0)
    np.random.shuffle(idxs)
    for idx in range(len(sg)):
        data_idx = idxs[idx]
        print('-' * 10, 'data index: ', data_idx)
        pc = sg.get_pc(data_idx)
        print('Point cloud shape:', pc.shape)

        # Load box labels
        objects = sg.get_label_objects(data_idx)
        print('Objects:', objects)

        # Dump OBJ files for the colored point cloud
        # for num_point in [10000, 20000, 40000, 80000]:
        #     sampled_pcrgb = pc_util.random_sampling(pc, num_point)
        #     pc_util.write_ply_rgb(sampled_pcrgb[:, 0:3],
        #                           (sampled_pcrgb[:, 3:] * 256).astype(np.int8),
        #                           os.path.join(dump_dir, 'pcrgb_%dk.obj' % (num_point // 1000)))
        # Dump OBJ files for 3D bounding boxes
        # l,w,h correspond to dx,dy,dz
        # heading angle is from +X rotating towards -Y
        # (+X is degree, -Y is 90 degrees)
        oriented_boxes = []
        for obj in objects:
            obb = np.zeros(9)
            obb[0:3] = obj.centroid
            # Some conversion to map with default setting of w,l,h
            # and angle in box dumping
            obb[3:6] = np.array([obj.l, obj.w, obj.h]) * 2
            obb[6:9] = np.array([obj.roll, obj.pitch, obj.yaw])
            print('Object cls, roll, pitch, yaw, l, w, h:',
                  obj.class_id, obj.roll, obj.pitch, obj.yaw, obj.l, obj.w, obj.h)
            oriented_boxes.append(obb)
        if len(oriented_boxes) > 0:
            oriented_boxes = np.vstack(tuple(oriented_boxes))
            pc_util.write_oriented_bbox(oriented_boxes,
                                        os.path.join(dump_dir, 'obbs.ply'))
        else:
            print('-' * 30)
            continue

        # Draw 3D boxes on depth points
        box3d = []
        for obj in objects:
            corners_3d = compute_box_3d(obj)
            print('Corners 3D: ', corners_3d)
            box3d.append(corners_3d)
        pc_box3d = np.concatenate(box3d, 0)
        print(pc_box3d.shape)
        pc_util.write_ply(pc_box3d, os.path.join(dump_dir, 'box3d_corners.ply'))
        print('-' * 30)
        print('Point clouds and bounding boxes saved to PLY files under %s' % dump_dir)
        print('Type anything to continue to the next sample...')
        input()


def extract_sg_data(idx_filename, split, output_folder, num_point=20000,
                    type_whitelist=None, skip_empty_scene=True):
    """ Extract scene point clouds and
   bounding boxes (centroids, box sizes, heading angles, semantic classes).
   Dumped point clouds and boxes are in upright depth coord.

   Args:
       type_whitelist:
       output_folder:
       num_point:
       idx_filename: a TXT file where each line is an int number (index)
       split: training or testing
       skip_empty_scene: if True, skip scenes that contain no object (no objet in whitelist)

   Dumps:
       <id>_pc.npz of (N,6) where N is for number of subsampled points and 6 is
           for XYZ and RGB (in 0~1) in upright depth coord
       <id>_bbox.npy of (K,8) where K is the number of objects, 8 is for
           centroids (cx,cy,cz), dimension (l,w,h), heanding_angle and semantic_class
       <id>_votes.npz of (N,10) with 0/1 indicating whether the point belongs to an object,
           then three sets of GT votes for up to three objects. If the point is only in one
           object's OBB, then the three GT votes are the same.
   """
    if type_whitelist is None:
        type_whitelist = CLASS_WHITELIST
    dataset = sg_object("/home/jzhang72/Downloads/sharp-kt/dataset", split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        objects = dataset.get_label_objects(data_idx)

        # Skip scenes with 0 object
        if skip_empty_scene and (len(objects) == 0 or
                                 len([obj for obj in objects if id2class[int(obj.class_id)] in type_whitelist]) == 0):
            continue

        object_list = []
        for obj in objects:
            if id2class[int(obj.class_id)] not in type_whitelist:
                continue
            obb = np.zeros(17)
            obb[0:3] = obj.centroid
            # Note that compared with that in data_viz, we do not time 2 to l,w.h
            # neither do we flip the heading angle
            obb[3:6] = np.array([obj.l, obj.w, obj.h])
            obb[6:9] = np.array([obj.roll, obj.pitch, obj.yaw])
            obb[9:16] = obj.sg_para
            obb[16] = obj.class_id
            object_list.append(obb)
        if len(object_list) == 0:
            obbs = np.zeros((0, 17))
        else:
            obbs = np.vstack(object_list)  # (K, id+center+size+rotation+sg)

        pc_upright = dataset.get_pc(data_idx)
        pc_upright_subsampled = pc_util.random_sampling(pc_upright, num_point)

        np.savez_compressed(os.path.join(output_folder, '%06d_pc.npz' % data_idx),
                            pc=pc_upright_subsampled)
        np.save(os.path.join(output_folder, '%06d_bbox.npy' % data_idx), obbs)


def load_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [SGObject3d(line) for line in lines]
    return objects


def load_points(pc_filename):
    pc = np.loadtxt(pc_filename)
    return pc


def compute_box_3d(obj):
    center = obj.centroid

    # compute rotational matrix around yaw axis
    R = np.matmul(sunrgbd_utils.rotx(obj.roll), sunrgbd_utils.roty(obj.pitch))
    R = np.matmul(R, sunrgbd_utils.rotz(obj.yaw))
    # b,a,c = dimension
    # print R, a,b,c

    # 3d bounding box dimensions
    l = obj.l  # along heading arrow
    w = obj.w  # perpendicular to heading arrow
    h = obj.h

    # rotate and translate 3d bounding box
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]

    return np.transpose(corners_3d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true', help='Run data visualization.')
    parser.add_argument('--compute_median_size', action='store_true', help='Compute median 3D bounding box sizes for each class.')
    args = parser.parse_args()

    if args.viz:
        data_viz("/home/jzhang72/Downloads/sharp-kt/dataset")
        exit()

    extract_sg_data(os.path.join(BASE_DIR, 'train_data_idx.txt'),
                    split='training',
                    output_folder=os.path.join(BASE_DIR, 'sg_pc_bbox_train'),
                    num_point=50000, skip_empty_scene=True)
    extract_sg_data(os.path.join(BASE_DIR, 'val_data_idx.txt'),
                    split='training',
                    output_folder=os.path.join(BASE_DIR, 'sg_pc_bbox_val'),
                    num_point=50000, skip_empty_scene=True)
