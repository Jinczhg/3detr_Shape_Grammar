#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python ../main.py \
--dataset_name sg \
--max_epoch 400 \
--nqueries 128 \
--base_lr 3e-4 \
--matcher_giou_cost 3 \
--matcher_cls_cost 1 \
--matcher_center_cost 5 \
--matcher_objectness_cost 5 \
--loss_giou_weight 0 \
--loss_no_object_weight 0.1 \
--loss_sg_para_weight 2.0 \
--loss_size_weight 1 \
--loss_angle_reg_weight 1 \
--loss_angle_cls_weight 0.2 \
--save_separate_checkpoint_every_epoch 25 \
--dataset_num_workers 16 \
--batchsize_per_gpu 16 \
--eval_every_epoch 30 \
--checkpoint_dir ../outputs/sg_0407_2024_v4_room_non_hierarchy
