import argparse
import importlib
import os
import sys
import time
import os

import numpy as np
import open3d as o3d
from utils.pc_util import rotx, roty, rotz
from datasets.SG import sg_data
# from datasets.SUNRGBD import sunrgbd_utils
from models.model_3detr import build_preencoder, build_encoder, build_decoder, Model3DETR

import torch
import torch.optim as optim

from utils.pc_util import random_sampling
from utils.ap_calculator import parse_predictions, parse_predictions_eval
import datasets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

colors = [
    [255, 87, 51],  # Orange: "table",
    [52, 152, 219],  # Blue: "chair",
    [46, 204, 113],  # Green: "couch",
    [155, 89, 182],  # Purple: "bookshelf",
    [241, 196, 15],  # Yellow: "window",
    [231, 76, 60],  # Red: "door",
    [26, 188, 156],  # Turquoise
    [192, 57, 43],  # Dark Red
    [230, 126, 34],  # Light Orange
    [0, 255, 255],  # Cyan
    [255, 165, 0],  # Orange [Coral]
    [255, 0, 255],  # Magenta
    [0, 128, 0],  # Dark Green
    [128, 0, 128],  # Purple [Indigo]
    [255, 255, 0],  # Yellow [Lemon]
    [128, 128, 128]  # Gray
]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sg', help='Dataset: sunrgbd, scannet or sg [default: sg]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
##### Model #####
parser.add_argument(
    "--model_name",
    default="3detr",
    type=str,
    help="Name of the model",
    choices=["3detr"],
)
### Encoder
parser.add_argument(
    "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
)
# Below options are only valid for vanilla encoder
parser.add_argument("--enc_nlayers", default=3, type=int)
parser.add_argument("--enc_dim", default=256, type=int)
parser.add_argument("--enc_ffn_dim", default=128, type=int)
parser.add_argument("--enc_dropout", default=0.1, type=float)
parser.add_argument("--enc_nhead", default=4, type=int)
parser.add_argument("--enc_pos_embed", default=None, type=str)
parser.add_argument("--enc_activation", default="relu", type=str)

### Decoder
parser.add_argument("--dec_nlayers", default=8, type=int)
parser.add_argument("--dec_dim", default=256, type=int)
parser.add_argument("--dec_ffn_dim", default=256, type=int)
parser.add_argument("--dec_dropout", default=0.1, type=float)
parser.add_argument("--dec_nhead", default=4, type=int)

### MLP heads for predicting bounding boxes
parser.add_argument("--mlp_dropout", default=0.3, type=float)
parser.add_argument(
    "--nsemcls",
    default=-1,
    type=int,
    help="Number of semantic object classes. Can be inferred from dataset",
)

### Other model params
parser.add_argument("--preenc_npoints", default=2048, type=int)
parser.add_argument(
    "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
)
parser.add_argument("--nqueries", default=256, type=int)
parser.add_argument("--use_color", default=False, action="store_true")

args = parser.parse_args()


def demo_viz(pred, pc_file_path, bb3d_gt_file_path=None):
    point_cloud = []
    with open(pc_file_path, 'r') as f:
        for line in f.readlines():
            point = line.split(' ')
            x = float(point[0])
            y = float(point[1])
            z = float(point[2])
            point_as_array = [x, y, z]
            point_cloud.append(point_as_array)
    point_cloud = np.asarray(point_cloud)

    if bb3d_gt_file_path is not None:
        bbox = []
        with open(bb3d_gt_file_path, 'r') as f:
            for line in f.readlines():
                point = line.split(' ')
                cls = float(point[0])
                cx = float(point[1])
                cy = float(point[2])
                cz = float(point[3])
                l = float(point[4])
                w = float(point[5])
                h = float(point[6])
                roll = float(point[7])
                pitch = float(point[8])
                yaw = float(point[9])
                point_as_array = [cls, cx, cy, cz, l, w, h, roll, pitch, yaw]
                bbox.append(point_as_array)
        bbox = np.asarray(bbox)

    # Load data
    data = []
    # load point cloud
    point_cloud = np.vstack((point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    # pcd = pcd.uniform_down_sample(5)
    # pcd.paint_uniform_color([0.7, 0.7, 0.7])

    # load ground truth
    if bb3d_gt_file_path is not None:
        for i in range(bbox.shape[0]):
            cls = int(bbox[i][0])
            l = bbox[i][4]
            w = bbox[i][5]
            h = bbox[i][6]
            roll = bbox[i][7]
            pitch = bbox[i][8]
            yaw = bbox[i][9]
            cx = bbox[i][1]
            cy = bbox[i][2]
            cz = bbox[i][3]
            center = np.array([cx, cy, cz])
            size = np.array([2 * l, 2 * w, 2 * h])  # ground truth [l, w, h] is only half of the size
            xform = np.matmul(rotz(yaw), roty(pitch))
            xform = np.matmul(xform, rotx(roll))
            sample_3d = o3d.geometry.OrientedBoundingBox(center, xform, size)
            sample_3d.color = [1, 0, 0]
            data.append(sample_3d)
            # if cls > 5:  # only paint components
            #     pcd = paint_the_object(pcd, center, size, colors[cls])

    # # load prediction
    for j in range(len(pred[0])):
        class_p = int(pred[0][j][0])
        # Flip X-right, Y-forward, Z-up (SUNRGBD coordinate) back to X-right, Y-up, Z-forward (shape grammar coordinate)
        roll_p = pred[0][j][1]
        pitch_p = -pred[0][j][3]
        yaw_p = -pred[0][j][2]
        center_p = pred[0][j][5]
        center_p[[0, 2, 1]] = center_p[[0, 1, 2]]
        size_p = pred[0][j][6]
        size_p[[0, 2, 1]] = size_p[[0, 1, 2]]
        xform_p = np.matmul(rotz(yaw_p), roty(pitch_p))
        xform_p = np.matmul(xform_p, rotx(roll_p))
        sample_3d_p = o3d.geometry.OrientedBoundingBox(center_p, xform_p, size_p)
        sample_3d_p.color = [0, 1, 0]
        data.append(sample_3d_p)
        # if class_p > -1:
        #     pcd = paint_the_object(pcd, center_p, size_p, colors[class_p])

    # draw the point cloud and the GT and predicted bounding boxes
    data.append(pcd)
    o3d.visualization.draw_geometries(data)


def paint_the_object(pcd, center_p, size_p, color):
    points = np.asarray(pcd.points)
    object_point_indices = []
    # Get min and max coordinates of the bounding box
    min_coords = center_p - 0.5 * size_p  # this size here is only the half of the size of the bounding box
    max_coords = center_p + 0.5 * size_p
    # Find points that lie within the bounding box
    indices_within_bbox = np.all((min_coords <= points) & (points <= max_coords), axis=1)
    object_point_indices.append(np.where(indices_within_bbox)[0])
    # Create color array for the point cloud
    point_colors = np.asarray(pcd.colors)
    point_colors[object_point_indices] = np.asarray(color) / 255
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    return pcd


if __name__ == '__main__':

    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files')
    dataset_config = datasets.DATASET_FUNCTIONS[args.dataset][1]()
    # if args.dataset == 'sunrgbd':
    #     sys.path.append(os.path.join(ROOT_DIR, 'SUNRGBD'))
    #     checkpoint_path = os.path.join(ROOT_DIR, 'pretrained_votenet_on_sunrgbd.tar')
    #     pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.txt')
    #     pointcloud = sunrgbd_utils.load_depth_points_mat(pc_path)
    if args.dataset == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'SCANNET'))
        checkpoint_path = os.path.join(ROOT_DIR, 'pretrained_votenet_on_scannet.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_scannet.txt')
        pointcloud = np.loadtxt(pc_path)
    elif args.dataset == 'sg':
        sys.path.append(os.path.join(ROOT_DIR, 'SG'))
        checkpoint_path = os.path.join(ROOT_DIR, 'outputs/sg_0407_2024_v4_room_non_hierarchy/checkpoint_best.pth')
        pc_path = os.path.join(demo_dir, '9/pointcloud.txt')
        bb3d_gt_path = os.path.join(demo_dir, '9/bb3d.txt')
        pointcloud = sg_data.load_points(pc_path)
    else:
        print('Unkown dataset.')
        exit(-1)

    eval_config_dict = {'remove_empty_box': False, 'use_3d_nms': True, 'nms_iou': 0.25,
                        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
                        'use_cls_confidence_only': False, 'conf_thresh': 0.5, 'no_nms': False, 'dataset_config': dataset_config}

    # Init the model and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    net = Model3DETR(
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
    )
    net = net.to(device)
    print('Constructed model.')

    # Load checkpoint
    optimizer = optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)" % (checkpoint_path, epoch))

    # Load and preprocess input point cloud
    net.eval()  # set model to eval mode (for bn and dp)
    pc = random_sampling(pointcloud, 20000)
    pc_dims_min = pc.min(axis=0)
    pc_dims_max = pc.max(axis=0)
    pc = np.expand_dims(pc.astype(np.float32), 0)  # (1,20000,3)
    pc_dims_min = np.expand_dims(pc_dims_min.astype(np.float32), 0)  # (1,20000,3)
    pc_dims_max = np.expand_dims(pc_dims_max.astype(np.float32), 0)  # (1,20000,3)
    print('Loaded point cloud data: %s' % pc_path)

    # Model inference
    inputs = {'point_clouds': torch.from_numpy(pc).to(device),
              "point_cloud_dims_min": torch.from_numpy(pc_dims_min).to(device),
              "point_cloud_dims_max": torch.from_numpy(pc_dims_max).to(device)
              }
    tic = time.time()
    with torch.no_grad():
        outputs = net(inputs)
    toc = time.time()
    print('Inference time: %f' % (toc - tic))
    outputs = outputs["outputs"]
    point_clouds = inputs['point_clouds']
    predicted_box_corners = outputs["box_corners"]
    roll_continuous_angle = outputs["roll_angle_continuous"]
    pitch_continuous_angle = outputs["pitch_angle_continuous"]
    yaw_continuous_angle = outputs["yaw_angle_continuous"]
    sem_cls_probs = outputs["sem_cls_prob"]
    objectness_probs = outputs["objectness_prob"]
    center_unnormalized = outputs["center_unnormalized"]
    size_unnormalized = outputs["size_unnormalized"]
    sg_para = outputs["sg_para"]
    pred_map_cls = parse_predictions_eval(predicted_box_corners, roll_continuous_angle, pitch_continuous_angle, yaw_continuous_angle,
                                          sem_cls_probs, objectness_probs, point_clouds, center_unnormalized, size_unnormalized, sg_para,
                                          eval_config_dict)
    # pred_map_cls: cls, roll, pitch, yaw, corners, center, size, sg_para, sem_cls_prob
    f = open(os.path.join(demo_dir, 'pred_results.txt'), 'w')
    for k in range(len(pred_map_cls[0])):
        classID = pred_map_cls[0][k][0]
        center = pred_map_cls[0][k][5]
        size = pred_map_cls[0][k][6]
        EulerAngles = pred_map_cls[0][k][1:4]
        sgParameters = pred_map_cls[0][k][7]
        f.write("%d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %d\n" % (
            classID, center[0], center[1], center[2], size[0], size[1], size[2], EulerAngles[0], EulerAngles[1], EulerAngles[2],
            sgParameters[0], sgParameters[1], sgParameters[2], sgParameters[3], sgParameters[4], sgParameters[5]))
        print("Class ID: " + str(classID))
        print("Euler angles: " + str(EulerAngles))
        print("Center: " + str(center))
        print("SG parameters: " + str(sgParameters) + "\n")
    f.close()
    print('Finished detection. %d objects detected.' % (len(pred_map_cls[0])))
    demo_viz(pred_map_cls, pc_path, bb3d_gt_path)
    # dump_dir = os.path.join(demo_dir, '%s_results' % (args.dataset))
    # if not os.path.exists(dump_dir): os.mkdir(dump_dir)
    # MODEL.dump_results(end_points, dump_dir, dataconfig, True)
    # print('Dumped detection results to folder %s' % (dump_dir))
