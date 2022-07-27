import argparse
import importlib
import os
import sys
import time

import numpy as np

from models import build_model
from models.model_3detr import build_preencoder, build_encoder, build_decoder, Model3DETR
from optimizer import build_optimizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sunrgbd', help='Dataset: sunrgbd or scannet [default: sunrgbd]')
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

import torch
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from utils.pc_util import random_sampling
# from ap_helper import parse_predictions
import datasets


def preprocess_point_cloud(point_cloud):
    """ Prepare the numpy point cloud (N,3) for forward pass """
    point_cloud = point_cloud[:, 0:3]  # do not use color for now
    floor_height = np.percentile(point_cloud[:, 2], 0.99)
    height = point_cloud[:, 2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, args.num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0)  # (1,40000,4)
    return pc


if __name__ == '__main__':

    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files')
    DC = datasets.DATASET_FUNCTIONS[args.dataset][1]()
    if args.dataset == 'sunrgbd':
        sys.path.append(os.path.join(ROOT_DIR, 'SUNRGBD'))
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_sunrgbd.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.txt')
    elif args.dataset == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'SCANNET'))
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_scannet.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_scannet.txt')
    elif args.dataset == 'sg':
        sys.path.append(os.path.join(ROOT_DIR, 'SG'))
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_scannet.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_sg.txt')
    else:
        print('Unkown dataset.')
        exit(-1)

    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
                        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
                        'conf_thresh': 0.5, 'dataset_config': DC}

    # Init the model and optimzier
    # MODEL = importlib.import_module('votenet')  # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = MODEL.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
    #                     sampling='seed_fps', num_class=DC.num_class,
    #                     num_heading_bin=DC.num_heading_bin,
    #                     num_size_cluster=DC.num_size_cluster,
    #                     mean_size_arr=DC.mean_size_arr).to(device)

    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    net = Model3DETR(
        pre_encoder,
        encoder,
        decoder,
        DC,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
    )
    net = net.to(device)
    print('Constructed model.')

    # Load checkpoint
    optimizer = optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0)
    checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)" % (checkpoint_path, epoch))

    # Load and preprocess input point cloud
    net.eval()  # set model to eval mode (for bn and dp)
    pc = np.loadtxt(pc_path)
    pc = preprocess_point_cloud(pc)
    print('Loaded point cloud data: %s' % pc_path)

    # Model inference
    inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
    tic = time.time()
    with torch.no_grad():
        end_points = net(inputs)
    toc = time.time()
    print('Inference time: %f' % (toc - tic))
    # end_points['point_clouds'] = inputs['point_clouds']
    # pred_map_cls = parse_predictions(end_points, eval_config_dict)
    # print('Finished detection. %d object detected.' % (len(pred_map_cls[0])))
    #
    # dump_dir = os.path.join(demo_dir, '%s_results' % (FLAGS.dataset))
    # if not os.path.exists(dump_dir): os.mkdir(dump_dir)
    # MODEL.dump_results(end_points, dump_dir, DC, True)
    # print('Dumped detection results to folder %s' % (dump_dir))
