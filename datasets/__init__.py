# Copyright (c) Facebook, Inc. and its affiliates.
# from .scannet import ScannetDetectionDataset, ScannetDatasetConfig
from .sunrgbd import SunrgbdDetectionDataset, SunrgbdDatasetConfig
from .sg import SGDetectionDataset, SGDatasetConfig

DATASET_FUNCTIONS = {
    # "scannet": [ScannetDetectionDataset, ScannetDatasetConfig],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
    "sg": [SGDetectionDataset, SGDatasetConfig],
}


def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()
    
    dataset_dict = {
        "train": dataset_builder(
            dataset_config, 
            split_set="train", 
            root_dir=args.dataset_root_dir, 
            # meta_data_dir=args.meta_data_dir,
            use_color=args.use_color,
            augment=False       # SG dataset in the camera coordinate currently not working with augmentation
        ),
        "test": dataset_builder(
            dataset_config, 
            split_set="val", 
            root_dir=args.dataset_root_dir, 
            use_color=args.use_color,
            augment=False
        ),
    }
    return dataset_dict, dataset_config
    