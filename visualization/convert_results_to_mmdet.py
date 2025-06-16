import utils
import logging
import argparse
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.models import build_model
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from models.utils import VERSION
import pickle
 
def main():
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--override', nargs='+', action=DictAction)
    parser.add_argument('--score_threshold', default=0.3)
    args = parser.parse_args()
 
    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)
 
    # use val-mini for visualization
    cfgs.data.val.ann_file = cfgs.data.test.ann_file
 
    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')
 
    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)
 
    # you need one GPU
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
 
    utils.init_logging(None, cfgs.debug)
    logging.info('Using GPU: %s' % torch.cuda.get_device_name(0))
    logging.info('Setting random seed: 0')
 
    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()
    model = MMDataParallel(model, [0])
 
    logging.info('Loading checkpoint from %s' % args.weights)
    checkpoint = load_checkpoint(
        model, args.weights, map_location='cuda', strict=True,
        logger=logging.Logger(__name__, logging.ERROR)
    )
 
    if 'version' in checkpoint:
        VERSION.name = checkpoint['version']
 
    mmdet_bboxes = []
    for i, data in enumerate(val_loader):
        model.eval()
       
        with torch.no_grad():
            results = model(return_loss=False, rescale=True, **data)
            results = results[0]['pts_bbox']
 
        mmdet_bboxes.append(results)
 
    with open('submission/pts_bbox/results.pkl', 'wb') as f:
        pickle.dump(mmdet_bboxes, f)
 
 
if __name__ == '__main__':
    main()
