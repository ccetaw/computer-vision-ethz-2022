from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.optim as optim
import numpy as np
import random

from lib.core.function import train
from lib.core.function import validate
from lib.core.loss import CrossEntropy2D
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger

import lib.dataset as dataset
import lib.models as models

def parse_args():
    parser = argparse.ArgumentParser(description='Train image segmentation network')
    # training
    parser.add_argument('--out_dir',
                        help='directory to save outputs',
                        default='out',
                        type=str)
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=10,
                        type=int)
    parser.add_argument('--gpu',
                        action='store_true',
                        help='whether to use GPU or not')
    parser.add_argument('--num_workers',
                        help='num of dataloader workers',
                        default=1,
                        type=int)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Create text logger and tensorboard logger
    logger, tb_logger = create_logger(
        args.out_dir, phase='valid', create_tf_logs=True)

    logger.info(pprint.pformat(args))

    model = models.seg_net_lite.get_seg_net()

    if tb_logger:
        writer_dict = {
            'logger': tb_logger,
            'train_global_steps': 0,
            'valid_global_steps': 0,
            'vis_global_steps': 0,
        }
    else:
        writer_dict = None

    # define loss function (criterion)
    if args.gpu:
        model = model.cuda()
        criterion = CrossEntropy2D(ignore_index=255).cuda()
    else:
        criterion = CrossEntropy2D(ignore_index=255)

    # Load best model
    model_state_file = os.path.join(args.out_dir,
                                    'model_best.pth.tar')
    logger.info('=> loading model from {}'.format(model_state_file))
    state_dict = torch.load(model_state_file, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    val_dataset = dataset.mnist(is_train=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # evaluate on validation set
    perf_indicator = validate(val_loader, val_dataset, model,
                              criterion, args.out_dir, writer_dict, args)

    writer_dict['logger'].close()


if __name__ == '__main__':
    main()
