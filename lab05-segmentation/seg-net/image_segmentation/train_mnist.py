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

# For reproducibility, fixing random seeds is usually a good practice.
seed = 37
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

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
    parser.add_argument('--eval_interval',
                        help='evaluation interval',
                        default=1,
                        type=int)
    parser.add_argument('--gpu',
                        action='store_true',
                        help='whether to use GPU or not')
    parser.add_argument('--num_workers',
                        help='num of dataloader workers',
                        default=4,
                        type=int)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Create text logger and tensorboard logger
    logger, tb_logger = create_logger(
        args.out_dir, phase='train', create_tf_logs=True)

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

    # Define loss function (criterion) and optimizer
    if args.gpu:
        model = model.cuda()
        criterion = CrossEntropy2D(ignore_index=255).cuda()
    else:
        criterion = CrossEntropy2D(ignore_index=255)

    optimizer = optim.Adam(model.parameters())

    # Create training and validation datasets
    train_dataset = dataset.mnist(is_train=True)
    val_dataset = dataset.mnist(is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    best_perf = 0.0 # best performance so far (mean IoU)
    best_model = False
    train_epochs = 20   # 20 epochs should be enough, if your implementation is right
    for epoch in range(train_epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,
              args.out_dir, writer_dict, args)

        if (epoch + 1) % args.eval_interval == 0:
            # evaluate on validation set
            perf_indicator = validate(val_loader, val_dataset, model,
                                      criterion, args.out_dir, writer_dict, args)

            # update best performance
            if perf_indicator > best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False
        else:
            perf_indicator = -1
            best_model = False

        # update best model so far
        logger.info('=> saving checkpoint to {}'.format(args.out_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'last_epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }, best_model, args.out_dir)


    final_model_state_file = os.path.join(args.out_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['logger'].close()


if __name__ == '__main__':
    main()
