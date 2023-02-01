from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time

import torch
import numpy as np

def create_logger(out_dir, phase='train', create_tf_logs=True):
    """Create text logger and TensorBoard writer objects

    Args:
        out_dir (str): output directory for saving logs.
        phase (str): short description for log, will be appended to log filename.
        create_tf_logs (bool): whether to create TensorBoard writer or not
    Returns:
        logger: text logger
        writer: TensorBoard writer
    """
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    final_log_file = os.path.join(out_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    if create_tf_logs:
        try:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(os.path.join(out_dir, 'logs'))
        except:
            writer = None
    else:
        writer = None

    return logger, writer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    """Save model checkpoint

    Args:
        states: model states.
        is_best (bool): whether to save this model as best model so far.
        output_dir (str): output directory to save the checkpoint
        filename (str): checkpoint name
    """
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))
