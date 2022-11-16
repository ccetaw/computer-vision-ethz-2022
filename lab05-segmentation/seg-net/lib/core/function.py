from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch

from lib.core.evaluate import calc_IoU
from lib.core.inference import get_final_preds
from lib.utils.vis import vis_segments

logger = logging.getLogger(__name__)

def train(train_loader, model, criterion, optimizer, epoch,
          output_dir, writer_dict, args):
    """Train the model for one epoch

    Args:
        train_loader (torch.utils.data.DataLoader): dataloader for training set.
        model (torch.nn.Module): image segmentation module.
        criterion (torch.nn.Module): loss function for image segmentation.
        optimizer (torch.optim.Optimizer): optimizer for model parameters.
        epoch (int): current training epoch.
        output_dir (str): directory to save logs.
        writer_dict (dict): dictionary containing tensorboard related objects.
        args: arguments from the main script.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if len(input.shape) > 4:
            # Note that in the MNIST dataloader, we return 3-dimentional tensor before we make the batch,
            # thus the batch of images returned from the dataloader would be 1 x B x 3 x H x W. Same applies
            # for semantic masks
            input = input.view(input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4])
            target = target.view(target.shape[0] * target.shape[1], target.shape[2], target.shape[3])

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)

        # compute loss
        target = target.to(output.device)
        loss = criterion(output, target)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.frequent == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['logger']
                global_steps = writer_dict['train_global_steps']

                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1


def validate(val_loader, val_dataset, model, criterion,
             output_dir, writer_dict, args):
    """Validate the model

    Args:
        val_loader (torch.utils.data.DataLoader): dataloader for validation set.
        val_dataset (): validation dataset, which contains mean and std for (un)normalizing images.
        model (torch.nn.Module): image segmentation module.
        criterion (torch.nn.Module): loss function for image segmentation.
        output_dir (str): directory to save logs.
        writer_dict (dict): dictionary containing tensorboard related objects.
        args: arguments from the main script.
    Returns:
        perf_indicator (float): performance indicator. In the case of image segmentation, we return
                                mean IoU over all validation images.
    """
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_preds = []
    all_gts = []

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if len(input.shape) > 4:
                # Note that in the MNIST dataloader, we return 3-dimentional tensor before we make the batch,
                # thus the batch of images returned from the dataloader would be 1 x B x 3 x H x W. Same applies
                # for semantic masks
                input = input.view(input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4])
                target = target.view(target.shape[0] * target.shape[1], target.shape[2], target.shape[3])

            # compute output
            output = model(input)

            # compute loss
            target = target.to(output.device)
            loss = criterion(output, target)

            # Upsample output, if it has different resolution to target
            output = torch.nn.functional.interpolate(
                output,
                size=(target.size(1), target.size(2)),
                mode="bilinear",
                align_corners=False)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            preds = get_final_preds(output.detach().cpu().numpy())

            all_preds.extend(preds)
            all_gts.extend(target.detach().cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.frequent == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)

                logger.info(msg)

                if writer_dict:
                    writer = writer_dict['logger']
                    global_steps = writer_dict['vis_global_steps']

                    # Pick a random image in the batch to visualize
                    idx = np.random.randint(0, num_images)

                    # Unnormalize the image to [0, 255] to visualize
                    input_image = input.detach().cpu().numpy()[idx]
                    input_image = input_image * val_dataset.std.squeeze(0) + val_dataset.mean.squeeze(0)
                    input_image[input_image > 1.0] = 1.0
                    input_image[input_image < 0.0] = 0.0

                    target_image = target.detach().cpu().numpy()[idx].astype(np.int64)
                    target_image = vis_segments(target_image, 11)

                    output = torch.nn.functional.softmax(output, dim=1)
                    labels = torch.argmax(output, dim=1, keepdim=False)

                    labels = labels.detach().cpu().numpy()[idx]
                    output_vis = vis_segments(labels, 11)


                    writer.add_image('input_image', input_image, global_steps,
                        dataformats='CHW')
                    writer.add_image('result_vis', output_vis, global_steps,
                        dataformats='HWC')
                    writer.add_image('gt_mask', target_image, global_steps,
                        dataformats='HWC')

                    writer_dict['vis_global_steps'] = global_steps + 1

        # Calculate IoU score for entire validation set
        all_preds = np.concatenate(all_preds, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)

        avg_iou_score = calc_IoU(all_preds, all_gts, 11)

        perf_indicator = avg_iou_score

        logger.info('Mean IoU score: {:.3f}'.format(avg_iou_score))

        if writer_dict:
            writer = writer_dict['logger']
            global_steps = writer_dict['valid_global_steps']

            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_iou_score', avg_iou_score, global_steps)

            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
