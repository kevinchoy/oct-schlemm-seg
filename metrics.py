import csv
import glob
import os
from pathlib import Path

"""Code for "K. C. Choy, G. Li, W. D. Stamer, S. Farsiu, Open-source deep learning-based automatic segmentation of
mouse Schlemm’s canal in optical coherence tomography images. Experimental Eye Research, 108844 (2021)."
Link: https://www.sciencedirect.com/science/article/pii/S0014483521004103
DOI: 10.1016/j.exer.2021.108844
The data and software here are only for research purposes. For licensing, please contact Duke University's Office of
Licensing & Ventures (OLV). Please cite our corresponding paper if you use this material in any form. You may not
redistribute our material without our written permission. """

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score

import torchvision.transforms.functional as TF
from utils.save_utils import calculate_stats
import numpy as np
import cv2
import math
from surface_distance import compute_surface_distances, compute_average_surface_distance, compute_robust_hausdorff, \
    compute_surface_dice_at_tolerance, compute_dice_coefficient


class MetricLogger:
    def __init__(self, metric, fp):
        self.metric = metric
        self.fp = fp
        self.scores = []
        self.scores_per_fold = []

    def __call__(self, input, target, fold):
        score = self.metric(target.cpu().view(-1), input.cpu().view(-1))
        self.scores.append(score)
        try:
            self.scores_per_fold[fold].append(score)
        except KeyError:
            self.scores_per_fold[fold] = []
            self.scores_per_fold[fold].append(score)


def PrecisionScore(input, target):
    return precision_score(target.cpu().view(-1), input.cpu().view(-1))


def RecallScore(input, target):
    return recall_score(target.cpu().view(-1), input.cpu().view(-1))


def sens_spec(input, target):
    # https://github.com/pytorch/pytorch/issues/1249
    smooth = 0.01
    iflat = input.view(-1)
    tflat = target.view(-1)

    tp = (iflat * tflat).sum()
    fn = tflat.sum() - tp
    fp = iflat.sum() - tp
    dice = (2. * tp + smooth) / (iflat.sum() + tflat.sum() + smooth)
    dice2 = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    sensitivity = tp / tflat.sum()

    ibackground = (1 - input).view(-1)
    tbackground = (1 - target).view(-1)
    tn = (ibackground * tbackground).sum()
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def spatial_metrics(mask_gt, mask_pred, vertical=1.170/1024, horizontal=0.001):
    # We make sure we do not have active pixels on the border of the image,
    # because this will add additional 2D surfaces on the border of the image
    # because the image is padded with background.
    if mask_gt.dtype is not np.bool:
        mask_gt = mask_gt.astype(np.bool)

    if mask_pred.dtype is not np.bool:
        mask_pred = mask_pred.astype(np.bool)

    surface_distances = compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(vertical, horizontal))
    average_surface_distance = compute_average_surface_distance(surface_distances)
    hausdorff = compute_robust_hausdorff(surface_distances, 75)
    surface_dice = compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=0.005)
    # dice = compute_dice_coefficient(mask_gt, mask_pred)
    # print(average_surface_distance, hausdorff, surface_dice, dice)
    return average_surface_distance, hausdorff, surface_dice


def get_gt_masks(root_dir):
    masks = sorted(glob.glob(os.path.join(root_dir, '*', 'mask', '*')))
    return masks

def get_pred_masks(root_dir, n_folds=6):
    masks = sorted(glob.glob(os.path.join(root_dir, 'segmentation/*/best_val_loss_last_20/seg_output/*PredictedMask*')))
    d = {}
    for i in range(1, n_folds+1):
        d[i] = []
    for fp in masks:
        fold = int(Path(fp).parts[-4])
        d[fold].append(fp)

    masks_ordered = []
    for i in reversed(range(1, n_folds+1)):
        masks_ordered = masks_ordered + d[i]
    return masks_ordered


def compute_metrics(save_dir, pred_masks, gt_masks, csv_name='metrics'):
    if isinstance(gt_masks, str):
        gt_masks = get_gt_masks(gt_masks)

    dice_list = []
    area_pred_list = []
    recall_list = []
    precision_list = []
    average_surface_distance_g2p_list = []
    average_surface_distance_p2g_list = []
    surface_dice_list = []
    hausdorff_list = []
    fnames = []
    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        fp2 = Path(pred_mask).parts[-1]
        fnames.append(fp2)
        # load images
        gt = Image.open(gt_mask)
        if gt.mode != 'L':
            gt = gt.convert('L')
        pred = Image.open(pred_mask)
        if pred.mode != 'L':
            pred = pred.convert('L')
        gt = TF.to_tensor(gt).to(torch.uint8)
        pred = TF.to_tensor(pred).to(torch.uint8)

# difference here
        # pixel overlap metrics
        dice = dice_coeff(pred, gt).item()
        area_pred = torch.sum(pred).item()
        precision = PrecisionScore(pred, gt)
        recall = RecallScore(pred, gt)

        # spatial metrics
        average_surface_distance, hausdorff, surface_dice = spatial_metrics(gt.squeeze().numpy(), pred.squeeze().numpy())

        # print('precision: {}, recall: {}'.format(precision, recall))
        dice_list.append(dice)
        area_pred_list.append(area_pred)
        recall_list.append(recall)
        precision_list.append(precision)

        hausdorff_list.append(hausdorff)
        average_surface_distance_g2p_list.append(average_surface_distance[0])
        average_surface_distance_p2g_list.append(average_surface_distance[1])
        surface_dice_list.append(surface_dice)

    dice_ave, dice_med, dice_std = calculate_stats(dice_list)
    precision_ave, precision_med, precision_std = calculate_stats(precision_list)
    recall_ave, recall_med, recall_std = calculate_stats(recall_list)

    hausdorff_ave, hausdorff_med, hausdorff_std = calculate_stats(hausdorff_list)
    surface_dice_ave, surface_dice_med, surface_dice_std = calculate_stats(surface_dice_list)
    average_surface_distance_g2p_ave, average_surface_distance_g2p_med, average_surface_distance_g2p_std = calculate_stats(average_surface_distance_g2p_list)
    average_surface_distance_p2g_ave, average_surface_distance_p2g_med, average_surface_distance_p2g_std = calculate_stats(average_surface_distance_p2g_list)

    print(dice_ave, dice_med, dice_std)
    print(precision_ave, precision_med, precision_std)
    print(recall_ave, recall_med, recall_std)

    fp1 = os.path.join(save_dir, csv_name + '_pixel.csv')
    with open(fp1, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['fname', 'dsc', 'precision', 'recall'])
        for fn, dsc, precision, recall in zip(fnames, dice_list, precision_list, recall_list):
            writer.writerow([fn, dsc, precision, recall])
    fp1 = os.path.join(save_dir, csv_name + '_pixel_summary.csv')
    with open(fp1, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'mean', 'median', 'stddev'])
        writer.writerow(['dice', dice_ave, dice_med, dice_std])
        writer.writerow(['precision', precision_ave, precision_med, precision_std])
        writer.writerow(['recall', recall_ave, recall_med, recall_std])

    fp2 = os.path.join(save_dir, csv_name + '_spatial.csv')
    with open(fp2, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['fname', 'hausdorff', 'surface_dice', 'average_surface_distance_g2p', 'average_surface_distance_p2g'])
        for fp2, hausdorff, surface_dice, average_surface_distance_g2p, average_surface_distance_p2g in zip(fnames, hausdorff_list, surface_dice_list, average_surface_distance_g2p_list, average_surface_distance_p2g_list):
            writer.writerow([fp2, hausdorff, surface_dice, average_surface_distance_g2p, average_surface_distance_p2g ])
    fp2 = os.path.join(save_dir, csv_name + '_spatial_summary.csv')
    with open(fp2, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'mean', 'median', 'stddev'])
        writer.writerow(['hausdorff', hausdorff_ave, hausdorff_med, hausdorff_std])
        writer.writerow(['surface dice', surface_dice_ave, surface_dice_med, surface_dice_std])
        writer.writerow(['average surface distance g2p', average_surface_distance_g2p_ave, average_surface_distance_g2p_med, average_surface_distance_g2p_std])
        writer.writerow(['average surface distance p2g', average_surface_distance_p2g_ave, average_surface_distance_p2g_med, average_surface_distance_p2g_std])


def run_metrics(experiment_dir, gt_masks=None, csv_name='metrics'):
    pred_masks = get_pred_masks(experiment_dir)
    compute_metrics(experiment_dir, pred_masks, gt_masks, csv_name=csv_name)


def dice_coeff(input, target):
    # https://github.com/pytorch/pytorch/issues/1249
    smooth = 0.01
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    result = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return result