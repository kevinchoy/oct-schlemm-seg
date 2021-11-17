import csv
import glob
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes

from skimage.morphology import erosion, dilation
from skimage.morphology import disk

from skimage import measure

from metrics import compute_metrics


def get_largest_conncomp(segmentation):
    labels = measure.label(segmentation, background=0)
    if np.max(labels) == 0:
        return segmentation
    else:
        largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
        return largestCC


def get_gt_masks(root_dir):
    masks = sorted(glob.glob(os.path.join(root_dir, '*', 'mask', '*')))
    return masks


def get_pred_masks(root_dir, dirname='seg_output'):
    masks = sorted(glob.glob(os.path.join(root_dir, 'segmentation/*/best_val_loss_last_20/'+dirname+'/*PredictedMask*')))
    d = {}
    for i in range(1, 8):
        d[i] = []
    for fp in masks:
        fold = int(Path(fp).parts[-4])
        d[fold].append(fp)

    masks_ordered = []
    for i in reversed(range(1, 8)):
        masks_ordered = masks_ordered + d[i]
    return masks_ordered


def post_processing(src_dir, selem_size=8):
    selem = disk(selem_size)
    pred_masks = get_pred_masks(src_dir)
    dest_dir = os.path.join(*Path(pred_masks[0]).parts[:-5], 'post')
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    fnames = []
    for pred_mask in pred_masks:
        fn = Path(pred_mask).parts[-1]
        fnames.append(fn)
        # load images
        pred = Image.open(pred_mask)
        if pred.mode != '1':
            pred = pred.convert('1')
        im_0 = np.asarray(pred)
        # im_1 = opening(im_0, selem)
        # im_2 = binary_fill_holes(im_1)
        # out = get_largest_conncomp(im_2)

        im_1 = dilation(im_0, selem)
        im_2 = get_largest_conncomp(im_1)

        im_3 = erosion(im_2, selem)
        out = binary_fill_holes(im_3)

        dest_fp = os.path.join(dest_dir, fn)
        im = Image.fromarray(out)
        im.save(dest_fp)


def post_processing_on_one_image(im, selem_size=8):
    selem = disk(selem_size)
    im_1 = dilation(im, selem)
    im_2 = get_largest_conncomp(im_1)
    im_3 = erosion(im_2, selem)
    out = binary_fill_holes(im_3)
    return out


if __name__ == '__main__':
    src_dir = '/home/kevin/Projects/oct_sc_segmentation/experiments_5/7_fold_cv/unet_fused_att_multiscale_dilated_res/2021-01-25/21-19-18'

    # post_processing(src_dir)
    pred_masks = get_pred_masks(src_dir, dirname='post')
    post_masks = sorted(glob.glob(os.path.join(src_dir, 'post/*PredictedMask*')))
    gt_masks = '/home/kevin/Projects/oct_sc_segmentation/data_root/sc_datasets/sc_speckle5_361-440_pigmented_n34'
    compute_metrics(src_dir, post_masks, gt_masks=gt_masks, csv_name='metrics_post')

