"""Code for "K. C. Choy, G. Li, W. D. Stamer, S. Farsiu, Open-source deep learning-based automatic segmentation of
mouse Schlemm’s canal in optical coherence tomography images. Experimental Eye Research, 108844 (2021)."
Link: https://www.sciencedirect.com/science/article/pii/S0014483521004103
DOI: 10.1016/j.exer.2021.108844
The data and software here are only for research purposes. For licensing, please contact Duke University's Office of
Licensing & Ventures (OLV). Please cite our corresponding paper if you use this material in any form. You may not
redistribute our material without our written permission. """

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import pandas as pd
from PIL import Image
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import io
import pathlib
import torch
import math

from torchvision.utils import make_grid


def read_runs(fold_data):
    epochs = []
    with open(fold_data[0]) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            epochs.append(float(row[0]))

    n_epochs = len(epochs)
    n_folds = len(fold_data)

    loss_all = np.zeros((n_folds, n_epochs))
    dice_all = np.zeros((n_folds, n_epochs))

    for i in range(n_folds):
        fn = fold_data[i]
        # fn = os.path.join(data_dir, '{:02d}/results.csv'.format(fold))
        loss = []
        dice = []
        with open(fn) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)  # skip headers
            for row in reader:
                # epochs.append(float(row[0]))
                loss.append(float(row[1]))
                dice.append(float(row[2]))
        loss = np.array(loss)
        loss_all[i, :] = loss

        dice = np.array(dice)
        dice_all[i, :] = dice
    return epochs, loss_all, dice_all


def make_plots(epochs, ave_dices, med_dices, label=None):
    plt.figure(figsize=[20, 8])
    plt.subplot(1, 2, 1)
    plot_dice(epochs, ave_dices, title='Average DSC')
    plt.subplot(1, 2, 2)
    plot_dice(epochs, med_dices, title='Median DSC')


def plot_dice(epochs, metric, title, fs=20, fs2=15):
    plt.plot(epochs, metric)
    plt.xlabel('Epochs', fontsize=fs)
    plt.ylabel('DSC', fontsize=fs)
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.axis([0, 200, 0, 1])
    plt.title(title, fontsize=fs)
    #     plt.legend(fontsize=fs2)
    plt.tight_layout()


def calculate_stats(x):
    ave_x = np.mean(x, axis=0)
    med_x = np.median(x, axis=0)
    std_x = np.std(x, axis=0)
    return ave_x, med_x, std_x


def show_results(data_dir, label, save_dir=None, experiment=None, fn=None, fn_indices=None):
    subjects = list(range(1, 37))
    subjects.remove(33)
    subjects.remove(34)

    epochs, loss_all, dice_all = read_runs(data_dir, subjects)

    max_dice_indices = list(np.argmax(dice_all, axis=1))
    max_dice_values = list(np.max(dice_all, axis=1))

    ave_dices = np.mean(dice_all, axis=0)
    med_dices = np.median(dice_all, axis=0)
    std_dices = np.std(dice_all, axis=0)

    ave_max_dice = np.mean(max_dice_values)
    std_max_dice = np.std(max_dice_values)
    med_max_dice = np.median(max_dice_values)

    print(label)
    print('mean: {}, median: {}, stddev {}'.format(ave_max_dice, med_max_dice, std_max_dice))
    print()
    make_plots(epochs, ave_dices, med_dices, label)

    if save_dir:
        plt.savefig(os.path.join(save_dir, experiment + '.png'))
        plt.savefig(os.path.join(data_dir, experiment + '.png'))

    if fn_indices:
        with open(fn_indices, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([experiment] + max_dice_indices)
    if fn:
        with open(fn, 'a') as f:
            #             f.write("name,mean,median,stddev\n")
            f.write("{},{},{},{}\n".format(experiment, ave_max_dice, med_max_dice, std_max_dice))

    for i, idx in enumerate(max_dice_indices):
        print('{0:2d}, {1:03d}, {2:.3f}'.format(i + 1, idx, max_dice_values[i]))

    return ave_dices, med_dices, std_dices


def save_results(experiment_dir):
    fn = os.path.join(experiment_dir, 'dice_results.csv')
    fold_data = sorted(glob.glob(os.path.join(experiment_dir, 'folds', '**', 'results.csv')))
    subjects = [p.rsplit('/')[-2] for p in fold_data]
    epochs, loss_all, dice_all = read_runs(fold_data)

    max_dice_epochs = list(np.argmax(dice_all, axis=1) + 1)
    max_dice_values = list(np.max(dice_all, axis=1))

    with open(fn, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['mouse', 'epoch', 'best dice'])
        for subject, epoch, dice in zip(subjects, max_dice_epochs, max_dice_values):
            writer.writerow([subject, epoch, dice])

    ave_max_dice, med_max_dice, std_max_dice = calculate_stats(max_dice_values)
    print('mean: {}, median: {}, stddev {}'.format(ave_max_dice, med_max_dice, std_max_dice))

    ave_dices = np.mean(dice_all, axis=0)
    med_dices = np.median(dice_all, axis=0)

    make_plots(epochs, ave_dices, med_dices)
    plt.savefig(os.path.join(experiment_dir, 'dice_plots.png'))

    fn2 = os.path.join(experiment_dir, 'dice_overall.csv')
    with open(fn2, 'w') as f:
        writer = csv.writer(f)
        writer.writerows([['average', ave_max_dice], ['median', med_max_dice], ['stddev', std_max_dice]])


def main(experiment, root_dir='/home/kevin/Projects/oct_sc_segmentation/experiments/'):
    plt.figure(figsize=[20, 8])
    data_dir = root_dir + experiment
    save_dir = os.path.join(os.path.expanduser('~'), 'Projects', 'oct_sc_segmentation', 'figures')
    fn = os.path.join(save_dir, 'dice_results.csv')
    fn_indices = os.path.join(save_dir, 'max_dice_indices.csv')
    ave_dices, med_dices, std_dices = show_results(data_dir, label='average', experiment=experiment, save_dir=save_dir,
                                                   fn=fn, fn_indices=fn_indices)


def load_results(experiment_dir, verbose=False):
    fn = os.path.join(experiment_dir, 'dice_results.csv')
    if not os.path.exists(fn):
        save_results(experiment_dir=experiment_dir)
    dice = []
    with open(fn, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            dice.append(float(row[2]))

    if verbose:
        median_dsc = np.median(dice)
        mean_dsc = np.mean(dice)
        std_dsc = np.std(dice)
        print(Path(fn).parts[-2] + ':')
        print('mean: {:.3f}, median: {:.3f}, std: {:.3f}'.format(mean_dsc, median_dsc, std_dsc))
    return dice


def calculate_performance(experiment_dir):
    fold_data = sorted(glob.glob(os.path.join(experiment_dir, 'segmentation', '**', 'best_val_loss_last_20', 'metrics.csv')))
    dice_folds = []
    precision_folds = []
    recall_folds = []
    fnames = []
    for fn in reversed(fold_data):
        df = pd.read_csv(fn)
        fnames.append(df.name[:-3].to_numpy())
        dice_folds.append(df.dice[:-3].to_numpy())
        precision_folds.append(df.precision[:-3].to_numpy())
        recall_folds.append(df.recall[:-3].to_numpy())
    fnames = np.concatenate(fnames)
    dice_folds = np.concatenate(dice_folds)
    precision_folds = np.concatenate(precision_folds)
    recall_folds = np.concatenate(recall_folds)

    dice_ave, dice_med, dice_std = calculate_stats(dice_folds)
    precision_ave, precision_med, precision_std = calculate_stats(precision_folds)
    recall_ave, recall_med, recall_std = calculate_stats(recall_folds)

    print('Dice -- mean: {:.4f}, median: {:.4f}, stddev {:.4f}'.format(dice_ave, dice_med, dice_std))
    print('Precision -- mean: {:.4f}, median: {:.4f}, stddev {:.4f}'.format(precision_ave, precision_med, precision_std))
    print('Recall -- mean: {:.4f}, median: {:.4f}, stddev {:.4f}'.format(recall_ave, recall_med, recall_std))
    #
    fn2 = os.path.join(experiment_dir, 'performance.csv')
    with open(fn2, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'mean', 'median', 'stddev'])
        writer.writerow(['dice', dice_ave, dice_med, dice_std])
        writer.writerow(['precision', precision_ave, precision_med, precision_std])
        writer.writerow(['recall', recall_ave, recall_med, recall_std])

    fn3 = os.path.join(experiment_dir, 'all.csv')
    with open(fn3, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['fname', 'dsc', 'precision', 'recall'])
        for fn, dsc, precision, recall in zip(fnames, dice_folds, precision_folds, recall_folds):
            writer.writerow([fn, dsc, precision, recall])


def save_mask(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        fp: Union[Text, pathlib.Path, BinaryIO],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        format: Optional[str] = None,
) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im = im.convert('1')
    im.save(fp, format=format)


def save_figure(fname, mean_image, sv_image, mask_image, probability_map, mask_pred, title, show_fig=False):
    n_row = 1
    n_col = 5
    plt.figure(figsize=[12, 3])
    plt.subplot(n_row, n_col, 1)
    plt.imshow(mean_image, cmap='gray')
    plt.title('Mean Intensity')
    plt.axis('off')

    plt.subplot(n_row, n_col, 2)
    plt.imshow(sv_image, cmap='gray')
    plt.title('Speckle Variance')
    plt.axis('off')

    plt.subplot(n_row, n_col, 3)
    plt.imshow(mask_image, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(n_row, n_col, 4)
    plt.imshow(probability_map)
    plt.axis('off')
    plt.title('Probability Map')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim(0, 1)

    plt.subplot(n_row, n_col, 5)
    plt.imshow(mask_pred.cpu()[0, 0], cmap='gray')
    plt.title('Predicted Segmentation')
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()  # rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(top=0.85)
    plt.savefig(fname)
    if show_fig:
        plt.show()


def save_figure_single_input(fname, input, mask_image, probability_map, mask_pred, title, label, show_fig=False):
    n_inputs = input.shape[1]
    input_numpy = [input[:, i].squeeze(0).cpu().detach().numpy() for i in range(n_inputs)]
    n_row = 1
    n_col = n_inputs + 3
    plt.figure(figsize=[12, 3])

    for i in range(n_inputs):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(input_numpy[i], cmap='gray')
        plt.title(label[i])
        plt.axis('off')

    plt.subplot(n_row, n_col, n_inputs+1)
    plt.imshow(mask_image, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(n_row, n_col, n_inputs+2)
    plt.imshow(probability_map)
    plt.axis('off')
    plt.title('Probability Map')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim(0, 1)

    plt.subplot(n_row, n_col, n_inputs+3)
    plt.imshow(mask_pred.cpu().squeeze(), cmap='gray')
    plt.title('Predicted Segmentation')
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()  # rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(top=0.85)
    # plt.savefig(os.path.join(save_dir, 'dsc{:0.3f}-{}-iop{}.png'.format(
    #     dice, fname, pressure).replace('.', '_', 1)))
    plt.savefig(fname)
    if show_fig:
        plt.show()

if __name__ == '__main__':
    dirname = '/home/kevin/Projects/oct_sc_segmentation/experiments_3/7_fold_cv/unet_fused_att_dsv_multiscale_speckle5/2020-12-22/13-31-17'
    print(dirname)
    calculate_performance(dirname)
    print('- - - -')
