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

def read_runs(fold_data):
    epochs = []
    # fn = os.path.join(data_dir, '01/results.csv')

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
    #             for val in max_dice_indices:
    #                 writer.writerow([val])
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


# plots for one fold
def plot_for_one_fold(data_dir, i):
    fn = os.path.join(data_dir, '{:02d}/results.csv'.format(i + 1))
    epochs = []
    loss = []
    dice = []
    with open(fn) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            epochs.append(float(row[0]))
            loss.append(float(row[1]))
            dice.append(float(row[2]))

    plt.figure(figsize=[20, 8])
    plt.subplot(121)
    plt.title('Loss')
    plt.plot(epochs, loss)
    plt.subplot(122)
    plt.title('DSC')
    plt.plot(epochs, dice)
    plt.ylim((0, 1))
    plt.tight_layout()
    print('max dice[{}]: {:.3f}'.format(i + 1, max(dice)))

    return epochs, loss, dice


def main(experiment, root_dir='/home/kevin/Projects/oct_sc_segmentation/experiments/'):
    plt.figure(figsize=[20, 8])
    data_dir = root_dir + experiment
    save_dir = os.path.join(os.path.expanduser('~'), 'Projects', 'oct_sc_segmentation', 'figures')
    fn = os.path.join(save_dir, 'dice_results.csv')
    fn_indices = os.path.join(save_dir, 'max_dice_indices.csv')
    ave_dices, med_dices, std_dices = show_results(data_dir, label='average', experiment=experiment, save_dir=save_dir,
                                                   fn=fn, fn_indices=fn_indices)

    # show_results(data_dir, 'average', save_dir, experiment, n_folds=1)


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


if __name__ == '__main__':
    # main(experiment='2020-07-02_unet_batchnorm')
    experiments = []
    # experiment = '/home/kevin/Projects/oct_sc_segmentation/experiments/2020-07-16_unet_average_image_only'
    # experiment = '/home/kevin/Projects/oct_sc_segmentation/experiments_2/old_unet_baseline/2020-07-28/13-09-06'
    experiment = '/home/kevin/Projects/oct_sc_segmentation/experiments_2/old_unet_mean_image_only/2020-07-29/15-15-47'
    experiments.append(experiment)

    for experiment in experiments:
        save_results(experiment)
        # load_results(experiment, verbose=True)
