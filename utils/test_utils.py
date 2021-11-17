"""Code for "K. C. Choy, G. Li, W. D. Stamer, S. Farsiu, Open-source deep learning-based automatic segmentation of
mouse Schlemm’s canal in optical coherence tomography images. Experimental Eye Research, 108844 (2021)."
Link: https://www.sciencedirect.com/science/article/pii/S0014483521004103
DOI: 10.1016/j.exer.2021.108844
The data and software here are only for research purposes. For licensing, please contact Duke University's Office of
Licensing & Ventures (OLV). Please cite our corresponding paper if you use this material in any form. You may not
redistribute our material without our written permission. """

import csv
import numpy as np
import os
import matplotlib.pyplot as plt


def make_plots(experiment_dir, metric='DSC'):
    # experiment_dir = '/home/kevin/Projects/oct_sc_segmentation/experiments/'+experiment
    csv_fp = os.path.join(experiment_dir, 'analysis.csv')
    dices = []
    areas = []
    if metric == 'DSC':
        idx = 4
    elif metric == 'Sensitivity':
        idx = 5
    elif metric == 'Specificity':
        idx = 6

    with open(csv_fp) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            #         print(row)
            areas.append(float(row[3]))
            dices.append(float(row[idx]))

    plt.figure()
    plt.scatter(areas, dices)
    plt.ylabel(metric)
    plt.xlabel('Ground truth SC area (pixels)')
    plt.savefig(os.path.join(experiment_dir, 'area_vs_'+metric+'_2020-09-10.png'))
    plt.show()
    print()


def pressure_boxplot(experiment_dir, metric='DSC'):
    # experiment_dir = '/home/kevin/Projects/oct_sc_segmentation/experiments/'+experiment
    csv_fp = os.path.join(experiment_dir, 'analysis.csv')
    dices = []
    areas = []
    dice_by_iop = {}
    if metric == 'DSC':
        idx = 4
    elif metric == 'Sensitivity':
        idx = 5
    elif metric == 'Specificity':
        idx = 6

    with open(csv_fp) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            iop = float(row[2])
            areas.append(float(row[3]))
            dice = float(row[idx])
            dices.append(dice)

            if iop not in dice_by_iop.keys():
                dice_by_iop[iop] = [dice]
            else:
                dice_by_iop[iop].append(dice)
    plt.figure()

    data = []
    num_samples = []
    for k in sorted(dice_by_iop.keys()):
        data.append(dice_by_iop[k])
        num_samples.append(len(dice_by_iop[k]))
        print('{}: {}'.format(k, len(dice_by_iop[k])))
    num_boxes = len(data)

    fig, ax =plt.subplots()
    ax.boxplot(data)
    ax.set_xticklabels(sorted(dice_by_iop.keys()))


    pos = np.arange(num_boxes) + 1
#    upper_labels = [str(np.round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax.get_xticklabels()):
        k = tick % 2
        ax.text(pos[tick], .965, num_samples[tick],
                 transform=ax.get_xaxis_transform(),
                 horizontalalignment='center', size='x-small',
                 weight=weights[k])

    ax.set_ylim(-0.05, 1)
    ax.set_ylabel(metric)
    ax.set_xlabel('IOP (mmHg)')
    plt.savefig(os.path.join(experiment_dir, metric+'_iop_2020-09-10.png'))
    plt.show()


