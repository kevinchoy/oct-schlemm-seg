#!/usr/bin/env python
# coding: utf-8
"""Code for "K. C. Choy, G. Li, W. D. Stamer, S. Farsiu, Open-source deep learning-based automatic segmentation of
mouse Schlemm’s canal in optical coherence tomography images. Experimental Eye Research, 108844 (2021)."
Link: https://www.sciencedirect.com/science/article/pii/S0014483521004103
DOI: 10.1016/j.exer.2021.108844
The data and software here are only for research purposes. For licensing, please contact Duke University's Office of
Licensing & Ventures (OLV). Please cite our corresponding paper if you use this material in any form. You may not
redistribute our material without our written permission. """

import glob
import os
from pathlib import Path

import numpy as np
import time
import csv
import datetime
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split, KFold

from data_augmentation import SelectKeys, RandomCrop, ToPILImage, RandomHorizontalFlip, RandomScale, RandomTranslate, \
    RandomRotate, ToTensor, Rescale, CenterCrop, Normalize, ScaleRange
import random
from postprocessing import post_processing, get_pred_masks
from utils import util
from utils.config_loader import load_optimizer, load_loss, load_model, load_dataset
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.plots import save_results
from utils.util import load_saved_model, undo_transforms
from utils.save_utils import calculate_performance, save_figure_single_input
from metrics import PrecisionScore, RecallScore, compute_metrics, dice_coeff
import math

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


def flatten(t):
    return [item for sublist in t for item in sublist]


def train(model, device, train_loader, criterion, optimizer, epoch, scheduler=None):
    model.train()
    train_loss = 0
    for batch_idx, sample in enumerate(train_loader):
        data = sample['inputs']  # [N, 2, H, W]
        mask = sample['mask']  # [N, H, W]
        data, mask = data.to(device).float(), mask.to(device).float()
        # plot_vis(mask, win=implot_out, opts={'caption': 'output'})

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        # output = torch.sigmoid(output)
        loss = criterion(output, mask.unsqueeze(1))
        # loss = criterion(output.squeeze(1), mask)
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
            # print(scheduler.get_lr())

        train_loss += loss.cpu().detach().item()
        if batch_idx % (len(train_loader) // 3) == 0:
            print('Train({})[{:.0f}%]: Loss: {:.4f}'.format(
                epoch, 100. * batch_idx / len(train_loader), train_loss / (batch_idx + 1)))
    return train_loss / (batch_idx + 1)


def validation(model, device, val_loader, criterion, epoch, writer=None):
    model.eval()
    val_loss = 0
    pred_batch = []
    gt_batch = []
    img_batch = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            data = sample['inputs']  # [N, 2, H, W]
            mask = sample['mask']  # [N, H, W]
            data, mask = data.to(device).float(), mask.to(device).float()
            output = model(data)
            # loss = criterion(output.squeeze(1), mask)
            loss = criterion(output, mask.unsqueeze(1))
            val_loss += loss.cpu().detach().item()

            pred_batch.append((output > 0.5).float())
            img_batch.append(data[:, 0].unsqueeze(0))
            gt_batch.append(mask.unsqueeze(0))

        if writer is not None and (epoch == 1 or epoch % 3 == 0):
            if epoch == 1:
                # gt_batch = [sample['mask'].unsqueeze(0) for sample in val_loader]
                # img_batch = [sample['inputs'][:, 0] for sample in val_loader]
                writer.add_images('val/inputs', torch.cat(img_batch, dim=1).permute(1, 0, 2, 3), global_step=epoch)
                writer.add_images('val/ground_truth', torch.cat(gt_batch, dim=1).permute(1, 0, 2, 3), global_step=epoch)
            writer.add_images('val/predicted_masks', torch.cat(pred_batch, dim=0), global_step=epoch)

        print('Validation({}): Loss: {:.4f}'.format(epoch, val_loss / (batch_idx + 1)))
        return val_loss / (batch_idx + 1)


def evaluate(model, device, test_loader, epoch, writer=None, save_dir=None, orig_shape=[512, 512]):
    model.eval()

    tot = 0
    tot_precision = 0
    tot_recall = 0
    pred_batch = []

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_dir, 'seg_output')).mkdir(parents=True, exist_ok=True)
        csv_fn = 'metrics.csv'
        with open(os.path.join(save_dir, csv_fn), 'w') as f:
            w = csv.writer(f)
            w.writerow(['name', 'dice', 'precision', 'recall'])

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data = sample['inputs']
            mask = sample['mask']
            data, mask = data.to(device).float(), mask.to(device).float()

            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0

            fname = sample['metadata']['fname'][0]
            pressure = sample['metadata']['pressure']
            pressure = pressure.float()

            output = model(data)
            output = torch.sigmoid(output)
            mask_pred = (output >= 0.5).float()

            pred_batch.append(mask_pred)
            dsc = dice_coeff(mask_pred, mask).item()
            tot += dsc

            precision = PrecisionScore(input=mask_pred, target=mask)
            tot_precision += precision
            recall = RecallScore(input=mask_pred, target=mask)
            tot_recall += recall

            if save_dir is not None:
                with open(os.path.join(save_dir, csv_fn), 'a') as f:
                    w = csv.writer(f)
                    w.writerow([fname, dsc, precision, recall])

                pressure = int(pressure) if not math.isnan(pressure) else int(0)

                mask_image = mask.squeeze(0).cpu().detach().numpy()
                probability_map = output.squeeze(1).squeeze(0).cpu().detach().numpy()

                fn = '{}-iop{}-GroundTruth.png'.format(fname, pressure)
                fn = os.path.join(save_dir, fn)
                save_image(mask, fn)
                fn = '{}-iop{}-PredictedMask-dsc{:0.3f}.png'.format(fname, pressure, dsc).replace('.', '_', 1)
                fn = os.path.join(save_dir, fn)
                save_image(mask_pred, fn)

                fn = '{}-iop{}-PredictedMask-dsc{:0.3f}.png'.format(fname, pressure, dsc).replace('.', '_', 1)
                fp = os.path.join(save_dir, 'seg_output', fn)
                mask_pred_orig_shape = undo_transforms(mask_pred.squeeze(), shape=orig_shape)
                save_image(mask_pred_orig_shape.cpu().squeeze(), fp=fp)

                # save figure
                fig_title = 'Mouse #{}, IOP={}, DSC={:0.3f}'.format(fname[:2], pressure, dsc)
                fig_fname = os.path.join(save_dir, 'fig-{}-iop{}-dsc{:0.3f}.png'.format(
                    fname, pressure, dsc).replace('.', '_', 1))
                save_figure_single_input(fig_fname, input=data, mask_image=mask_image,
                            probability_map=probability_map, mask_pred=mask_pred, label=configs.dataset.labels, title=fig_title)

        dice_mean = tot / (batch_idx + 1)
        precision_mean = tot_precision / (batch_idx + 1)
        recall_mean = tot_recall / (batch_idx + 1)

    if save_dir is not None:
        with open(os.path.join(save_dir, csv_fn), 'a') as f:
            w = csv.writer(f)
            w.writerows([['mean dice', dice_mean], ['precision', precision_mean], ['recall', recall_mean]])

    if writer is not None and (epoch == 1 or epoch % 2 == 0):
        if epoch == 1:
            gt_batch = [sample['mask'].unsqueeze(0) for sample in test_loader]
            img_batch = [sample['inputs'][:, 0].unsqueeze(0) for sample in test_loader]

            writer.add_images('test/inputs', torch.cat(img_batch, dim=1).permute(1, 0, 2, 3), global_step=epoch)
            writer.add_images('test/ground_truth', torch.cat(gt_batch, dim=1).permute(1, 0, 2, 3), global_step=epoch)
        writer.add_images('test/predicted_masks', torch.cat(pred_batch, dim=0), global_step=epoch)

    print('Test({}): Dice: {:.4f}'.format(epoch, dice_mean))

    return dice_mean, precision_mean, recall_mean


# @hydra.main(config_path="conf/config.yaml")
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    global configs
    configs = cfg
    print(OmegaConf.to_yaml(cfg))
    # Initialization
    args = cfg.args
    num_epochs = args.num_epochs
    inputs = args.input_features

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print('device:', device)
    if args.start_fold or args.load_checkpoint:
        experiment_dir = os.path.join(args.save_root, args.experiment)
    else:
        current_time = datetime.datetime.now()
        experiment = os.path.join(args.experiment, current_time.strftime('%Y-%m-%d'), current_time.strftime('%H-%M-%S'))
        experiment_dir = os.path.join(args.save_root, experiment)
    util.mkdir(experiment_dir)

    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(cfg.pretty())

    # === Define Image Transforms ===
    # input image is 460 x 381 (width x height)
    # axial resolution: 1170/1024 micron/pixel
    # lateral resolution: 500 micron / 500 pixel = 1 micron/pixel
    axial_res = 1170 / 1024  # microns/pixel
    lateral_res = 1

    if cfg.dataset.image_size:
        height_pixels, width_pixels = cfg.dataset.image_size
        inputs = cfg.dataset.inputs
    else:
        height_pixels = 381
        width_pixels = 460

    height_um = round(height_pixels * axial_res)  # 435
    width_um = round(width_pixels * lateral_res)  # 460

    width_crop = 512
    height_crop = 512
    scaling_factor = 512 / min(height_um, width_um)  # resize height (smaller dimension) to 512
    height_scaled = round(scaling_factor * height_um)  # height: 435 --> 512
    width_scaled = round(scaling_factor * width_um)  # width: 460 --> 541

    mean = cfg.dataset.mean
    std = cfg.dataset.std

    translate_lim = cfg.data_aug.random_translate
    scale_lim = cfg.data_aug.random_scale
    degrees = cfg.data_aug.random_rotate

    trfm = transforms.Compose([
        SelectKeys(keys=(*inputs, 'mask')),
        ScaleRange(keys=(*inputs,)),
        ToPILImage(),  # following transforms operate on PIL image
        Rescale((height_um, width_um)),  # make isomorphic
        RandomHorizontalFlip(),
        RandomRotate(degrees),
        RandomTranslate(translate_lim),
        RandomScale(scale_lim),
        Rescale((height_scaled, width_scaled)),
        ToTensor(),
        RandomCrop((height_crop, width_crop)),
        Normalize(keys=(*inputs,), mean=mean, std=std),
    ])

    trfm_valid = transforms.Compose([
        SelectKeys(keys=(*inputs, 'mask')),
        ScaleRange(keys=(*inputs,)),
        ToPILImage(),
        Rescale((height_scaled, width_scaled)),  # isomorphic and scaled
        CenterCrop((height_crop, width_crop)),
        ToTensor(),
        Normalize(keys=(*inputs,), mean=mean, std=std),
    ])
    print(trfm)
    print(trfm_valid)

    # dataset
    dataset = load_dataset(cfg.dataset, transform=trfm)
    test_dataset = load_dataset(cfg.dataset, transform=trfm_valid)  # no data augmentation

    # 6-fold cross validation:
    n_folds = 6
    kf = KFold(n_splits=n_folds)
    for i, (train_valid_indices, test_indices) in enumerate(kf.split(dataset.mouse_id)):
        fold = i + 1

        if args.start_fold:
            if fold < int(args.start_fold):
                continue

        eyes_indices = list(reversed(dataset.subject_list))

        # Create the dataloader
        train_valid_idx = [s for i, s in enumerate(eyes_indices) if i in train_valid_indices]
        test_idx = [s for i, s in enumerate(eyes_indices) if i in test_indices]
        train_idx, valid_idx = train_test_split(train_valid_idx, test_size=1 / (n_folds - 1),
                                                random_state=args.seed + fold)

        train_idx = flatten(train_idx)
        valid_idx = flatten(valid_idx)
        test_idx = flatten(test_idx)

        trainset = torch.utils.data.Subset(dataset, train_idx)
        validset = torch.utils.data.Subset(test_dataset, valid_idx)
        testset = torch.utils.data.Subset(test_dataset, test_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                   pin_memory=False, num_workers=args.num_workers)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False,
                                                   pin_memory=False, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                                  pin_memory=False, num_workers=args.num_workers)

        print('\nFold #{}\n- - - - - - - - -'.format(fold))
        print('Number of samples in training set:', len(trainset))
        print('Number of samples in validation set:', len(validset))
        print('Number of samples in test set:', len(testset))
        # make directories
        save_dir = os.path.join(experiment_dir, 'folds', '{:02d}'.format(fold))
        saved_models_dir = os.path.join(save_dir, 'saved_models')
        best_models_dir = os.path.join(save_dir, 'best_models')
        segmentation_dir = os.path.join(experiment_dir, 'segmentation', '{:02d}'.format(fold))

        Path(saved_models_dir).mkdir(parents=True, exist_ok=True)
        Path(best_models_dir).mkdir(parents=True, exist_ok=True)
        Path(segmentation_dir).mkdir(parents=True, exist_ok=True)

        # Initialize model
        model = load_model(cfg.model)
        if args.load_checkpoint is not None:
            model_path = os.path.join(best_models_dir, 'best_model.tar')
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])  # Loading

        if args.data_parallel and torch.cuda.device_count() > 1:
            print('Using {} GPUs'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model = model.to(device)
        if fold == 1:
            num_params = sum(p.numel() for p in model.parameters())  # Total parameters
            num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            with open(os.path.join(experiment_dir, 'model.txt'), 'w') as f:
                f.write('num_params: {}\n'.format(num_params))
                f.write('num_trainable_params: {}\n\n'.format(num_trainable_params))
                f.write('{}'.format(model.__str__()))

        # Set loss function
        criterion = load_loss(cfg.loss)
        # Set optimizer
        optimizer = load_optimizer(model, cfg.optimizer)

        if args.use_scheduler:
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.5)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.0000001)
            # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=5)
            # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
            # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3)

        writer = SummaryWriter(
            os.path.join(experiment_dir, 'runs', '{:02d}'.format(fold))) if args.tensorboard else None

        val_loss_best = 1
        val_loss_last_20 = 1

        dice_threshold = 0
        with open(os.path.join(save_dir, 'results.csv'), 'w') as f:
            f.write('epoch,train_loss,val_loss,test_dice,test_precision,test_recall,elapsed_time,lr\n')

        #########################
        # --- Training Loop --- #
        #########################
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            epoch_save_dir = os.path.join(segmentation_dir, str(epoch))

            train_loss = train(model, device, train_loader, criterion, optimizer, epoch, scheduler=scheduler)
            val_loss = validation(model, device, valid_loader, criterion, epoch, writer=writer)
            test_dice, test_precision, test_recall = evaluate(model, device, test_loader, epoch, writer=writer)

            # print(torch.cuda.memory_summary())
            if args.tensorboard:
                writer.add_scalar('loss/train', train_loss, epoch)
                writer.add_scalar('loss/val', val_loss, epoch)
                writer.add_scalar('dice/test', test_dice, epoch)

            # Save checkpoints
            if val_loss < val_loss_best:
                print('Saving model with validation loss = {:.4f}...'.format(val_loss))
                fn = os.path.join(best_models_dir, 'model_best_val_loss.tar')
                if os.path.exists(fn):
                    os.remove(fn)
                util.save_checkpoint(model, optimizer, epoch, fn, train_loss, test_dice)
                val_loss_best = val_loss
                best_epoch = epoch

            if (epoch > num_epochs - 20) and val_loss < val_loss_last_20:
                print('Saving model with validation loss = {:.4f}...'.format(val_loss))
                fn = os.path.join(best_models_dir, 'model_best_val_loss_last20.tar')
                if os.path.exists(fn):
                    os.remove(fn)
                util.save_checkpoint(model, optimizer, epoch, fn, train_loss, test_dice)
                val_loss_last_20 = val_loss

            if epoch in [1, 3, 6, 10, 60, 100, 120, 150, 180, 200, num_epochs]:
                print('Saving checkpoint...')
                fn = os.path.join(saved_models_dir, 'ckpt_state_epoch_{}.tar'.format(epoch))
                util.save_checkpoint(model, optimizer, epoch, fn, train_loss, test_dice)
                evaluate(model, device, test_loader, epoch, save_dir=epoch_save_dir, orig_shape=cfg.dataset.image_size)

            if args.use_scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                elif isinstance(scheduler, MultiStepLR):
                    scheduler.step()
                print('Optimizer Learning rate: {0:.5f}'.format(optimizer.param_groups[0]['lr']))

            elapsed_time = time.time() - start_time
            print('elapsed time: {}'.format(elapsed_time))

            with open(os.path.join(save_dir, 'results.csv'), 'a') as f:
                f.write("{},{},{},{},{},{},{},{}\n".format(
                    epoch, train_loss, val_loss,
                    test_dice, test_precision, test_recall,
                    elapsed_time, optimizer.param_groups[0]['lr']))

        # save best model
        with open(os.path.join(best_models_dir, 'dice_scores.csv'), 'w') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'dice'])
            w.writerow([best_epoch, dice_threshold])

        checkpoint = torch.load(os.path.join(best_models_dir, 'model_best_val_loss.tar'))
        model = load_saved_model(model=model, checkpoint=checkpoint['model_state_dict'],
                                 data_parallel=args.data_parallel, device=args.device)
        evaluate(model, device, test_loader, checkpoint['epoch'],
                 save_dir=os.path.join(segmentation_dir, 'best_val_loss'), orig_shape=cfg.dataset.image_size)

        checkpoint = torch.load(os.path.join(best_models_dir, 'model_best_val_loss_last20.tar'))
        model = load_saved_model(model=model, checkpoint=checkpoint['model_state_dict'],
                                 data_parallel=args.data_parallel, device=args.device)
        evaluate(model, device, test_loader, checkpoint['epoch'],
                 save_dir=os.path.join(segmentation_dir, 'best_val_loss_last_20'), orig_shape=cfg.dataset.image_size)

        if writer:
            writer.close()  # close tensorboard

    save_results(experiment_dir)
    calculate_performance(experiment_dir)

    compute_metrics(experiment_dir, pred_masks=get_pred_masks(experiment_dir), gt_masks=cfg.dataset.params.root_dir,
                    csv_name='metrics')

    post_processing(experiment_dir)
    post_masks = sorted(glob.glob(os.path.join(experiment_dir, 'post/*PredictedMask*')))

    compute_metrics(experiment_dir, post_masks, gt_masks=cfg.dataset.params.root_dir, csv_name='metrics_post')


if __name__ == '__main__':
    main()
