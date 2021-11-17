import math
import random
import yaml
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset
from torchvision import transforms, utils
from pathlib import Path

from metrics import dice_coeff
from data_augmentation import Rescale, ToPILImage, ToTensor, SelectKeys, CenterCrop, ScaleRange, Normalize
from models.networks.parts import HookBasedFeatureExtractor

from utils.config_loader import load_model, load_dataset
from omegaconf import DictConfig
from torch.nn import functional as F
from skimage.transform import resize
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from solver import flatten

def get_feature_maps(input, net, layer_name, upscale):
    feature_extractor = HookBasedFeatureExtractor(net, layer_name, upscale)
    return feature_extractor.forward(input)


def plotNNFilterOverlay(input_im, units, figure_id, interp='bilinear',
                        colormap=cm.jet, colormap_lim=None, title='', alpha=0.8, save_path='fig.png'):
    plt.ion()
    filters = units.shape[2]
    fig = plt.figure(figure_id, figsize=(5, 5))
    fig.clf()

    for i in range(filters):
        plt.imshow(input_im[:,:,0], interpolation=interp, cmap='gray')
        plt.imshow(units[:,:,i], interpolation=interp, cmap=colormap, alpha=alpha)
        plt.axis('off')
        # plt.colorbar()
        # plt.title(title, fontsize='small')
        if colormap_lim:
            plt.clim(colormap_lim[0],colormap_lim[1])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def main(cfg: DictConfig, layer_name, save_root='visualize_attention'):
    # Initialization
    args = cfg.args
    inputs = args.input_features

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = 'cuda:0' # args.device if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print('device:', device)

    if cfg.dataset.mean and cfg.dataset.std:
        mean = cfg.dataset.mean
        std = cfg.dataset.std
    else:
        # sv3
        mean = [0.28693876, 0.31857252]
        std = [0.2341083, 0.13121088]

    # input image is 460 x 381 (width x height)
    # axial resolution: 1170/1024 micron/pixel
    # lateral resolution: 500 micron / 500 pixel = 1 micron/pixel
    width_crop = 512
    height_crop = 512

    axial_res = 1170 / 1024  # microns/pixel
    lateral_res = 1

    height_pixels = 381
    width_pixels = 460
    height_um = int(height_pixels * axial_res)  # 435
    width_um = int(width_pixels * lateral_res)  # 460
    scaling_factor = 512/min(height_um, width_um)  # resize height (smaller dimension) to 512
    # scaling_factor = height_crop / height_um  # resize height (smaller dimension) to 512
    height_scaled = int(scaling_factor * height_um)  # height: 435 --> 512
    width_scaled = int(scaling_factor * width_um)  # width: 460 --> 541
    trfm_valid = transforms.Compose([
        SelectKeys(keys=(*inputs, 'mask')),
        ScaleRange(keys=(*inputs,)),
        ToPILImage(),
        Rescale((height_scaled, width_scaled)),  # isomorphic and scaled
        CenterCrop((height_crop, width_crop)),
        ToTensor(),
        Normalize(keys=(*inputs,), mean=mean, std=std),
    ])
    print(trfm_valid)

    # dataset
    # dataset = load_dataset(cfg.dataset, transform=trfm_valid)
    test_dataset = load_dataset(cfg.dataset, transform=trfm_valid)  # no data augmentation

    # 7-fold cross validation:
    n_folds = 6
    kf = KFold(n_splits=n_folds)
    for i, (train_valid_indices, test_indices) in enumerate(kf.split(test_dataset.mouse_id)):
        fold = i + 1

        eyes_indices = list(reversed(test_dataset.subject_list))

        # Create the dataloader
        train_valid_idx = [s for i, s in enumerate(eyes_indices) if i in train_valid_indices]
        test_idx = [s for i, s in enumerate(eyes_indices) if i in test_indices]
        train_idx, valid_idx = train_test_split(train_valid_idx, test_size=1 / (n_folds - 1),
                                                random_state=args.seed + fold)

        # train_idx = flatten(train_idx)
        # valid_idx = flatten(valid_idx)
        test_idx = flatten(test_idx)

        # trainset = torch.utils.data.Subset(dataset, train_idx)
        # validset = torch.utils.data.Subset(test_dataset, valid_idx)
        testset = torch.utils.data.Subset(test_dataset, test_idx)

        # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
        #                                            pin_memory=False, num_workers=args.num_workers)
        # valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False,
        #                                            pin_memory=False, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                                  pin_memory=False, num_workers=args.num_workers)

        print('\nFold #{}\n- - - - - - - - -'.format(fold))
        # print('Number of samples in training set:', sorted(train_idx))
        # print('Number of samples in validation set:', sorted(valid_idx))
        # print('Number of samples in test set:', sorted(test_idx))
        # print('Number of samples in training set:', len(trainset))
        # print('Number of samples in validation set:', len(validset))
        print('Number of samples in test set:', len(testset))
        # continue

        # make directories
        save_dir = os.path.join(experiment_dir, 'folds', '{:02d}'.format(fold))
        best_models_dir = os.path.join(save_dir, 'best_models')
        vis_dir = os.path.join(experiment_dir, 'visualization_no_color_bar', '{:02d}'.format(fold))
        Path(vis_dir).mkdir(exist_ok=True, parents=True)
        # Initialize model
        model = load_model(cfg.model)
        model_path = os.path.join(best_models_dir, 'model_best_val_loss_last20.tar')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])  # Loading

        if args.data_parallel and torch.cuda.device_count() > 1:
            print('Using {} GPUs'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model = model.to(device)
        model.eval()

        attentions = []
        with torch.no_grad():
            for i, sample in enumerate(test_loader):

                # metadata
                fname = sample['metadata']['fname'][0]
                pressure = sample['metadata']['pressure'].item()
                pressure = int(pressure) if not math.isnan(pressure) else int(0)

                data = sample['inputs']
                data = data.to(device)
                mean_image = data[:, 0].cpu().detach().permute(1,2,0).numpy()
                sv_image = data[:, 1].cpu().detach().permute(1,2,0).numpy()
                mask = sample['mask']

                orig_input_img = data.permute(2, 3, 1, 0).cpu().numpy()

                if layer_name == 'attentionblock2':
                    layer_name = 'attentionblock2'
                    # Display the input image and Down_sample the input image
                    inp_fmap, out_fmap = get_feature_maps(input=data, net=model, layer_name=layer_name, upscale=False)

                    upsampled_attention = F.upsample(out_fmap[1], size=data.size()[2:], mode='bilinear').data.squeeze().cpu()  #.permute(1,2,3,0).cpu().numpy()

                    fp = os.path.join(vis_dir, 'attention_{}.png'.format(layer_name))
                    utils.save_image(upsampled_attention, fp=fp, normalize=True)
                    # attention = upsampled_attention.numpy()

                    # # Output of the attention block
                    # fmap_0 = out_fmap[0].squeeze().permute(1, 2, 0).cpu().numpy()
                    # fmap_size = fmap_0.shape

                if layer_name == 'fusedattention2':
                    layer_name = 'fusedattention2'
                    inp_fmap, out_fmap = get_feature_maps(input=data, net=model, layer_name=layer_name, upscale=False)
                    upsampled_attention = F.upsample(out_fmap[1], size=data.size()[2:], mode='bilinear').data.squeeze().cpu()  #.permute(1,2,3,0).cpu().numpy()
                    fp = os.path.join(vis_dir, 'attention_{}.png'.format(layer_name))
                    utils.save_image(upsampled_attention, fp=fp, normalize=True)

                # if layer_name == 'fusedattention2':
                #     inp_fmap, out_fmap = get_feature_maps(input=data, net=model, layer_name=layer_name,
                #                                           upscale=False)
                #     # fused_attn = torch.zeros(orig_input_img.shape[:2])
                #     for i, attn in enumerate(out_fmap[1:]):
                #         upsampled_attention = F.upsample(attn, size=data.size()[2:],
                #                                          mode='bilinear').data.squeeze().cpu()  # .numpy()
                #         # fused_attn.add_(upsampled_attention)
                #         fp = os.path.join(vis_dir, 'test_{}_{}.png'.format(layer_name, i))
                #         utils.save_image(upsampled_attention, filename=fp, normalize=True)

                # attention = upsampled_attention.numpy()


                attention = out_fmap[1].squeeze().cpu().numpy()
                attention = np.expand_dims(
                    resize(attention, (mean_image.shape[0], mean_image.shape[1]), mode='constant', preserve_range=True),
                    axis=2)

                output = model(data.float())
                output = torch.sigmoid(output)
                mask_pred = (output >= 0.5).float()
                dice = dice_coeff(mask_pred, mask.unsqueeze(0).to(device).float()).item()

                fp = os.path.join(vis_dir, fname+'-Average-AttentionOverlay.png')
                plotNNFilterOverlay(mean_image, attention, figure_id=i, interp='bilinear', colormap=cm.jet,
                                    title='IOP:{} mmHg, Dice:{:0.3f}'.format(pressure, dice), alpha=0.5, save_path=fp)
                # fn = os.path.join(vis_dir, fname+'-SpeckledVar-AttentionOverlay.png')
                # plotNNFilterOverlay(sv_image, attention, figure_id=i, interp='bilinear', colormap=cm.jet,
                #                     title='IOP:{} mmHg, Dice:{:0.3f}'.format(pressure, dice), alpha=0.5, save_path=fn)
                attentions.append(attention)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Attention')
    parser.add_argument('--experiment', type=str, required=False)
    args = parser.parse_args()
    if args.experiment is not None:
        experiment_dir = args.experiment
    else:
        # experiment_dir = '/home/kevin/Projects/oct_sc_segmentation/experiments_5/7_fold_cv/unet_fused_att_multiscale_dilated_res/2021-01-23/23-48-16/'
        # experiment_dir = '/home/kevin/Projects/oct_sc_segmentation/experiments_5/7_fold_cv/unet_fused_att_multiscale_dilated_res/2021-01-25/21-19-18'
        # experiment_dir = '/home/kevin/Projects/oct_sc_segmentation/experiments_5/7_fold_cv/unet_att_dsv/2021-01-25/01-33-15'
        experiment_dir = '/home/kevin/Projects/oct_sc_segmentation/experiments_5/6fold_cv_high_quality_correct_normalization_cosanneal_tmult2_t0_20/unet_att_resconv/2021-03-28/23-58-03'
    experiment_datetime = '/'.join(experiment_dir.rsplit('/')[-2:])
    fp = os.path.join(experiment_dir, 'config.yaml')
    with open(fp, 'r') as file:
        cfg = yaml.load(file)
        cfg = DictConfig(cfg)
    cfg.args.experiment = os.path.join(cfg.args.experiment, experiment_datetime)
    model_type = 'fusedattention2'
    # model_type = 'attentionblock2'
    test_dir = 'visualize_attention'
    main(cfg, save_root=test_dir, layer_name=model_type)

