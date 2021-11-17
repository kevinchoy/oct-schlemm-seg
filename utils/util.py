from __future__ import print_function
import os
import torch
import yaml
from PIL import Image
from omegaconf import DictConfig
from torch import nn
from torchvision.transforms import functional as TF


def load_saved_model(model, checkpoint, data_parallel=False, device='cuda'):
    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])  # Loading
    else:
        model.load_state_dict(checkpoint)  # Loading
    if data_parallel and torch.cuda.device_count() > 1:
        print('Using {} GPUs'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model = model.to(device)
    return model


def save(model, save_dir, fn):
    save_path = os.path.join(save_dir, fn)
    torch.save(model.state_dict(), save_path)  # Saving


def save_checkpoint(model, optimizer, epoch, ckpt_path, loss, dice):
    if isinstance(model, nn.DataParallel):  # model in parallel wrapper
        model = model.module
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'dice': dice,
    }, ckpt_path)


def load_checkpoint(ckpt_path, model, optimizer):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    dice = checkpoint['dice']
    return model, optimizer, epoch, loss, dice


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def undo_transforms(mask_transformed, shape, axial_res=1170/1024, lateral_res=1):
    # resize image to original size 361 x 440
    height_orig_pixels, width_orig_pixels = shape
    # height_um = round(height_orig_pixels * axial_res)  # 412.470703125
    # width_um = round(width_orig_pixels * lateral_res)  # 440
    height_um = height_orig_pixels * axial_res  # 412
    width_um = width_orig_pixels * lateral_res  # 440
    scaling_factor = 512/min(height_um, width_um)  # resize height (smaller dimension) to 512
    height_scaled = round(scaling_factor * height_um)  # height: 412 --> 512
    width_scaled = round(scaling_factor * width_um)  # width: 440 --> 546

    # Undo center crop
    mask_transformed = center_pad(mask_transformed, (height_scaled, width_scaled))  # 512 x 546
    # im = TF.to_pil_image(mask_transformed)
    # im = TF.resize(im, (height_pixels, width_pixels))  # 381 x 460
    # Undo resize
    im = TF.resize(mask_transformed.unsqueeze(0), (height_orig_pixels, width_orig_pixels), interpolation=Image.NEAREST)
    return im


def center_pad(im, out_shape):
    h0, w0 = im.shape
    h1, w1 = out_shape
    dy = round((h1-h0)/2)
    dx = round((w1-w0)/2)
    im_array = torch.zeros(out_shape)
    im_array[dy:dy+h0, dx:dx+w0] = im
    return im_array


def load_cfg(fp):
    with open(fp, 'r') as file:
        cfg = yaml.load(file)
        cfg = DictConfig(cfg)
    return cfg