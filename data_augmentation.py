"""Code for "K. C. Choy, G. Li, W. D. Stamer, S. Farsiu, Open-source deep learning-based automatic segmentation of
mouse Schlemm’s canal in optical coherence tomography images. Experimental Eye Research, 108844 (2021)."
Link: https://www.sciencedirect.com/science/article/pii/S0014483521004103
DOI: 10.1016/j.exer.2021.108844
The data and software here are only for research purposes. For licensing, please contact Duke University's Office of
Licensing & Ventures (OLV). Please cite our corresponding paper if you use this material in any form. You may not
redistribute our material without our written permission. """

import torch
from skimage import io, transform
import numpy as np
import torchvision.transforms.functional as TF
from numbers import Number


class SelectKeys(object):
    def __init__(self, keys=('image', 'speckled_var', 'mask')):
        self.keys = keys

    def __call__(self, sample):
        transformed_sample = {}
        metadata = sample.pop('metadata')
        for key in self.keys:
            transformed_sample[key] = sample[key]
        transformed_sample['metadata'] = metadata
        return transformed_sample


class ScaleRange(object):
    def __init__(self, keys, intensity_range=(0, 1)):
        self.keys = keys
        self.intensity_range = intensity_range

    def __call__(self, sample):
        # scale range to [0, 1]
        for key in self.keys:
            image = sample[key]
            im_max = image.max()
            im_min = image.min()
            sample[key] = (image - im_min) / (im_max - im_min)

        return sample


class Normalize(object):
    def __init__(self, keys, mean, std, inplace=False):
        self.keys = keys
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        inputs = [sample.pop(key) for key in self.keys]
        inputs = torch.stack(inputs, dim=0)
        # normalize distribution
        inputs = TF.normalize(inputs, self.mean, self.std, self.inplace)
        sample['inputs'] = inputs
        # target = sample['mask']
        # metadata = sample['metadata']

        # return {'inputs': inputs, 'metadata': metadata, 'mask': target}
        return sample


class Crop(object):
    def __init__(self, h1, w1, h2, w2):
        # assert isinstance(output_size, (int, tuple))
        # if isinstance(output_size, int):
        #     self.output_size = (output_size, output_size)
        # else:
        #     assert len(output_size) == 2
        #     self.output_size = output_size
        self.h1 = h1
        self.w1 = w1
        self.h2 = h2
        self.w2 = w2

    def __call__(self, sample):
        # h, w = sample[keys[0]].shape[:2]
        # new_h, new_w = self.output_size
        # top = 0
        # left = 0

        transformed_sample = {}
        metadata = sample.pop('metadata')

        for key, image in sample.items():
            # transformed_sample[key] = image[top: top + new_h,
            #                                 left: left + new_w]
            transformed_sample[key] = image[self.h1: self.h2,
                                      self.w1: self.w2]
        transformed_sample['metadata'] = metadata

        return transformed_sample


class CenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        transformed_sample = {}
        metadata = sample.pop('metadata')

        for key, image in sample.items():
            transformed_sample[key] = TF.center_crop(image, self.output_size)
        transformed_sample['metadata'] = metadata

        return transformed_sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

   Args:
       output_size (tuple or int): Desired output size. If tuple, output is
           matched to output_size. If int, smaller of image edges is matched
           to output_size keeping aspect ratio the same.
   """

    def __init__(self, output_size, keys=('image', 'speckled_var')):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.keys = keys

    def __call__(self, sample):
        # keys = list(sample.keys())
        # h, w = sample[keys[0]].shape[:2]
        # # h, w = image.shape[:2]
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        #     new_h, new_w = self.output_size

        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        transformed_sample = {}
        metadata = sample.pop('metadata')
        for key, image in sample.items():
            transformed_sample[key] = TF.resize(image, (new_h, new_w))
        transformed_sample['metadata'] = metadata

        return transformed_sample


class ToPILImage(object):
    def __call__(self, sample):
        transformed_sample = {}
        metadata = sample.pop('metadata')

        for key, image in sample.items():
            # transformed_sample[key] = TF.to_pil_image(image.astype('float32'))
            # image = image.astype('uint8')
            transformed_sample[key] = TF.to_pil_image(image.astype('float32'))
            # transformed_sample[key] = TF.to_pil_image(image.astype('uint8'))
            # transformed_sample[key] = TF.to_pil_image(image)
        transformed_sample['metadata'] = metadata
        return transformed_sample


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        # flip = torch.randint(low=0, high=1, size=(1,)) == 1
        # flip = random.randint(0, 1) == 1

        if torch.rand(1) < self.p:
            transformed_sample = {}
            metadata = sample.pop('metadata')
            for key, image in sample.items():
                transformed_sample[key] = TF.hflip(image)
            transformed_sample['metadata'] = metadata
            return transformed_sample

        else:
            return sample

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        flip = torch.rand(1)
        if flip < self.p:
            transformed_sample = {}
            metadata = sample.pop('metadata')
            for key, image in sample.items():
                transformed_sample[key] = TF.vflip(image)
            transformed_sample['metadata'] = metadata
            return transformed_sample

        else:
            return sample


class RandomTranslate(object):
    def __init__(self, translate=None):
        self.translate = translate

    def __call__(self, sample):
        translate = self.translate

        img_size = sample['mask'].size
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        transformed_sample = {}
        metadata = sample.pop('metadata')
        for key, image in sample.items():
            transformed_sample[key] = TF.affine(image, angle=0, translate=translations, scale=1, shear=0)

        transformed_sample['metadata'] = metadata

        return transformed_sample


class RandomScale(object):
    def __init__(self, scale_ranges=None):
        self.scale_ranges = scale_ranges
        # self.r1 = r1
        # self.r2 = r2

    def __call__(self, sample):
        # r = torch.rand(1)
        scale_ranges = self.scale_ranges
        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0
        # scale = (self.r1 - self.r2) * r + self.r2
        transformed_sample = {}
        metadata = sample.pop('metadata')
        for key, image in sample.items():
            transformed_sample[key] = TF.affine(image, angle=0, translate=(0, 0), scale=scale, shear=0)

        transformed_sample['metadata'] = metadata

        return transformed_sample


class RandomRotate(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, sample):

        # image, speckled_var, mask, skew, kurtosis = sample['image'], sample['speckled_var'], sample['mask'], \
        #                                             sample['skew'], sample['kurtosis']

        # angle = random.uniform(self.degrees[0], self.degrees[1])
        # r = np.random.sample()  # does not work
        r = torch.rand(1).item()
        angle = (self.degrees[0] - self.degrees[1]) * r + self.degrees[1]

        transformed_sample = {}
        metadata = sample.pop('metadata')

        for key, image in sample.items():
            transformed_sample[key] = TF.rotate(image, angle,
                                                self.resample, self.expand, self.center)

        transformed_sample['metadata'] = metadata
        return transformed_sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

   Args:
       output_size (tuple or int): Desired output size. If int, square crop
           is made.
   """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # image, speckled_var, mask, skew, kurtosis = sample['image'], sample['speckled_var'], sample['mask'], \
        #                                             sample['skew'], sample['kurtosis']

        keys = list(sample.keys())
        h, w = sample[keys[0]].shape[:2]
        # h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = torch.randint(low=0, high=h - new_h, size=(1,)) if h != new_h else 0
        left = torch.randint(low=0, high=w - new_w, size=(1,)) if w != new_w else 0

        # top = np.random.randint(0, h - new_h)
        # left = np.random.randint(0, w - new_w)

        transformed_sample = {}
        metadata = sample.pop('metadata')

        for key, image in sample.items():
            transformed_sample[key] = image[top: top + new_h,
                                      left: left + new_w]

        transformed_sample['metadata'] = metadata

        return transformed_sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image, speckled_var, mask, skew, kurtosis = sample['image'], sample['speckled_var'], sample['mask'], \
        #                                             sample['skew'], sample['kurtosis']
        transformed_sample = {}
        metadata = sample.pop('metadata')

        for key, image in sample.items():
            transformed_sample[key] = TF.to_tensor(image).squeeze(0)
        transformed_sample['metadata'] = metadata
        return transformed_sample


class StackInputs(object):
    """Concatenate average image and speckled variance image."""

    def __init__(self, input_keys=('image', 'speckled_var', 'skew', 'kurtosis'), target_key='mask'):
        self.input_keys = input_keys
        self.target_key = target_key

    def __call__(self, sample):
        inputs = []
        for key in self.input_keys:
            inputs.append(sample.pop(key).unsqueeze(0))
        inputs = torch.cat(inputs, dim=0)
        sample['inputs'] = inputs
        # target = sample[self.target_key]
        # metadata = sample['metadata']
        # return {'inputs': inputs, 'metadata': metadata, 'mask': target}
        return sample
