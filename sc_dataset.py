from __future__ import print_function, division
import os
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from PIL import Image
import csv

# import visdom
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class OCTSchlemmsCanalDataset(Dataset):
    """OCT Schlemm's Canal dataset."""

    def __init__(self, root_dir,
                 fp='/home/kevin/Projects/oct_sc_segmentation/OCT-SchlemmsCanal-Seg/sc_data_master.csv',
                 transform=None, file_ext='*.tif', max_iop=None):
        """
        Args:
            root_dir (string): Directory containing the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        image_paths = sorted((Path(__file__).parent / root_dir).rglob('*/mask/'+file_ext))  # [path/01/mask/fn1, /path/01/mask/fn2, ...]
        self.subject_indices = [im_path.parts[-3] for im_path in image_paths]  # [01, ..., 02, ...]
        self.fnames = [im_path.parts[-1].rsplit('-', 1)[0] for im_path in image_paths]  # [fn1, fn2, ...]
        self.image_paths = [os.path.join(*im_path.parts[:-2]) for im_path in image_paths]  # [path/01/, path/01/, ...]
        self.pressure = {}

        if fp:
            path = Path(__file__).parent / fp
            with path.open() as f:
            # with open(fp, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    subject_idx = row[2]
                    if subject_idx != '':
                        fname = str(subject_idx).zfill(2) + '-' + row[1].lower()
                        if row[4] != '':
                            iop = float(row[4])
                        else:
                            iop = float('nan')
                        self.pressure[fname] = iop

        if max_iop:
            high_iop_idx = []
            for i, fname in enumerate(self.fnames):
                iop = self.pressure[fname.lower()]
                if iop > max_iop:
                    # print(i, self.image_paths[i], fname, iop)
                    high_iop_idx.append(i)
                    del self.pressure[fname.lower()]
            for idx in sorted(high_iop_idx, reverse=True):
                # print(self.subject_indices[idx], self.fnames[idx)
                del self.fnames[idx]
                del self.image_paths[idx]
                del self.subject_indices[idx]

        self.mouse_id = sorted(list(set(self.subject_indices)))
        self.subject_dict = {}
        self.subject_list = []

        for m in self.mouse_id:
            self.subject_dict[m] = []
        for idx, m in enumerate(self.subject_indices):
            self.subject_dict[m].append(idx)
        for m in self.mouse_id:
            self.subject_list.append(self.subject_dict[m])

        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_paths[idx]
        sample_name = self.fnames[idx]
        metadata = {'image_path': self.image_paths[idx],
                    'fname': self.fnames[idx],
                    'pressure': self.pressure[sample_name.lower()],
                    'subject_idx': self.subject_indices[idx]
                    }

        img_path = os.path.join(image_path, 'average', sample_name + '-Average.tif')
        image = io.imread(img_path)
        sv_path = os.path.join(image_path, 'speckled_var', sample_name + '-SpeckledVar.tif')
        speckled_var = io.imread(sv_path)
        mask_path = os.path.join(image_path, 'mask', sample_name + '-Mask.tif')
        mask = io.imread(mask_path)

        sample = {'image': image, 'speckled_var': speckled_var, 'mask': mask, 'metadata': metadata}

        if self.transform:
            sample = self.transform(sample)

        return sample


class OCTSchlemmsCanalDatasetUnlabeled(Dataset):
    """OCT Schlemm's Canal dataset."""

    def __init__(self, root_dir, transform=None, file_ext='*.tif'):
        """
        Args:
            root_dir (string): Directory containing the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        image_paths = sorted(Path(root_dir).rglob('*/average/'+file_ext))  # [path/01/mask/fn1, /path/01/mask/fn2, ...]
        self.subject_indices = [im_path.parts[-3] for im_path in image_paths]  # [01, ..., 02, ...]
        self.fnames = [im_path.parts[-1].rsplit('-', 1)[0] for im_path in image_paths]  # [fn1, fn2, ...]
        self.image_paths = [os.path.join(*im_path.parts[:-2]) for im_path in image_paths]  # [path/01/, path/01/, ...]
        self.mouse_id = sorted(list(set(self.subject_indices)))
        self.subject_dict = {}
        self.subject_list = []

        for m in self.mouse_id:
            self.subject_dict[m] = []
        for idx, m in enumerate(self.subject_indices):
            self.subject_dict[m].append(idx)
        for m in self.mouse_id:
            self.subject_list.append(self.subject_dict[m])

        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_paths[idx]
        sample_name = self.fnames[idx]
        metadata = {'image_path': self.image_paths[idx],
                    'fname': self.fnames[idx],
                    # 'pressure': self.pressure[sample_name.lower()],
                    'subject_idx': self.subject_indices[idx]
                    }

        img_path = os.path.join(image_path, 'average', sample_name + '-Average.tif')
        image = io.imread(img_path)
        sv_path = os.path.join(image_path, 'speckled_var', sample_name + '-SpeckledVar.tif')
        speckled_var = io.imread(sv_path)

        sample = {'image': image, 'speckled_var': speckled_var, 'metadata': metadata}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    root_dir = '/home/kevin/Projects/oct_sc_segmentation/data_root/data_pigmented_raw_folds_2020-03-26'
    # root_dir = '/home/kevin/Projects/oct_sc_segmentation/data_root/data_all_folds_2020-08-25'
    csv_dir = '/home/kevin/Projects/oct_sc_segmentation/OCT-SchlemmsCanal-Seg/sc_data_master_38.csv'
    # dataset = OCTSchlemmsCanalSeriesDataset(root_dir, csv_dir)
    dataset = OCTSchlemmsCanalDataset(root_dir)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # vis = visdom.Visdom()

    print(len(dataset))
    batch_size = 2
    for fold, m in enumerate(dataset.mouse_id):
        print(dataset.mouse_id[fold], m)
        train_indices = [int(i) for i in range(len(dataset)) if i != fold]
        test_indices = [int(fold)]
        trainset = torch.utils.data.Subset(dataset, train_indices)
        validset = torch.utils.data.Subset(dataset, test_indices)
        print(train_indices)
        print(test_indices)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=2)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=False,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=2)
        for batch_idx, sample in enumerate(valid_loader):
            print('batch idx:', batch_idx)
            for s in sample:
                mask = s['mask']
                print(s['metadata']['fname'][0])
                # vis.image(mask)
            print('done')



