"""
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from data_fastMRI import transforms


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge,  sample_rate, phase, seed=42):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' else 'reconstruction_rss'

        phase = root.split("/")[-1]
        self.phase = phase
        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        print('Loading dataset :', root)
        random.seed(seed)
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            data = h5py.File(fname, 'r')
            padding_left = None
            padding_right = None
            kspace = data['kspace']
            num_slices = kspace.shape[0]

            num_start = 0
            self.examples += [(fname, slice, padding_left, padding_right) for slice in range(num_start, num_slices - num_start)]
        if self.phase.startswith('train') and sample_rate > 1:
            self.paths_for_run = []
            for element in self.examples:
                for i in range(int(sample_rate)):
                    self.paths_for_run.append(element)
            self.examples = self.paths_for_run


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice, padding_left, padding_right = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            mask = np.asarray(data['mask']) if 'mask' in data else None
            target = data[self.recons_key][slice] if self.recons_key in data else None
            attrs = dict(data.attrs)
            attrs['padding_left'] = padding_left
            attrs['padding_right'] = padding_right
            return self.transform(kspace, mask, target, attrs, fname.name, slice)


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, resolution, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, mask, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            mask (numpy.array): Mask from the test dataset
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
        """
        kspace = transforms.to_tensor(kspace)

        # Apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = transforms.apply_mask(
                kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2n(masked_kspace)

        target = transforms.ifft2n(kspace)

        # Absolute value
        # abs_image = transforms.complex_abs(image)
        # mean = torch.tensor(0.0)
        # std = abs_image.mean()

        abs_target = transforms.complex_abs(target)
        mean = torch.tensor(0.0)
        std = abs_target.max()

        # Normalize input
        image = image.permute(2, 0, 1)
        image = transforms.normalize(image, mean, std, eps=0)

        # Normalize target
        target = target.permute(2, 0, 1)
        target = transforms.normalize(target, mean, std, eps=0)

        masked_kspace = masked_kspace.permute(2, 0, 1)
        masked_kspace = transforms.normalize(masked_kspace, mean, std, eps=0)

        mask = mask.repeat(image.shape[1], 1, 1).squeeze().unsqueeze(0)
        # print("max=", attrs['max'].astype(np.float32))
        return image, target, mean, std, attrs['norm'].astype(np.float32), fname, slice, attrs['max'].astype(np.float32), mask, masked_kspace


def create_dataset(data_path, data_transform, bs, shuffle, phase, challenge, num_workers, sample_rate=None, display=False):
    dataset = SliceData(
        root=data_path,
        transform=data_transform,
        sample_rate=sample_rate,
        challenge=challenge,
        phase=phase
    )
    if display:
        dataset = [dataset[i] for i in range(100, 108)]
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, pin_memory=True,num_workers=num_workers)

def kspace_dc(pred_kspace, ref_kspace, mask):
    return (1 - mask) * pred_kspace + mask * ref_kspace


def image_dc(pred_image, ref_kspace, mask):
    return transforms.ifft2n(kspace_dc(transforms.fft2n(pred_image), ref_kspace, mask))


#
# if __name__ == '__main__':
#     from config.configs_CCrestormer import config
#     from data_fastMRI.subsample import create_mask_for_mask_type
#     import evaluate
#     from torchvision.utils import save_image
#     import os
#
#     test_dir = "/mnt/disk/zxh/dataset/fastMRI/val2"
#
#     mask = create_mask_for_mask_type(config.mask_type, config.center_fractions,
#                                              config.accelerations)
#
#     val_data_transform = DataTransform(config.resolution, config.challenge, mask, use_seed=True)
#
#     dataset_test = create_dataset(test_dir, val_data_transform, 1, False, 'valid',
#                                   config.challenge, config.data_loaders_num_workers, 1.0)
#
#     metrics = np.zeros((len(dataset_test), 3))
#     print(len(dataset_test), metrics.shape)
#     for ii, batch in enumerate(dataset_test, 0):
#         zf, label, mean, std, norm, fname, slice, max, mask, masked_kspace = batch
#
#         mag = lambda x: (x[:, 0:1, ...] ** 2 + x[:, 1:2, ...] ** 2) ** 0.5
#         # zf, label = transforms.complex_abs(zf), transforms.complex_abs(label)
#         mean = mean.unsqueeze(1).unsqueeze(2)
#         std = std.unsqueeze(1).unsqueeze(2)
#         zf = transforms.complex_abs(zf) * std + mean
#         label = transforms.complex_abs(label) * std + mean
#         print("\r"+"{}/{}".format(ii+1, len(dataset_test)), end="", flush=True)
#         # zf = mag(zf) * std + mean
#         # label = mag(label) * std + mean
#         # zf, label = zf.unsqueeze(0), label.unsqueeze(0)
#         x_input, x_target = zf.to("cuda"), label.to("cuda")
#
#         metrics[ii, 0] = evaluate.psnr_torch(x_target, x_input)
#         metrics[ii, 1] = evaluate.ssim_torch(x_target, x_input)
#         metrics[ii, 2] = evaluate.nmse_torch(x_target, x_input)
#
#         if 5 <= ii <= len(dataset_test) - 5:
#             save_image(mask, os.path.join('imgs','1_mask.png'), normalize=False, scale_each=True)
#             save_image(label, os.path.join('imgs','label_{:05d}.png'.format(ii)), nrow=8, normalize=True, scale_each=True)
#             save_image(zf, os.path.join('imgs','zf_{:05d}_{:.4f}.png'.format(ii, metrics[ii, 0])), nrow=8, normalize=True, scale_each=True)
#
#     # print(metrics[0, 0].shape)
#     # metrics[:, 1] = metrics[:, 1] * 100
#     print("\nHybrid")
#     print("%.3f +/- %.3f" % (metrics[:, 0].mean(), metrics[:, 0].std()))
#     print("%.3f +/- %.3f" % (metrics[:, 1].mean(), metrics[:, 1].std()))
#     print("%.3f +/- %.3f" % (metrics[:, 2].mean(), metrics[:, 2].std()))
    #
#
