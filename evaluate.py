"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from runstats import Statistics
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from data_fastMRI import transforms


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(
        #gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=False, data_range=pred.max() - pred.min()
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )

def mse_torch(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    gt, pred = gt.cpu().detach().numpy(), pred.cpu().detach().numpy()
    return np.mean((gt - pred) ** 2)


def nmse_torch(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    gt = gt.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr_torch(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    gt, pred = gt.cpu().detach().numpy(), pred.cpu().detach().numpy(),
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim_torch(gt, pred):
    gt, pred = gt.cpu().detach().numpy(), pred.cpu().detach().numpy()
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(
        #gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=False, data_range=pred.max() - pred.min()
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )


def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file, 'r') as target, h5py.File(
          args.predictions_path / tgt_file.name, 'r') as recons:
            if args.acquisition and args.acquisition != target.attrs['acquisition']:
                continue

            if args.acceleration and target.attrs['acceleration'] != args.acceleration:
                continue

            target = target[recons_key][()]
            recons = recons['reconstruction'][()]
            target = transforms.center_crop(target, (target.shape[-1], target.shape[-1]))
            recons = transforms.center_crop(recons, (target.shape[-1], target.shape[-1]))
            metrics.push(target, recons)
    return metrics


