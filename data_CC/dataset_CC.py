'''
# -----------------------------------------
Data Loader
# -----------------------------------------
'''

import os
import glob
import torch
import torch.utils.data as data
from data_cc.data_augment import *


def np_to_complex(x):
    return x[..., 0] + 1j * x[..., 1]

def complex_to_channel(data):
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return data


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def getimage_slice(filepath, imshape, norm, start_num=0):
    num = 0
    files = sorted(glob.glob(os.path.join(filepath, "*.npy")))
    # print(files)
    slice_nums = []

    for fname in sorted(files):
        slice_num = np.load(fname).shape[0]
        num += slice_num
        slice_nums.append(slice_num)

    num -= 2 * start_num * len(files)
    rec_data = np.zeros((num, imshape[0], imshape[1], 2))
    kspace_data = np.zeros((num, imshape[0], imshape[1], 2))
    aux_counter = 0
    for ii in range(len(files)):
        slice_num = slice_nums[ii]
        aux_kspace = np.load(files[ii]) / norm
        aux = slice_num - 2 * start_num
        aux2 = np.fft.ifft2(aux_kspace[start_num: slice_num - start_num, :, :, 0] + 1j * aux_kspace[start_num: slice_num -start_num, :, :, 1])
        rec_data[aux_counter:aux_counter + aux, :, :, 0] = aux2.real
        rec_data[aux_counter:aux_counter + aux, :, :, 1] = aux2.imag
        kspace_data[aux_counter:aux_counter + aux, :, :, 0] = aux_kspace[start_num:slice_num-start_num, :, :, 0]
        kspace_data[aux_counter:aux_counter + aux, :, :, 1] = aux_kspace[start_num:slice_num-start_num, :, :, 1]
        aux_counter += aux
    return aux_counter, kspace_data, rec_data


# save k-space and image domain stats
def save_stas(kspace_data, rec_data, under_rate):
    stats = np.zeros(4)
    stats[0] = kspace_data.mean()
    stats[1] = kspace_data.std()
    aux = np.abs(rec_data[:, :, :, 0] + 1j * rec_data[:, :, :, 1])
    stats[2] = aux.mean()
    stats[3] = aux.std()
    np.save("Data/stats_fs_norm_" + under_rate + ".npy", stats)
    return stats


class DatasetCC(data.Dataset):
    def __init__(self, aux_couter, kspace_data, rec_data, mask,  is_training=False):
        super(DatasetCC, self).__init__()

        self.mask = mask
        self.is_training = is_training
        self.aux_counter = aux_couter
        self.kspace_data = kspace_data
        self.rec_data = rec_data

    def __len__(self):
        return self.aux_counter

    def __getitem__(self, index):
        kspace = self.kspace_data[index]
        rec_data = self.rec_data[index]

        if self.is_training:
            seed = np.random.randint(0, 2024, 1)
            rec_data = random_reverse(rec_data, seed)
            rec_data = random_flip(rec_data, seed)
            rec_data = random_rotate(rec_data, seed)
            kspace = complex_to_channel(np.fft.fft2(rec_data[..., 0] + 1j * rec_data[..., 1]))


        rec_img = rec_data.copy()
        kspace = np.expand_dims(kspace, 0)

        kspace_under = kspace.copy()
        sampling_mask = self.mask
        mask = np.concatenate((sampling_mask[:, :, np.newaxis], sampling_mask[:, :, np.newaxis]), axis=-1) > 0
        mask = mask.astype(np.float32)


        kspace_under = kspace_under * np.fft.ifftshift(mask)
        zf = np.fft.ifft2((kspace_under[..., 0] + 1j * kspace_under[..., 1]))

        zf = np.squeeze(complex_to_channel(zf), 0)
        kspace_under = np.squeeze(kspace_under, 0)

        zf, rec_img, mask, kspace_under = float2tensor3(zf), float2tensor3(rec_img), float2tensor3(mask), float2tensor3(kspace_under)

        return {'L': zf, 'H': rec_img,  'mask': mask,  'y_measure': kspace_under}


# convert float to 3-dimensional torch tensor
def float2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2 or data.size(-3) == 2
    return (data ** 2).sum(dim=-1).sqrt() if data.size(-1) == 2 else (data ** 2).sum(dim=-3).sqrt()

def get_datasets(config, mask, start_num=0, is_training=False):
    """Get the dataset loaders.

    Returns
    -------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        The training and validation data loaders
    """
    imshape = (config.input_shape[0], config.input_shape[1])
    norm = np.sqrt(config.input_shape[0] * config.input_shape[1])
    filepath = config.trainDir if is_training else config.validDir
    sample_counter, kspace, rec_data = getimage_slice(filepath, imshape, norm, start_num=start_num)
    dataset = DatasetCC(sample_counter, kspace, rec_data, mask, is_training=is_training)
    return dataset

