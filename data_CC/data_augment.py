import numpy as np
# import skimage.io
# from glob import glob
# from natsort import natsorted
# import os
from scipy.ndimage.interpolation import rotate
import scipy.misc


def data_augmentation(image, mode):
    out = np.transpose(image, (1,2, 0))

    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))



def data_aug(image, mode):
    assert ((image.ndim == 2) | (image.ndim == 3))
    if mode == 0:
        out = image
    elif mode == 1:
        # flip up and down
        out = image[..., ::-1, ::1]
    elif mode == 2:
        out = image.copy()
        out = rotate(out, 90, axes=(1, 2), reshape=False)
        out = out
    elif mode == 3:
        out = image.copy()
        out = rotate(out, 90, axes=(1, 2), reshape=False)
        out = out[..., ::-1, ::1]

    elif mode == 4:
        out = rotate(image, 180, axes=(1, 2), reshape=False)
        out = out

    elif mode == 5:
        out = rotate(image, 180, axes=(1, 2), reshape=False)
        out = out[..., ::-1, ::1]


    elif mode == 6:
        out = rotate(image, 270, axes=(1, 2), reshape=False)
    elif mode == 7:
        out = rotate(image, 270, axes=(1, 2), reshape=False)
        out = out[..., ::-1, ::1]
    return out


def random_flip(image, seed=None):
    assert ((image.ndim == 2) | (image.ndim == 3))
    if seed:
        np.random.seed(seed)
    random_flip = np.random.randint(1, 5)
    if random_flip == 1:
        flipped = image[::1, ::-1, ...]
        image = flipped
    elif random_flip == 2:
        flipped = image[::-1, ::1, ...]
        image = flipped
    elif random_flip == 3:
        flipped = image[::-1, ::-1, ...]
        image = flipped
    elif random_flip == 4:
        flipped = image
        image = flipped
    return image


def random_reverse(image, seed=None):
    assert ((image.ndim == 2) | (image.ndim == 3))
    if seed:
        np.random.seed(seed)
    random_reverse = np.random.randint(1, 3)
    if random_reverse == 1:
        reverse = image[::1, ...]
    elif random_reverse == 2:
        reverse = image[::-1, ...]
    return reverse


def random_rotate(image, seed=None):
    assert ((image.ndim == 2) | (image.ndim == 3))
    if seed:
        np.random.seed(seed)
    random_rotatedeg = np.random.randint(-40, 40)
    rotated = image.copy()
    from scipy.ndimage.interpolation import rotate
    rotated = rotate(image, random_rotatedeg, axes=(0, 1), reshape=False)
    image = rotated
    return image


def random_square_rotate(image, seed=None):
    assert ((image.ndim == 2) | (image.ndim == 3))
    if seed:
        np.random.seed(seed)
    random_rotatedeg = 90 * np.random.randint(0, 4)
    rotated = image.copy()
    from scipy.ndimage.interpolation import rotate
    rotated = rotate(rotated, random_rotatedeg, axes=(0, 1))
    image = rotated
    return image


def random_crop(image, seed=None):
    assert ((image.ndim == 2) | (image.ndim == 3))
    if seed:
        np.random.seed(seed)
    limit = np.random.randint(1, 12)  # Crop pixel
    randy = np.random.randint(0, limit)
    randx = np.random.randint(0, limit)
    cropped = image[:, randy:-(limit - randy), randx:-(limit - randx)]
    return cropped

