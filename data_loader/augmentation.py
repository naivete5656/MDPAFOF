import numpy as np
import random
from PIL import Image
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import RandomResizedCrop

import torch
from torch.nn import functional as F

def random_flipeach(crop1):
    # random flip function
    randval = np.random.randint(4)
    if randval == 1:
        crop1 = crop1[::-1]
    elif randval == 2:
        crop1 = crop1[:, ::-1]

    elif randval == 3:
        crop1 = crop1[::-1, ::-1]
    return crop1


def random_flip(crop1, crop2, gt=None):
    # random flip function
    randval = np.random.randint(4)
    if randval == 1:
        crop1 = crop1[::-1]
        crop2 = crop2[::-1]
        if gt is not None:
            gt = gt[::-1]
    elif randval == 2:
        crop1 = crop1[:, ::-1]
        crop2 = crop2[:, ::-1]
        if gt is not None:
            gt = gt[:, ::-1]

    elif randval == 3:
        crop1 = crop1[::-1, ::-1]
        crop2 = crop2[::-1, ::-1]
        if gt is not None:
            gt = gt[::-1, ::-1]

    return crop1, crop2, gt

def random_rot_each(crop1):
    # random rotation function
    randval = np.random.randint(4)
    crop1 = np.rot90(crop1, randval)
    return crop1

def random_rot(crop1, crop2, gt=None):
    # random rotation function
    randval = np.random.randint(4)
    crop1 = np.rot90(crop1, randval)
    crop2 = np.rot90(crop2, randval)
    if gt is not None:
        gt = np.rot90(gt, randval)

    return crop1, crop2, gt

def random_gaussian(crop1, crop2):
    # random rotation function
    crop1 = crop1 + np.random.rand(crop1.shape[0], crop1.shape[1]) * 0.01
    crop2 = crop2 + np.random.rand(crop2.shape[0], crop2.shape[1]) * 0.01
    return crop1, crop2

def random_brightness(img1, img2):
    img1 = img1 + 0.1 * np.random.normal(0, 0.3)
    img1 = img1.clip(0, 1)

    img2 = img2 + 0.1 * np.random.normal(0, 0.3)
    img2 = img2.clip(0, 1)
    return img1, img2

def random_scaling(img1, img2, hw_size=60):
    img_size = img1.shape
    sacaleparam = np.random.uniform(0.8, 1.2)
    new_size = int(img_size[0] * sacaleparam), int(img_size[1] * sacaleparam)
    img1 = np.array(Image.fromarray(img1).resize(new_size, resample=Image.BILINEAR))
    img2 = np.array(Image.fromarray(img2).resize(new_size, resample=Image.BILINEAR))

    center_pos = int(img1.shape[0] / 2)
    img1 = img1[center_pos-hw_size:center_pos+hw_size, center_pos-hw_size:center_pos+hw_size]
    img2 = img2[center_pos-hw_size:center_pos+hw_size, center_pos-hw_size:center_pos+hw_size]
    return img1, img2

def random_elastic_transform(crop1, crop2, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       reffered
       https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    """
    alpha = random.randint(0, 10)
    sigma = random.choice([3,7,11])
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = crop1.shape
    
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    crop1 =  map_coordinates(crop1, indices, order=1).reshape(shape)
    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    crop2 =  map_coordinates(crop2, indices, order=1).reshape(shape)

    return crop1, crop2
