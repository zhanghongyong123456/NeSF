import os
import shutil
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
import json
import numpy as np
import cv2
import random
from pytorch_ssim import ssim as ssim_metric
from PIL import Image

def gamma_correction(image, gamma=2.2):
    return image ** (1 / gamma)


def do_tone_map(hdr, save_path):
    tonemapped = gamma_correction(hdr)  
    return tonemapped# * 255

def load_npy(npy_path):
    return np.load(npy_path)

def load_img(img_path):
    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32)[:, :, :3] / 255.0
    return img

def load_indice(indice_path):
    with open(indice_path, "r") as f:
        data = f.readline()
    data = data.split(' ')
    return [int(data[0]), int(data[1])]

def create_or_recreate_folders(configs):
    """
    deletes existing folder if they already exist and
    recreates then. Only valid for training mode. does not work in
    resume mode
    :return:
    """

    folders = [configs['display_folder'],
               configs['summary'],
               configs['epoch_folder'],
               configs['display_val']]

    # iterate through the folders and delete them if they exist
    # then recreate them.
    # otherwise simply create them
    for i in range(len(folders)):
        folder = folders[i]
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            os.mkdir(folder)
        else:
            os.mkdir(folder)

def create_or_recreate_folders_v2(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        os.mkdir(folder)
    else:
        os.mkdir(folder)

def rotate_hdr(image, angle):
    # angle : 0 - 360 for rotation angle
    # (4096, 8192, 3)
    B, C, H, W = image.shape
    width = (angle / 360.0) * W

    front = image[:, :, :, 0:W - int(width)]
    back = image[:, :, :, W - int(width):]

    rotated = torch.cat([back, front], 3)
    return rotated


def mask_l1(output, target, mask, loss_f):
    loss = loss_f(output, target)
    loss = (loss * mask.float()).sum()  # gives \sigma_euclidean over unmasked elements

    if mask.shape[1] == 3:
        non_zero_elements = mask.sum()
    elif mask.shape[1] == 1:
        non_zero_elements = mask.sum() * 3
    else:
        raise RuntimeError('MaskChannelError')
    if non_zero_elements == 0:
        mse_loss_val = 0
    else:
        mse_loss_val = loss / non_zero_elements

    return mse_loss_val

class Mask_PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self, max_value):
        self.name = "Mask_PSNR"
        self.max_value = max_value # 255 or 1

    @staticmethod
    def __call__(img1, img2, mask, max_value):
        if mask.shape[1] == 1:
            mask = mask.repeat((1, 3, 1, 1))
        elif mask.shape[1] == 3:
            pass
        else:
            raise RuntimeError('MaskChannelError')
        pixelwise_loss = (img1 * mask.float() - img2 * mask.float()) ** 2
        non_zero_elements = mask.sum() 
        if non_zero_elements == 0:
            psnr = 30
        else:
            mse = pixelwise_loss.sum() / non_zero_elements
            psnr = 20 * torch.log10(max_value / torch.sqrt(mse))

        return psnr

def mask_ssim(image1, image2, mask):
    """ input tensor
    image1: B 3 H W
    image2: B 3 H W
    mask: B 1or3 H W
    """
    # ensure mask channel == 3
    if mask.shape[1] == 1:
        mask = mask.repeat((1, 3, 1, 1))
    elif mask.shape[1] == 3:
        pass
    else:
        raise RuntimeError('MaskChannelError')

    non_zero_elements = mask.sum() 
    if non_zero_elements == 0:
        ssim_value = 1
    else:
        ssim_value = ssim_metric(image1, image2, mask=mask)
    
    return ssim_value

def load_config(file):
    """
    takes as input a file path and returns a configuration file
    that contains relevant information to the training of the NN
    :param file:
    :return:
    """

    # load the file as a raw file
    loaded_file = open(file)

    # conversion from json file to dictionary
    config = json.load(loaded_file)

    # returning the file to the caller
    return config

### Eq. 3 in the main paper ###
def render_shading(shading_bases, illum_descriptor):
    expanded_shading_bases = torch.unsqueeze(shading_bases, 2).repeat(1, 1, 3, 1, 1)
    rendered = torch.sum((illum_descriptor * expanded_shading_bases), dim=1)
    return rendered
