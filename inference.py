import os
import os.path as osp
import torch
import shutil
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as utils
import torchvision.transforms.functional as TF
import json
import numpy as np
import cv2
import random
import lpips
import shutil
from model.illumination_model import IlluminationNet
from model.shading_model import ShadingNet
from model.albedo_model import AlbedoNet
from model.rendering_model import RenderingNet
from pytorch_ssim import ssim as ssim_metric
from utils import *


def load_data(example_path):
    # obtain paths
    unharmonized_input_path = osp.join(example_path, 'unharmonized_img.png')
    depth_path = osp.join(example_path, 'depth.npy')
    normal_path = osp.join(example_path, 'normal.npy')
    background_image_path =  osp.join(example_path, 'background_img.png')
    fore_mask_path = osp.join(example_path, 'foremask_img.png')
    gt_color_image_path = osp.join(example_path, 'gt_img.png')
    indice_path = osp.join(example_path, 'indice.txt')

    # load data
    depth_map = load_npy(depth_path).astype(np.float32)
    normal_map = load_npy(normal_path).astype(np.float32)
    fore_mask = load_img(fore_mask_path)
    gt_color_image = load_img(gt_color_image_path)
    unharmonized_input = load_img(unharmonized_input_path)
    background_image = load_img(background_image_path)
    indice = load_indice(indice_path)

    # convert numpy to tensor
    depth_tensor = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(depth_map), 0), 0)
    normal_tensor = torch.unsqueeze(torch.from_numpy(normal_map).permute(2, 0, 1), 0)
    fore_mask_tensor = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(fore_mask[...,0]), 0), 0)
    color_gt_tensor = torch.unsqueeze(torch.from_numpy(gt_color_image).permute(2, 0, 1), 0)
    background_tensor = torch.unsqueeze(torch.from_numpy(background_image).permute(2, 0, 1), 0)
    unharmonized_tensor = torch.unsqueeze(torch.from_numpy(unharmonized_input).permute(2, 0, 1), 0)

    dic = {
        'depth' : depth_tensor,
        'normal' : normal_tensor,
        'fore_mask' : fore_mask_tensor,
        'gt_color_image' : color_gt_tensor,
        'unharmonized_input' : unharmonized_tensor,
        'background_image' : background_tensor,
        'indice' : indice,
    }
    return dic

def do_image_placement(fore_unharmonized, fore_harmonized, fore_gt, background_img, foreground_mask, indice):
    # obtain the binary mask
    h, w, _ = fore_unharmonized.shape
    foreground_mask[foreground_mask > 0.5] = 1.0
    foreground_mask[foreground_mask < 0.5] = 0
    foreground_mask = foreground_mask.astype(np.uint8)

    fore_harmonized = fore_harmonized * np.expand_dims(foreground_mask, 2)

    # cropping the foreground image and mask
    points = cv2.boundingRect(foreground_mask)
    cropped_fore_unhar = fore_unharmonized[points[1]:(points[1] + points[3]), points[0]:(points[0] + points[2])]
    cropped_fore_har = fore_harmonized[points[1]:(points[1] + points[3]), points[0]:(points[0] + points[2])]
    cropped_fore_gt = fore_gt[points[1]:(points[1] + points[3]), points[0]:(points[0] + points[2])]
    cropped_mask = foreground_mask[points[1]:(points[1] + points[3]), points[0]:(points[0] + points[2])]

    ch, cw = cropped_mask.shape

    b_up = points[1]
    b_down = (points[1] + points[3])
    b_left = points[0]
    b_right = (points[0] + points[2])

    # obtain placement coordinates
    ratio = 0.7

    # top left case
    if indice[0] < h // 2 and indice[1] < w // 2:
        # print(f'---------------------------- Top Left ----------------------------------')
        ratio = 0.7
        nh, nw = int(ratio * ch), int(ratio * cw)
        h_index = indice[0] if (indice[0] + nh) <= h else ((indice[0] + h) - (indice[0] + nh))
        w_index = indice[1] if (indice[1] + nw) <= w else ((indice[1] + w) - (indice[1] + nw))


    # bottom left case
    elif indice[0] > h // 2 and indice[1] < w // 2:
        # print(f'---------------------------- bottom Left ----------------------------------')
        error = f'---------------------------- bottom Left ----------------------------------'
        ratio = 0.7
        nh, nw = int(ratio * ch), int(ratio * cw)
        h_index = indice[0] - nh if (indice[0] - nh) >= 0 else ((indice[0] + h) - (indice[0] + nh))
        w_index = indice[1] if (indice[1] + nw) <= w else ((indice[1] + w) - (indice[1] + nw))


    elif indice[0] < h // 2 and indice[1] > w // 2:
        # print(f'---------------------------- Top rhgit ----------------------------------')
        error = f'---------------------------- Top rhgit ----------------------------------'
        ratio = 0.7
        nh, nw = int(ratio * ch), int(ratio * cw)
        h_index = indice[0] if (indice[0] + nh) <= h else ((indice[0] + h) - (indice[0] + nh))
        w_index = indice[1] - nw if (indice[1] - nw >= 0) else 0

    else:
        # print(f'---------------------------- bottom right ----------------------------------')
        ratio = 0.7
        nh, nw = int(ratio * ch), int(ratio * cw)
        h_index = indice[0] - nh if (indice[0] - nh) >= 0 else ((indice[0] + h) - (indice[0] + nh))
        w_index = indice[1] - nw if (indice[1] - nw >= 0) else 0

    nw = nw if nw > 0 else 1
    nh = nh if nh > 0 else 1

    reshaped_crop_fore_un = cv2.resize((cropped_fore_unhar * 255).astype(np.uint8), (nw, nh))
    reshaped_crop_fore_har = cv2.resize((cropped_fore_har * 255).astype(np.uint8), (nw, nh))
    reshaped_crop_fore_mask = cv2.resize((cropped_mask * 255).astype(np.uint8), (nw, nh))
    reshaped_crop_fore_gt = cv2.resize((cropped_fore_gt * 255).astype(np.uint8), (nw, nh))

    reshaped_crop_fore_un = reshaped_crop_fore_un / 255.0
    reshaped_crop_fore_har = reshaped_crop_fore_har / 255.0
    reshaped_crop_fore_mask = reshaped_crop_fore_mask / 255.0
    reshaped_crop_fore_gt = reshaped_crop_fore_gt / 255.0

    # extract Background
    extracted_background = background_img[h_index:h_index + nh, w_index:w_index + nw]

    reshaped_crop_fore_mask = np.expand_dims(reshaped_crop_fore_mask, 2)

    # unharmonized image compositing and placement ------------------------------------------
    composited = reshaped_crop_fore_mask * reshaped_crop_fore_un + (
            1 - reshaped_crop_fore_mask) * extracted_background

    final_compositing_unharmonized = background_img.copy()
    final_compositing_unharmonized[h_index:h_index + nh, w_index:w_index + nw] = composited

    # harmonized image compositing and placement ---------------------------------------------
    composited = reshaped_crop_fore_mask * reshaped_crop_fore_har + (
            1 - reshaped_crop_fore_mask) * extracted_background

    final_compositing_harmonized = background_img.copy()
    final_compositing_harmonized[h_index:h_index + nh, w_index:w_index + nw] = composited

    composited = reshaped_crop_fore_mask * reshaped_crop_fore_gt + (
            1 - reshaped_crop_fore_mask) * extracted_background

    final_compositing_gt = background_img.copy()
    final_compositing_gt[h_index:h_index + nh, w_index:w_index + nw] = composited

    # creating the new_foreground mask
    new_mask = np.zeros((h, w, 1))
    new_mask[h_index:h_index + nh, w_index:w_index + nw] = reshaped_crop_fore_mask

    return final_compositing_unharmonized, final_compositing_harmonized, final_compositing_gt, new_mask

def do_post_inference_placement(fore_unharmonized,
                                fore_harmonized,
                                fore_gt,
                                background_img,
                                foreground_mask,
                                indice):
    np_unharmonized = torch.squeeze(fore_unharmonized, 0).permute(1, 2, 0).cpu().numpy()
    np_harmonized = torch.squeeze(fore_harmonized, 0).permute(1, 2, 0).cpu().numpy()
    np_fore_gt = torch.squeeze(fore_gt, 0).permute(1, 2, 0).cpu().numpy()
    np_background = torch.squeeze(background_img, 0).cpu().permute(1, 2, 0).numpy()
    np_fore_mask = torch.squeeze(torch.squeeze(foreground_mask, 0), 0).cpu().numpy()  # .permute(1, 2, 0)

    np_comp_unharm, np_comp_harm, np_comp_gt, np_new_mask = do_image_placement(np_unharmonized,
                                                                               np_harmonized,
                                                                               np_fore_gt,
                                                                               np_background,
                                                                               np_fore_mask,
                                                                               indice)

    return np_comp_unharm, np_comp_harm, np_comp_gt, np_new_mask



# ------------------------- configuration -------------------------
config = load_config('./configs/config_inference.json')['config']
print(config)

devices = config['gpus']
os.environ["CUDA_VISIBLE_DEVICES"] = devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_examples_path = config['input_examples_path']
save_path = config['save_path']
create_or_recreate_folders_v2(save_path)
# ------------------------- configuration -------------------------

# ------------------------- load the weights of the net -------------------------
BackIllumEstNet = IlluminationNet(K=config['K'])
ShadingNet = ShadingNet(K=config['K'])
AlbedoNet = AlbedoNet()
RenderingNet = RenderingNet()

NeuRenFrm_weights_path = config['stage1_model_path']
NeuRenFrm_weights = torch.load(NeuRenFrm_weights_path)
ShadingNet.load_state_dict(NeuRenFrm_weights['ShadingNet'], strict=True)
AlbedoNet.load_state_dict(NeuRenFrm_weights['AlbedoNet'], strict=True)
RenderingNet.load_state_dict(NeuRenFrm_weights['RenderingNet'], strict=True)

AlbedoNet = torch.nn.DataParallel(AlbedoNet)
AlbedoNet.to(device)

ShadingNet = torch.nn.DataParallel(ShadingNet)
ShadingNet.to(device)

RenderingNet = torch.nn.DataParallel(RenderingNet)
RenderingNet.to(device)

BackIllumEstNet = torch.nn.DataParallel(BackIllumEstNet)
BackIllumEstNet.to(device)

backillumnet_weights_path = config['stage2_model_path']
backillumnet_weights = torch.load(backillumnet_weights_path)
BackIllumEstNet.load_state_dict(backillumnet_weights, strict=True)

RenderingNet_params = sum(param.numel() for param in RenderingNet.parameters())
print('# RenderingNet parameters:', RenderingNet_params)
total_params = RenderingNet_params
AlbedoNet_params = sum(param.numel() for param in AlbedoNet.parameters())
print('# AlbedoNet parameters:', AlbedoNet_params)
total_params += AlbedoNet_params
ShadingNet_params = sum(param.numel() for param in ShadingNet.parameters())
print('# ShadingNet parameters:', ShadingNet_params)
total_params += ShadingNet_params
BackIllumEstNet_params = sum(param.numel() for param in BackIllumEstNet.parameters())
print('# BackIllumEstNet parameters:', BackIllumEstNet_params)
total_params += BackIllumEstNet_params
print('# Total parameters:', total_params)

print('All network weights loaded!')
# ------------------------- load the weights of the net -------------------------

if __name__ == '__main__':
    examples = os.listdir(input_examples_path)
    for i, example in enumerate(examples):
        # load the input examples
        dic = load_data(osp.join(input_examples_path, example))

        depth_map = dic['depth']
        normal_map = dic['normal']
        foreground_mask = dic['fore_mask']
        color_img_gt = dic['gt_color_image']
        unharmonized_img = dic['unharmonized_input']
        background_image = dic['background_image']
        indice = dic['indice']

        if torch.cuda.is_available():
            depth_map = depth_map.to(device)
            normal_map = normal_map.to(device)
            foreground_mask = foreground_mask.to(device)
            unharmonized_img = unharmonized_img.to(device)
            background_image = background_image.to(device)

        print(f'[{i+1}|{len(examples)}] processing!')

        # inference
        with torch.no_grad():
            depth_nd_normal = torch.cat([depth_map, normal_map], 1)
            illum_descriptor = BackIllumEstNet(background_image)

            shading_bases = ShadingNet(depth_nd_normal)
            shading = render_shading(shading_bases, illum_descriptor)

            _, albedo_feats = AlbedoNet(unharmonized_img)

            rendering_input = torch.cat([albedo_feats, shading, unharmonized_img], 1)
            harmonized_img = RenderingNet(rendering_input)

        # do compositing of the network output
        final_comp_unharmonized, final_comp_harmonized, final_comp_gt, new_mask = do_post_inference_placement(
            unharmonized_img,
            harmonized_img,
            color_img_gt,
            background_image,
            foreground_mask,
            indice)

        # reconvert the numpy back to torch tensor
        f_comp_un_torch = torch.unsqueeze(torch.from_numpy(final_comp_unharmonized).permute(2, 0, 1), 0).to(device)
        f_comp_ha_torch = torch.unsqueeze(torch.from_numpy(final_comp_harmonized).permute(2, 0, 1), 0).to(device)
        f_comp_gt_torch = torch.unsqueeze(torch.from_numpy(final_comp_gt).permute(2, 0, 1), 0).to(device)

        # save results
        display_data = torch.cat([f_comp_un_torch, f_comp_ha_torch, f_comp_gt_torch], dim=0)
        utils.save_image(display_data,
                         f"{save_path}/{example}.png",
                         nrow=f_comp_gt_torch.shape[0], padding=2, normalize=False)

    print('Done!')
            
