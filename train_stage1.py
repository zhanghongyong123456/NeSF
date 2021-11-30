import os
import torch
import shutil
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
from data.dataset import HarmonyDatasetV2 as HarmonyDataset
from model.neural_rendering_framework import NeuRenFrm as NeuRenNet
from utils import *


# ------------------------- configuration -------------------------
config = load_config('./configs/config_stage1.json')['config']
print(config)

devices = config['gpus']
os.environ["CUDA_VISIBLE_DEVICES"] = devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

writer = SummaryWriter(config['summary'])
display_folder = config['display_folder']
display_validation = config['display_val']
epoch_folder = config['epoch_folder']
train_mode = config['training']['mode']
NUM_EPOCHS = config['training']['epochs']
display_iter = config['training']['display']
TRAIN_DATA_PATH = config['train_data_path']
TEST_DATA_PATH = config['test_data_path']
lr = config['training']['lr']
display_shading = config['training']['display_iter_light_dir']
record_train_iter_loss = config['training']['record_train_iter_loss']
# ------------------------- configuration -------------------------


# ------------------------- dataset -------------------------
dataset = HarmonyDataset(TRAIN_DATA_PATH, resize=config['resize'])
test_dataset = HarmonyDataset(TEST_DATA_PATH, resize=config['resize'])

print(f'Dataset Train: {len(dataset)}')
print(f'Dataset Test : {len(test_dataset)}')

train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           num_workers=config['data_workers'],
                                           batch_size=config['train_batch'],
                                           shuffle=config['train_shuffle'])

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          num_workers=config['data_workers'],
                                          batch_size=config['val_batch'],
                                          shuffle=config['val_shuffle'])
# ------------------------- dataset -------------------------


# ------------------------- network setup -------------------------

NeuRenNet = NeuRenNet(K=config['K'])
print('# Neural Rendering Framework parameters:', sum(param.numel() for param in NeuRenNet.parameters()))
opt = optim.Adam(NeuRenNet.parameters(), lr=lr)

NeuRenNet = nn.DataParallel(NeuRenNet)
NeuRenNet.to(device)
# ------------------------- network setup -------------------------

# verify whether we want to continue with a training or start brand-new
if config['training']['continue']:
    # load weights
    print('------------------- Continue Training -------------------')
    weight = torch.load(f"{config['epoch_folder']}/ckpt{config['training']['epoch']}.pth")
    NeuRenNet.load_state_dict(weight)

    # set current epoch
    epoch = config['training']['epoch']
    count_train_iter_loss = config['training']['count_train_iter_loss']
else:
    print('------------------- Starting Training -------------------')
    create_or_recreate_folders(config)
    writer = SummaryWriter(config['summary'])
    epoch = 0
    count_train_iter_loss = 1

# # ------------------------- loss functions setup -------------------------
l1_loss = nn.L1Loss(reduction='none')
# # ------------------------- loss functions setup -------------------------

if __name__ == '__main__':
    for epoch in range(1 + epoch, NUM_EPOCHS + 1):

        iteration = 0
        total_mae_loss_shading = 0.0
        total_ssim_loss_shading = 0.0
        total_mae_loss_albedo = 0.0
        total_ssim_loss_albedo = 0.0
        total_mae_loss_image = 0.0
        total_ssim_loss_image = 0.0

        NeuRenNet = NeuRenNet.train()

        for tensors_dic in train_loader:

            iteration = iteration + 1

            background_image = tensors_dic['background_image']
            hdr_image = tensors_dic['hdr']
            depth_map = tensors_dic['depth']
            normal_map = tensors_dic['normal']
            shading_gt = tensors_dic['shading_gt']
            unharmonized_image = tensors_dic['unharmonized_input']
            gt_color_image = tensors_dic['gt_color_image']
            gt_albedo = tensors_dic['gt_albedo']
            foreground_mask = tensors_dic['fore_mask']

            if torch.cuda.is_available():
                background_image = background_image.to(device)
                hdr_image = hdr_image.to(device)
                depth_map = depth_map.to(device)
                normal_map = normal_map.to(device)
                shading_gt = shading_gt.to(device)
                unharmonized_image = unharmonized_image.to(device)
                gt_color_image = gt_color_image.to(device)
                gt_albedo = gt_albedo.to(device)
                foreground_mask = foreground_mask.to(device)

            NeuRenNet.zero_grad()

            shading_image, albedo_image, harmonized_img = NeuRenNet(depth_map, normal_map, hdr_image, unharmonized_image)

            # computing the loss function
            shading_loss = mask_l1(shading_image, shading_gt, foreground_mask, l1_loss)
            shading_ssim_loss = mask_ssim(shading_image, shading_gt, foreground_mask)
            albedo_loss = mask_l1(albedo_image, gt_albedo, foreground_mask, l1_loss)
            albedo_ssim_loss = mask_ssim(albedo_image, gt_albedo, foreground_mask)
            image_loss = mask_l1(harmonized_img, gt_color_image, foreground_mask, l1_loss)
            image_ssim_loss = mask_ssim(harmonized_img, gt_color_image, foreground_mask)

            final_loss = shading_loss + albedo_loss + image_loss + (1 - shading_ssim_loss) + (1 - albedo_ssim_loss) + (1 - image_ssim_loss)
            final_loss.backward()
            opt.step()

            # console print for the training in progress
            print('[%d/%d][%d], L_shading : %f, L_albedo : %f, L_image : %f' % (
                epoch, NUM_EPOCHS, iteration,
                shading_loss.item(), albedo_loss.item(), image_loss.item()))

            # display
            if iteration % display_iter == 0:
                resized_hdr = TF.resize(do_tone_map(hdr_image, None), (shading_image.shape[2], shading_image.shape[3]))

                display_data = torch.cat([shading_image,
                        shading_gt,
                        albedo_image,
                        gt_albedo,
                        harmonized_img, gt_color_image, resized_hdr], dim=0)

                utils.save_image(display_data,
                                 display_folder + "/Epoch_%d Iter_%d.jpg" % (epoch, iteration),
                                 nrow=hdr_image.shape[0], padding=2, normalize=False)


            # loss functions summary
            total_mae_loss_shading += shading_loss.item()
            total_ssim_loss_shading += shading_ssim_loss.item()
            total_mae_loss_albedo += albedo_loss.item()
            total_ssim_loss_albedo += albedo_ssim_loss.item()
            total_mae_loss_image += image_loss.item()
            total_ssim_loss_image += image_ssim_loss.item()

            if iteration % record_train_iter_loss == 0:
                writer.add_scalar('TrainIter/Shading_MAE', total_mae_loss_shading / iteration, count_train_iter_loss)
                writer.add_scalar('TrainIter/Shading_SSIM', total_ssim_loss_shading / iteration, count_train_iter_loss)
                writer.add_scalar('TrainIter/Albedo_MAE', total_mae_loss_albedo / iteration, count_train_iter_loss)
                writer.add_scalar('TrainIter/Albedo_SSIM', total_ssim_loss_albedo / iteration, count_train_iter_loss)
                writer.add_scalar('TrainIter/Image_MAE', total_mae_loss_image / iteration, count_train_iter_loss)
                writer.add_scalar('TrainIter/Image_SSIM', total_ssim_loss_image / iteration, count_train_iter_loss)
                count_train_iter_loss += 1

        # one epoch finished, output training loss, save model
        writer.add_scalar('Train/Shading_MAE', total_mae_loss_shading / iteration, epoch)
        writer.add_scalar('Train/Shading_SSIM', total_ssim_loss_shading / iteration, epoch)
        writer.add_scalar('Train/Albedo_MAE', total_mae_loss_albedo / iteration, epoch)
        writer.add_scalar('Train/Albedo_SSIM', total_ssim_loss_albedo / iteration, epoch)
        writer.add_scalar('Train/Image_MAE', total_mae_loss_image / iteration, epoch)
        writer.add_scalar('Train/Image_SSIM', total_ssim_loss_image / iteration, epoch)

        dico = {
            'IllumEncNet': NeuRenNet.module.IllumEncNet.state_dict(),
            'ShadingNet': NeuRenNet.module.ShadingNet.state_dict(),
            'AlbedoNet': NeuRenNet.module.AlbedoNet.state_dict(),
            'RenderingNet': NeuRenNet.module.RenderingNet.state_dict()
        }
        torch.save(dico, epoch_folder + '/ckpt_neurennet%d.pth' % epoch)

        # ------------------------------------------- Test image ----------------------------------------
        test_iteration = 0

        total_mae_loss_shading_test = 0.0
        total_ssim_loss_shading_test = 0.0
        total_mae_loss_albedo_test = 0.0
        total_ssim_loss_albedo_test = 0.0
        total_mae_loss_image_test = 0.0
        total_ssim_loss_image_test = 0.0

        NeuRenNet = NeuRenNet.eval()

        for tensors_dic in test_loader:

            test_iteration = test_iteration + 1

            background_image = tensors_dic['background_image']
            hdr_image = tensors_dic['hdr']
            depth_map = tensors_dic['depth']
            normal_map = tensors_dic['normal']
            shading_gt = tensors_dic['shading_gt']
            unharmonized_image = tensors_dic['unharmonized_input']
            gt_color_image = tensors_dic['gt_color_image']
            gt_albedo = tensors_dic['gt_albedo']
            foreground_mask = tensors_dic['fore_mask']

            if torch.cuda.is_available():
                background_image = background_image.to(device)
                hdr_image = hdr_image.to(device)
                depth_map = depth_map.to(device)
                normal_map = normal_map.to(device)
                shading_gt = shading_gt.to(device)
                unharmonized_image = unharmonized_image.to(device)
                gt_color_image = gt_color_image.to(device)
                gt_albedo = gt_albedo.to(device)
                foreground_mask = foreground_mask.to(device)

            with torch.no_grad():
                shading_image, albedo_image, harmonized_img = NeuRenNet(depth_map, normal_map, hdr_image, unharmonized_image)

            # computing the loss function
            mae_loss_shading = mask_l1(shading_image, shading_gt, foreground_mask, l1_loss)
            ssim_loss_shading = mask_ssim(shading_image, shading_gt, foreground_mask)
            mae_loss_albedo = mask_l1(albedo_image, gt_albedo, foreground_mask, l1_loss)
            ssim_loss_albedo = mask_ssim(albedo_image, gt_albedo, foreground_mask)
            mae_loss_image = mask_l1(harmonized_img, gt_color_image, foreground_mask, l1_loss)
            ssim_loss_image = mask_ssim(harmonized_img, gt_color_image, foreground_mask)

            total_mae_loss_shading_test += mae_loss_shading.item()
            total_ssim_loss_shading_test += ssim_loss_shading.item()
            total_mae_loss_albedo_test += mae_loss_albedo.item()
            total_ssim_loss_albedo_test += ssim_loss_albedo.item()
            total_mae_loss_image_test += mae_loss_image.item()
            total_ssim_loss_image_test += ssim_loss_image.item()

            print(f'Test [{test_iteration}|{len(test_dataset)}]')

            if epoch % 1 == 0:
                resized_hdr = TF.resize(do_tone_map(hdr_image, None), (shading_image.shape[2], shading_image.shape[3]))

                display_data = torch.cat([shading_image,
                        shading_gt,
                        albedo_image,
                        gt_albedo,
                        harmonized_img, gt_color_image, resized_hdr], dim=0)

                utils.save_image(display_data,
                                display_validation + "/Epoch_%d Iter_%d.jpg" % (epoch, test_iteration),
                                nrow=hdr_image.shape[0], padding=2, normalize=False)

        writer.add_scalar('Test/Shading_MAE', total_mae_loss_shading_test / test_iteration, epoch)
        writer.add_scalar('Test/Shading_SSIM', total_ssim_loss_shading_test / test_iteration, epoch)
        writer.add_scalar('Test/Albedo_MAE', total_mae_loss_albedo_test / test_iteration, epoch)
        writer.add_scalar('Test/Albedo_SSIM', total_ssim_loss_albedo_test / test_iteration, epoch)
        writer.add_scalar('Test/Image_MAE', total_mae_loss_image_test / test_iteration, epoch)
        writer.add_scalar('Test/Image_SSIM', total_ssim_loss_image_test / test_iteration, epoch)
