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
from pytorch_ssim import ssim as ssim_metric
from data.dataset import HarmonyDatasetV2 as HarmonyDataset
from model.illumination_model import IlluminationNet
from model.shading_model import ShadingNet
from utils import *

# ------------------------- configuration -------------------------
config = load_config('./configs/config_stage2.json')['config']
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

NeuRenFrm_weights_path = config['NeuRenFrm_weights_path']
# ------------------------- configuration -------------------------


# ------------------------- dataset -------------------------
dataset = HarmonyDataset(TRAIN_DATA_PATH, resize=config['resize'], hflip=config['training']['hflip'])
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

IllumEncNet = IlluminationNet(K=config['K'])
ShadingNet = ShadingNet(K=config['K'])

# load the weights of the net
NeuRenFrm_weights = torch.load(NeuRenFrm_weights_path)
IllumEncNet.load_state_dict(NeuRenFrm_weights['IllumEncNet'], strict=True)
ShadingNet.load_state_dict(NeuRenFrm_weights['ShadingNet'], strict=True)

IllumEncNet = nn.DataParallel(IllumEncNet)
IllumEncNet.to(device)

ShadingNet = nn.DataParallel(ShadingNet)
ShadingNet.to(device)

BackIllumEstNet = IlluminationNet(K=config['K'])
print('# BackIllumEstNet parameters:', sum(param.numel() for param in BackIllumEstNet.parameters()))
opt = optim.Adam(BackIllumEstNet.parameters(), lr=lr)

BackIllumEstNet = nn.DataParallel(BackIllumEstNet)
BackIllumEstNet.to(device)

IllumEncNet = IllumEncNet.eval()
# make all weights in the net non trainable
for p in IllumEncNet.parameters():
    p.requires_grad = False

ShadingNet = ShadingNet.eval()
for p in ShadingNet.parameters():
    p.requires_grad = False
# ------------------------- network setup -------------------------

# verify whether we want to continue with a training or start brand-new
if config['training']['continue']:
    # load weights
    print('------------------- Continue Training -------------------')
    weight = torch.load(f"{config['epoch_folder']}/ckpt_backillumnet{config['training']['epoch']}.pth")
    BackIllumEstNet.load_state_dict(weight, strict=True)

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
l1_loss_mean = nn.L1Loss(reduction='mean')
l1_loss = nn.L1Loss(reduction='none')
# # ------------------------- loss functions setup -------------------------

if __name__ == '__main__':
    for epoch in range(1 + epoch, NUM_EPOCHS + 1):

        iteration = 0
        l1_loss_total = 0.0
        shading_loss_total = 0.0
        BackIllumEstNet = BackIllumEstNet.train()

        for tensors_dic in train_loader:

            iteration = iteration + 1

            background_image = tensors_dic['background_image']
            hdr_image = tensors_dic['hdr']
            depth_map = tensors_dic['depth']
            normal_map = tensors_dic['normal']
            fore_mask = tensors_dic['fore_mask']


            if torch.cuda.is_available():
                background_image = background_image.to(device)
                hdr_image = hdr_image.to(device)
                depth_map = depth_map.to(device)
                normal_map = normal_map.to(device)
                fore_mask = fore_mask.to(device)

            BackIllumEstNet.zero_grad()

            # network forward pass
            pred_illum_descriptor = BackIllumEstNet(background_image)
            illum_descriptor = IllumEncNet(hdr_image)

            input_tensor = torch.cat([depth_map, normal_map], 1)
            shading_bases = ShadingNet(input_tensor)
            pred_shading = render_shading(shading_bases, pred_illum_descriptor)
            shading = render_shading(shading_bases, illum_descriptor)

            # computing the loss function
            loss = l1_loss_mean(pred_illum_descriptor, illum_descriptor)
            shading_loss = mask_l1(pred_shading, shading, fore_mask, l1_loss)

            final_loss = loss + shading_loss
            final_loss.backward()
            opt.step()

            # console print for the training in progress
            print('[%d/%d][%d], L1 : %f' % (
                epoch, NUM_EPOCHS, iteration,
                loss.item()))

            # display
            if iteration % display_iter == 0:

                with torch.no_grad():
                    shading_bases = ShadingNet(input_tensor)


                pred_shading = render_shading(shading_bases, pred_illum_descriptor)
                shading = render_shading(shading_bases, illum_descriptor)

                background_image = TF.resize(background_image, [pred_shading.shape[2], pred_shading.shape[3]])
                ldr_pano = TF.resize(do_tone_map(hdr_image, None), (pred_shading.shape[2], pred_shading.shape[3]))

                display_data = torch.cat([background_image, pred_shading, shading, ldr_pano], dim=0)
                utils.save_image(display_data,
                                 display_folder + "/Epoch_%d Iter_%d.jpg" % (epoch, iteration),
                                 nrow=hdr_image.shape[0], padding=2, normalize=False)

            # loss functions summary
            l1_loss_total += loss.item()
            shading_loss_total += shading_loss.item()

            if iteration % record_train_iter_loss == 0:
                writer.add_scalar('TrainIter/MAE', l1_loss_total / iteration, count_train_iter_loss)
                writer.add_scalar('TrainIter/MAE_Shadingloss', shading_loss_total / iteration, count_train_iter_loss)
                count_train_iter_loss += 1
                
        # one epoch finished, output training loss, save model
        writer.add_scalar('Train/MAE', l1_loss_total / iteration, epoch)
        writer.add_scalar('Train/MAE_Shadingloss', shading_loss_total / iteration, epoch)
        torch.save(BackIllumEstNet.state_dict(), epoch_folder + '/ckpt_backillumnet%d.pth' % epoch)

        # ------------------------------------------- Test image ----------------------------------------
        test_iteration = 0
        test_l1_loss = 0
        test_shading_mae_loss = 0
        BackIllumEstNet = BackIllumEstNet.eval()

        for tensors_dic in test_loader:

            test_iteration = test_iteration + 1

            background_image = tensors_dic['background_image']
            hdr_image = tensors_dic['hdr']
            depth_map = tensors_dic['depth']
            normal_map = tensors_dic['normal']
            fore_mask = tensors_dic['fore_mask']

            if torch.cuda.is_available():
                background_image = background_image.to(device)
                hdr_image = hdr_image.to(device)
                depth_map = depth_map.to(device)
                normal_map = normal_map.to(device)
                fore_mask = fore_mask.to(device)

            with torch.no_grad():
                input_tensor = torch.cat([depth_map, normal_map], 1)

                pred_illum_descriptor = BackIllumEstNet(background_image)
                illum_descriptor = IllumEncNet(hdr_image)

                shading_bases = ShadingNet(input_tensor)
                pred_shading = render_shading(shading_bases, pred_illum_descriptor)
                shading = render_shading(shading_bases, illum_descriptor)

            mae_loss = l1_loss_mean(pred_illum_descriptor, illum_descriptor)
            shading_loss = mask_l1(pred_shading, shading, fore_mask, l1_loss)

            test_l1_loss += mae_loss.item()
            test_shading_mae_loss += shading_loss.item()

            print(f'Test [{test_iteration}|{len(test_dataset)}]')

            if epoch % 1 == 0:
                background_image = TF.resize(background_image, [pred_shading.shape[2], pred_shading.shape[3]])
                ldr_pano = TF.resize(do_tone_map(hdr_image, None), (pred_shading.shape[2], pred_shading.shape[3]))

                display_data = torch.cat([background_image, pred_shading, shading, ldr_pano], dim=0)
                utils.save_image(display_data,
                                 display_validation + "/Test_Iter_%d.jpg" % test_iteration,
                                 nrow=hdr_image.shape[0], padding=2, normalize=False)

        writer.add_scalar('Test/MAE', test_l1_loss / test_iteration, epoch)
        writer.add_scalar('Test/MAE_Shadingloss', test_shading_mae_loss / test_iteration, epoch)
