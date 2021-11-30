import torch
import cv2
# import pyexr
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
# import imageio
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import shutil
import os
import glob
# import imageio
import random


class HarmonyDatasetV2(Dataset):

    def __init__(self, data_dir, filter=True, mode='train', random_flipping=False, random_rotation=False, resize=False,
                 add_noise=False, illum_labels=['sunny', 'sunrise-sunset', 'cloudy', 'night']):
        super(HarmonyDatasetV2, self).__init__()
        self.random_flipping = random_flipping
        self.random_rotation = random_rotation
        self.add_noise = add_noise
        self.resize = resize

        self.data_dir = data_dir
        self.background_images_dir = os.path.join(self.data_dir, 'backgrounds')
        self.hdr_dir = os.path.join(self.data_dir, 'hdr_images')
        self.objects_dir = os.path.join(self.data_dir, 'objects')
        self.placement_masks = os.path.join(self.data_dir, 'binary_masks')

        # ----------------------------------------------- hdr

        self.hdr_images_list = glob.glob(os.path.join(self.hdr_dir, '*.exr'))

        # filter the existing hdr images
        # if filter:
        #     self.hdr_images_list = self.do_hdr_filtering(self.hdr_dir, self.gt_shadings_dir)

        # ------------------------------------------------ shading imgs
        self.shading_gt_images_paths = self.get_all_gt(self.objects_dir)
        self.images_gt_images_paths = self.get_all_gt(self.objects_dir, type='images')

        self.shading_gt_images_paths.sort()
        self.images_gt_images_paths.sort()

        self.current_illum_label = self.data_dir.split('/')[-1]
        
        self.all_images_gt_images_paths = []
        for illum_label in illum_labels:
            self.all_images_gt_images_paths += self.get_all_gt(self.objects_dir.replace(self.current_illum_label, illum_label), type='images')
        self.all_images_gt_images_paths.sort()

    def get_all_background_images(self, folder):

        folders = os.listdir(folder)
        folders.sort()

        # get all images per folder
        background_paths = []
        for i in range(len(folders)):
            current_folder = folders[i]
            current_path = os.path.join(folder, current_folder)
            current_images_paths = glob.glob(os.path.join(current_path, '*.png'))

            background_paths += current_images_paths

        return background_paths

    def do_hdr_filtering(self, hdr_images_path, generated_images_path):

        generated_images_list = os.listdir(generated_images_path)

        new_list = []
        for i in range(len(generated_images_list)):
            current_hdr = generated_images_list[i]
            current_path = glob.glob(os.path.join(hdr_images_path, f'{current_hdr}.exr'))
            new_list.append(current_path[0])

        return new_list

    def _hdr_read(self, path):
        hdr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        hdr = np.flip(hdr, -1)
        hdr = np.clip(hdr, 0, None)
        return hdr

    def load_hdr(self, path):
        hdr = cv2.imread(path, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR)
        hdr = np.flip(hdr, -1)
        hdr = np.clip(hdr, 0, None)
        return hdr

    def _hdr_resize(self, img, h, w):
        img = cv2.resize(img, (w, h), cv2.INTER_AREA)
        return img

    def load_npy(self, npy_path):
        return np.load(npy_path)

    def _hdr_read_resize(self, path, is_training=False):
        hdr = self._hdr_read(path)
        h, w, _, = hdr.shape
        ratio = max(512 / h, 512 / w)
        h = round(h * ratio)
        w = round(w * ratio)
        hdr = self._hdr_resize(hdr, h, w)

        return hdr

    def get_binary_mask(self, indexed_background):
        
        th = indexed_background.split('/')[-1].split('_')[-3]
        ph = indexed_background.split('/')[-1].split('_')[-2]
        angle = indexed_background.split('/')[-1].split('_')[-1].split('.')[0]
        name = indexed_background.split('/')[-2]

        path_ = os.path.join(self.placement_masks, name)
        path = glob.glob(os.path.join(path_, f'background_*_{th}_{ph}_{angle}.png'))

        if len(path) == 0:
            print(f'------------------------------------- {path_}')

        return path[0]

    def get_indice(self, h, indices, w):
        if len(indices) == 0:
            indice = ((random.randint(0, (h // 2) - 1)), random.randint(0, w - 1, ))
        else:
            indice = indices[random.randint(0, len(indices) - 1)]
        return indice

    def __len__(self):
        return len(self.images_gt_images_paths)

    def __getitem__(self, index):

        # obtain the current path and object rotation angle,
        # illumination rotation angle and hdr illumination name
        indexed_shading = self.shading_gt_images_paths[index]
        indexed_image = self.images_gt_images_paths[index]

        indexed_background, binary_index = self.get_corresponding_background(indexed_image)
        indexed_mask = self.get_binary_mask(binary_index)

        indexed_object_name = indexed_shading.split('/')[-4]
        indexed_obj_angle = indexed_shading.split('/')[-1].split('.')[0].split('_')[1] #image_0_180.png 180
        hdr_name = indexed_shading.split('/')[-2]
        illumination_rotation = int(indexed_shading.split('/')[-1].split('.')[0].split('_')[2])

        gt_shadings_dir = os.path.join(self.objects_dir, os.path.join(indexed_object_name, 'shading'))
        depth_dir = os.path.join(self.objects_dir, os.path.join(indexed_object_name, 'depth'))
        normal_dir = os.path.join(self.objects_dir, os.path.join(indexed_object_name, 'normals'))
        mask_fore_dir = os.path.join(self.objects_dir, os.path.join(indexed_object_name, 'mask_foreground'))
        gt_albedo_dir = os.path.join(self.objects_dir, os.path.join(indexed_object_name, 'albedo'))

        current_unharmonized_path = self.get_unharmonized_all_input(indexed_image, self.all_images_gt_images_paths)
        current_depth_path = glob.glob(os.path.join(depth_dir, f'*_{indexed_obj_angle}.npy'))[0]
        current_normal_path = glob.glob(os.path.join(normal_dir, f'*_{indexed_obj_angle}.npy'))[0]
        current_mask_fore_path = glob.glob(os.path.join(mask_fore_dir, f'*_{indexed_obj_angle}.png'))[0]
        current_albedo_path = glob.glob(os.path.join(gt_albedo_dir, f'*_{indexed_obj_angle}.png'))[0]

        # read images
        depth_map = self.load_npy(current_depth_path).astype(np.float32)
        normal_map = self.load_npy(current_normal_path).astype(np.float32)
        placement_mask = Image.open(indexed_mask)
        unharmonized_image = Image.open(current_unharmonized_path)
        foreground_mask = Image.open(current_mask_fore_path)
        color_image = Image.open(indexed_image)
        albedo_image = Image.open(current_albedo_path)
        shading_image = Image.open(indexed_shading)
        background_image = Image.open(indexed_background)

        if self.resize:
            W, H = foreground_mask.size
            foreground_mask = foreground_mask.resize((W//self.resize,H//self.resize))
            color_image = color_image.resize((W//self.resize,H//self.resize))
            albedo_image = albedo_image.resize((W//self.resize,H//self.resize))
            shading_image = shading_image.resize((W//self.resize,H//self.resize))
            background_image = background_image.resize((W//self.resize,H//self.resize))
            unharmonized_image = unharmonized_image.resize((W//self.resize,H//self.resize))
            placement_mask = placement_mask.resize((W//self.resize,H//self.resize))
            depth_map = depth_map[...,::self.resize,::self.resize]
            normal_map = normal_map[...,::self.resize,::self.resize]

        if self.random_flipping:

            # Random horizontal flipping
            if random.random() > 0.5:
                foreground_mask = TF.hflip(foreground_mask)
                color_image = TF.hflip(color_image)
                albedo_image = TF.hflip(albedo_image)
                shading_image = TF.hflip(shading_image)

            # Random vertical flipping
            if random.random() > 0.5:
                foreground_mask = TF.vflip(foreground_mask)
                color_image = TF.vflip(color_image)
                albedo_image = TF.vflip(albedo_image)
                shading_image = TF.hflip(shading_image)

        if self.random_rotation:
            angle = random.randint(-180, 180)
            foreground_mask = TF.rotate(foreground_mask, angle)
            color_image = TF.rotate(color_image, angle)
            albedo_image = TF.rotate(albedo_image, angle)
            shading_image = TF.rotate(shading_image, angle)

        # if self.add_noise:
        #    color_image = TF.gaussian_blur(color_image, kernel_size=(5, 9), sigma=(0.1, 5))

        shading_image = np.array(shading_image, dtype=np.float32)[:, :, :3] / 255.0
        foreground_mask = np.array(foreground_mask, dtype=np.float32)[:, :, 0] / 255.0
        color_image = np.array(color_image, dtype=np.float32)[:, :, :3] / 255.0
        albedo_image = np.array(albedo_image, dtype=np.float32)[:, :, :3] / 255.0
        background_image = np.array(background_image, dtype=np.float32)[:, :, :3] / 255.0
        unharmonized_image = np.array(unharmonized_image, dtype=np.float32)[:, :, :3] / 255.0
        placement_mask = np.array(placement_mask).astype(np.float32)[:, :] / 255.0

        foreground_mask[foreground_mask < 0.5] = 0.0
        foreground_mask[foreground_mask > 0.5] = 1.0


        # masking the depth map and the normal map
        masked_depth = foreground_mask * depth_map
        masked_normal = foreground_mask * normal_map
        masked_shading = np.expand_dims(foreground_mask, 2) * shading_image
        masked_albedo = np.expand_dims(foreground_mask, 2) * albedo_image
        masked_color = np.expand_dims(foreground_mask, 2) * color_image
        masked_unharmonized = np.expand_dims(foreground_mask, 2) * unharmonized_image

        minval = np.min(masked_depth[np.nonzero(masked_depth)])
        maxval = np.max(masked_depth[np.nonzero(masked_depth)])
        masked_depth = (masked_depth - minval) / (maxval - minval)

        # based on the shading gt illumination name pick the corresponding hdr image
        hdr_path = glob.glob(os.path.join(os.path.join(self.hdr_dir, f'{hdr_name}*')))[0]
        hdr_image = self.load_hdr(hdr_path)


        # rotation used to set the hdr image so that it corresponds with blender
        hdr_image = rotate_hdr(hdr_image, 180)
        unrotated = hdr_image.copy()

        # applying the rotation that corresponds to the shading image
        rotated_hdr = rotate_hdr(hdr_image, illumination_rotation)
        original_hdr = rotated_hdr.copy()

        # convert all relevant tensors into pytorch tensors
        masked_depth_tensor = torch.unsqueeze(torch.from_numpy(masked_depth), 0)
        masked_normal_tensor = torch.from_numpy(masked_normal)
        fore_mask_tensor = torch.unsqueeze(torch.from_numpy(foreground_mask), 0)
        shading_gt_tensor = torch.from_numpy(masked_shading).permute(2, 0, 1)
        albedo_gt_tensor = torch.from_numpy(masked_albedo).permute(2, 0, 1)
        color_gt_tensor = torch.from_numpy(masked_color).permute(2, 0, 1)
        background_tensor = torch.from_numpy(background_image).permute(2, 0, 1)
        unharmonized_tensor = torch.from_numpy(masked_unharmonized).permute(2, 0, 1)
        placement_tensor = torch.from_numpy(placement_mask)

        hdr_tensor = torch.from_numpy(rotated_hdr).permute(2, 0, 1)
        unrotated_tensor = torch.from_numpy(unrotated).permute(2, 0, 1)

        # ------------------------------------------------------------------------------------------------------------


        return {
            'depth': masked_depth_tensor,
            'normal': masked_normal_tensor,
            'shading_gt': shading_gt_tensor,
            'fore_mask': fore_mask_tensor,
            'back_mask': 0,
            'hdr': hdr_tensor,
            'ohdr': 0,
            'unrotated': 0,
            'gt_albedo': albedo_gt_tensor,
            'gt_color_image': color_gt_tensor,
            'add_hdr': 0,
            'unharmonized_input': unharmonized_tensor,
            'add_shad': 0,
            'random_angles': 0,
            'background_image': background_tensor,
            'placement_mask': placement_tensor
        }

    def get_corresponding_background(self, indexed_image):

        image_name = indexed_image.split('/')[-1]
        if self.current_illum_label == 'sunny':
            scene_name = indexed_image.split('/')[-2]
            scene_name_binary = indexed_image.split('/')[-2]
        else:
            scene_name = indexed_image.split('/')[-2][:-2] + '8k'
            scene_name_binary = indexed_image.split('/')[-2][:-2] + '2k'

        hdr_theta = image_name.split('_')[2].split('.')[0]

        background_folder = os.path.join(self.background_images_dir, scene_name)
        background_folder_binary = os.path.join(self.background_images_dir, scene_name_binary)

        background_images = glob.glob(os.path.join(background_folder, f'background_*_{hdr_theta}_*_*.png'))
        background_images.sort()
        random_background = background_images[random.randint(0, len(background_images) - 1)]
        random_background_binary = os.path.join(background_folder_binary, random_background.split('/')[-1])

        return random_background, random_background_binary

    def get_all_gt(self, object_directory, type='shading'):

        images_paths = []
        objects = os.listdir(object_directory)

        # iterate over all the objects
        for o in range(len(objects)):
            current_object = objects[o]
            current_object_path = os.path.join(object_directory, current_object)
            current_object_shading_dir_path = os.path.join(current_object_path, type)

            # iterate over all the folders in the shading dir
            shading_dir = os.listdir(current_object_shading_dir_path)
            for s in range(len(shading_dir)):
                current_shading_dir = shading_dir[s]
                current_shading_dir_path = os.path.join(current_object_shading_dir_path, current_shading_dir)
                current_shading_dir_images_paths = glob.glob(os.path.join(current_shading_dir_path, '*.png'))

                images_paths += current_shading_dir_images_paths

        return images_paths



    def get_angle(self, angle, illumination_rotation):
        if (angle + illumination_rotation) == illumination_rotation:
            return 0
        elif (angle + illumination_rotation) > 360:
            return (angle + illumination_rotation) - 360
        elif (angle + illumination_rotation) == 360:
            return 0
        else:
            return angle + illumination_rotation



    def get_unharmonized_input(self, indexed_image, images_gt_images_paths):

        # remove the indexed image ground truth from the list of all images
        new_paths = images_gt_images_paths.copy()
        new_paths.remove(indexed_image)

        # filter all the list such that only images that have the same object angle as that
        # of the ground truth image are retained
        unhamonized_inputs = []
        gt_angle = indexed_image.split('/')[-1].split('.')[0].split('_')[1]
        for i in range(len(new_paths)):
            current_paths = new_paths[i]
            current_angle = current_paths.split('/')[-1].split('.')[0].split('_')[1]

            if current_angle == gt_angle:
                unhamonized_inputs.append(current_paths)

        # randomly select an image from the filtered list and return it
        return unhamonized_inputs[random.randint(0, len(unhamonized_inputs) - 1)]


    def get_unharmonized_input2(self, indexed_image, images_gt_images_paths):

        # remove the indexed image ground truth from the list of all images
        new_paths = images_gt_images_paths.copy()
        new_paths.remove(indexed_image)

        # filter all the list such that only images that have the same object angle as that
        # of the ground truth image are retained
        unhamonized_inputs = []
        gt_angle = indexed_image.split('/')[-1].split('.')[0].split('_')[1]
        gt_name = indexed_image.split('/')[-4]

        for i in range(len(new_paths)):
            current_paths = new_paths[i]
            current_angle = current_paths.split('/')[-1].split('.')[0].split('_')[1]
            current_name = current_paths.split('/')[-4]

            if current_angle == gt_angle and current_name == gt_name:
                unhamonized_inputs.append(current_paths)

        unhamonized_inputs.sort()
        # randomly select an image from the filtered list and return it
        unharmonized = unhamonized_inputs[random.randint(0, len(unhamonized_inputs) - 1)]
        return unharmonized

    def get_unharmonized_all_input(self, indexed_image, images_gt_images_paths):

        # remove the indexed image ground truth from the list of all images
        new_paths = images_gt_images_paths.copy()
        new_paths.remove(indexed_image)

        # filter all the list such that only images that have the same object angle as that
        # of the ground truth image are retained
        unhamonized_inputs = []
        gt_angle = indexed_image.split('/')[-1].split('.')[0].split('_')[1]
        gt_name = indexed_image.split('/')[-4]

        for i in range(len(new_paths)):
            current_paths = new_paths[i]
            current_angle = current_paths.split('/')[-1].split('.')[0].split('_')[1]
            current_name = current_paths.split('/')[-4]

            if current_angle == gt_angle and current_name == gt_name:
                unhamonized_inputs.append(current_paths)

        unhamonized_inputs.sort()
        # randomly select an image from the filtered list and return it
        unharmonized = unhamonized_inputs[random.randint(0, len(unhamonized_inputs) - 1)]
        return unharmonized

def rotate_hdr(image, angle):
    # angle : 0 - 360 for rotation angle
    H, W, C = image.shape
    width = (angle / 360.0) * W

    front = image[:, 0:W - int(width)]
    back = image[:, W - int(width):]

    rotated = np.concatenate((back, front), 1)
    return rotated


def display(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def gamma_correction(image, gamma=2.2):
    return image ** (1 / gamma)


def do_tone_map(hdr):
    tonemapped = gamma_correction(hdr)
    return tonemapped * 255
