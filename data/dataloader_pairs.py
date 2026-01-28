from torch.utils.data import Dataset

import numpy as np
import glob
import random
import cv2

from utils import utils_image as util


random.seed(1143)


def populate_train_list(lowlight_images_path, overlight_images_path, normallight_images_path):
	
	image_list_lowlight = glob.glob(lowlight_images_path + "*")
	image_list_overlight = glob.glob(overlight_images_path + "*")
	image_list_normallight = glob.glob(normallight_images_path + "*")

	train_list1 = sorted(image_list_lowlight)
	train_list2 = sorted(image_list_overlight)
	train_list3 = sorted(image_list_normallight)

	return train_list1, train_list2, train_list3

	

class dataloader(Dataset):

	def __init__(self, lowlight_images_path, overlight_images_path, normallight_images_path, patch_size, phase):

		self.train_list1, self.train_list2, self.train_list3 = populate_train_list(lowlight_images_path, overlight_images_path, normallight_images_path)
		self.phase = phase 
		self.patch_size = patch_size
		
		if phase == 'train':
			print("Total training examples:", len(self.train_list1))
		elif phase == 'val':
			print("Total validation examples:", len(self.train_list1))


	def __getitem__(self, index):
		if self.phase == 'train':
			data_lowlight_path = self.train_list1[index]
			data_overlight_path = self.train_list2[index]
			data_normallight_path = self.train_list3[index]
			data_lowlight = util.imread_uint(data_lowlight_path, n_channels=3)
			data_overlight = util.imread_uint(data_overlight_path, n_channels=3)
			data_normallight = util.imread_uint(data_normallight_path, n_channels=3)

			H, W = data_lowlight.shape[:2]
			# ---------------------------------
			# randomly crop the image (local)
			# ---------------------------------           
			rnd_h = random.randint(0, max(0, H - self.patch_size))
			rnd_w = random.randint(0, max(0, W - self.patch_size))
			patch_lowlight = data_lowlight[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]
			patch_overlight = data_overlight[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]
			patch_normallight = data_normallight[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]
            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
			mode = [random.randint(0, 7), random.randint(0, 7)]
			patch_lowlight = util.augment_img(patch_lowlight, mode=mode[0])
			patch_overlight = util.augment_img(patch_overlight, mode=mode[0])
			patch_normallight = util.augment_img(patch_normallight, mode=mode[0])
			# ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
			patch_lowlight = util.uint2tensor3(patch_lowlight)
			patch_overlight = util.uint2tensor3(patch_overlight)
			patch_normallight = util.uint2tensor3(patch_normallight)
			
			return {'under': patch_lowlight, 'over': patch_overlight, 'gt': patch_normallight, 'img_path': data_lowlight_path}
		
		elif self.phase == 'val':
			data_lowlight_path = self.train_list1[index]
			data_overlight_path = self.train_list2[index]
			data_normallight_path = self.train_list3[index]
			img_lowlight = util.imread_uint(data_lowlight_path, n_channels=3)
			img_overlight = util.imread_uint(data_overlight_path, n_channels=3)
			img_normallight = util.imread_uint(data_normallight_path, n_channels=3)

			img_lowlight = util.uint2tensor3(img_lowlight)
			img_overlight = util.uint2tensor3(img_overlight)
			img_normallight = util.uint2tensor3(img_normallight)

			return {'under': img_lowlight, 'over': img_overlight, 'gt': img_normallight, 'img_path': data_lowlight_path}

	def __len__(self):
		return len(self.train_list1)
	
