import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import glob
import random

from CLIP import clip
from itertools import cycle

from utils import utils_image as util

random.seed(1143)

device = "cpu"
#load clip
model, preprocess = clip.load("ViT-B/32", device=device, download_root="./clip_model/")#ViT-B/32
for para in model.parameters():
	para.requires_grad = False

def populate_train_list(lowlight_images_path, overlight_images_path, normallight_images_path):
	
	image_list_lowlight = glob.glob(lowlight_images_path + "*")
	image_list_overlight = glob.glob(overlight_images_path + "*")
	image_list_normallight = glob.glob(normallight_images_path + "*")
	image_ref_list=image_list_normallight.copy()
	image_input_list=image_list_lowlight.copy() + image_list_overlight.copy()

	if len(image_list_normallight) == 0 or len(image_list_lowlight) == 0:
		raise Exception("one of the image lists is empty!", len(image_list_normallight), len(image_list_lowlight))
	
	if len(image_ref_list) < len(image_input_list) // 2:
		image_ref_list = list(next(cycle(image_ref_list)) for _ in range(len(image_input_list) // 2))
	elif len(image_input_list) < len(image_ref_list) * 2:
		image_input_list = list(next(cycle(image_input_list)) for _ in range(len(image_ref_list) * 2))
	
	train_list = image_input_list + image_ref_list
	random.shuffle(train_list)

	return train_list


class dataloader(Dataset):

	def __init__(self, lowlight_images_path, overlight_images_path, normallight_image_path, patch_size):
		
		self.train_list = populate_train_list(lowlight_images_path, overlight_images_path, normallight_image_path)
		self.patch_size = patch_size
		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))


	def __getitem__(self, index):

		data_path = self.data_list[index]
		
		data = util.imread_uint(data_path)
		
		data = cv2.resize(data, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LANCZOS4)
		# ---------------------------------
		# augmentation - flip, rotate
		# ---------------------------------
		mode = random.randint(0, 7)
		data = util.augment_img(data, mode=mode)
		
		image = util.uint2tensor4(data)
		image_features = model.encode_image(image)
		image_features /= image_features.norm(dim=-1, keepdim=True)

		if "trainC" in data_path:
			label=torch.from_numpy(np.array([0., 0., 1.]))
		elif "over" in data_path:
			label=torch.from_numpy(np.array([1., 0., 0.]))
		else:
			label=torch.from_numpy(np.array([0., 1., 0.]))
	
		return {'image_features': image_features, 'label': label}

	def __len__(self):
		return len(self.data_list)

