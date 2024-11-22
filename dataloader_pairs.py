from random import randrange
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import glob
import random
import cv2
from CLIP import clip

random.seed(1143)

def transform_matrix_offset_center(matrix, x, y):
	"""Return transform matrix offset center.

	Parameters
	----------
	matrix : numpy array
		Transform matrix
	x, y : int
		Size of image.

	Examples
	--------
	- See ``rotation``, ``shear``, ``zoom``.
	"""
	o_x = float(x) / 2 + 0.5
	o_y = float(y) / 2 + 0.5
	offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
	reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
	transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
	return transform_matrix 

def img_rotate(img, angle, center=None, scale=1.0):
	"""Rotate image.
	Args:
		img (ndarray): Image to be rotated.
		angle (float): Rotation angle in degrees. Positive values mean
			counter-clockwise rotation.
		center (tuple[int]): Rotation center. If the center is None,
			initialize it as the center of the image. Default: None.
		scale (float): Isotropic scale factor. Default: 1.0.
	"""
	(h, w) = img.shape[:2]

	if center is None:
		center = (w // 2, h // 2)

	matrix = cv2.getRotationMatrix2D(center, angle, scale)
	rotated_img = cv2.warpAffine(img, matrix, (w, h),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=(0,0,0),)
	return rotated_img

def zoom(x, zx, zy, row_axis=0, col_axis=1):
	zoom_matrix = np.array([[zx, 0, 0],
							[0, zy, 0],
							[0, 0, 1]])
	h, w = x.shape[row_axis], x.shape[col_axis]

	matrix = transform_matrix_offset_center(zoom_matrix, h, w) 
	x = cv2.warpAffine(x, matrix[:2, :], (w, h),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=(0,0,0),)
	return x

def augmentation(img,hflip,vflip,rot90,rot,zo,angle,zx,zy):
    if hflip:
        img=cv2.flip(img,1)
    if vflip:
        img=cv2.flip(img,0)
    if rot90:
        img = img.transpose(1, 0, 2)
    if zo:
        img=zoom(img, zx, zy)
    if rot:
        img=img_rotate(img,angle)
    return img

def preprocess_aug(img_list):
    hflip=random.random() < 0.5
    vflip=random.random() < 0.5
    rot90=random.random() < 0.5
    rot=random.random() <0.3
    zo=random.random()<0.0
    angle=random.random()*180-90
    zoom_range=(0.5, 1.5)
    zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    aug_img_list=[]
    for img in img_list:
        img = np.uint8((np.asarray(img)))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img=augmentation(img,hflip,vflip,rot90,rot,zo,angle,zx,zy)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        aug_img_list.append(img)
    return aug_img_list

device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
#load clip

model, preprocess = clip.load("ViT-B/32", device=device, download_root="./clip_model/")#ViT-B/32
for para in model.parameters():
	para.requires_grad = False

def populate_train_list(lowlight_images_path,overlight_images_path,normallight_images_path):
	
    image_list_lowlight = glob.glob(lowlight_images_path + "*")
    image_list_overlight = glob.glob(overlight_images_path + "*")
    image_list_normallight = glob.glob(normallight_images_path + "*")

    train_list1 = sorted(image_list_lowlight)
    train_list2 = sorted(image_list_overlight)
    train_list3 = sorted(image_list_normallight)

    return train_list1, train_list2, train_list3

	

class dataloader(data.Dataset):

	def __init__(self, lowlight_images_path,overlight_images_path, normallight_images_path):

		self.train_list1, self.train_list2, self.train_list3 = populate_train_list(lowlight_images_path,overlight_images_path, normallight_images_path) 
		self.size = 224

		print("Total training examples (Backlit):", len(self.train_list1))


	def __getitem__(self, index):

		data_lowlight_path = self.train_list1[index]
		data_overlight_path = self.train_list2[index]
		data_normallight_path = self.train_list3[index]
		data_lowlight = Image.open(data_lowlight_path)
		data_overlight = Image.open(data_overlight_path)
		data_normallight = Image.open(data_normallight_path)

		data_lowlight = data_lowlight.resize((512,512), Image.LANCZOS)
		data_overlight = data_overlight.resize((512,512), Image.LANCZOS)
		data_normallight = data_normallight.resize((512,512), Image.LANCZOS)
		w, h = data_lowlight.size
		x, y = randrange(w - self.size + 1), randrange(h - self.size + 1)
		cropped_a = data_lowlight.crop((x, y, x + self.size, y + self.size))
		cropped_b = data_overlight.crop((x, y, x + self.size, y + self.size))
		cropped_c = data_normallight.crop((x, y, x + self.size, y + self.size))
		image_list = preprocess_aug([cropped_a,cropped_b,cropped_c])
        
		data_lowlight, data_overlight, data_normallight = image_list[0], image_list[1], image_list[2]
		
		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight_output = torch.from_numpy(data_lowlight).float().permute(2,0,1)
		data_overlight = (np.asarray(data_overlight)/255.0) 
		data_overlight_output = torch.from_numpy(data_overlight).float().permute(2,0,1)
		data_normallight = (np.asarray(data_normallight)/255.0) 
		data_normallight_output = torch.from_numpy(data_normallight).float().permute(2,0,1)
		
		return data_lowlight_output, data_overlight_output, data_normallight_output

	def __len__(self):
		return len(self.train_list1)
	
