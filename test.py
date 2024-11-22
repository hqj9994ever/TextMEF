import os
import argparse
import torch
import torchvision
import torch.optim
import model.model_small as model_small
import numpy as np
from PIL import Image
from torchvision import transforms

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='PyTorch implementation of TextMEF')
parser.add_argument('--input_u', help='directory of input folder', default='../Dataset/test_data/misaligned/under/')
parser.add_argument('--input_o', help='directory of input folder', default='../Dataset/test_data/misaligned/over/')
parser.add_argument('-o', '--output', help='directory of output folder', default='./inference/')
parser.add_argument('-c', '--ckpt', help='test ckpt path', default='./train1/snapshots_train_train1/epoch_200.pth')

args = parser.parse_args()

net = model_small.Net()

state_dict = torch.load(args.ckpt)       
net.load_state_dict(state_dict)
net.cuda()
to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

def inference(lowlight_image_path, overlight_image_path): 

	data_lowlight = Image.open(lowlight_image_path).convert('RGB')
	data_lowlight = (np.asarray(data_lowlight)/255.0) 
	
	data_lowlight = torch.from_numpy(data_lowlight).float().cuda()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.unsqueeze(0) 
	
	data_overlight = Image.open(overlight_image_path).convert('RGB')
	data_overlight = (np.asarray(data_overlight)/255.0) 
	
	data_overlight = torch.from_numpy(data_overlight).float().cuda()
	data_overlight = data_overlight.permute(2,0,1)
	data_overlight = data_overlight.unsqueeze(0)

	final = net(data_lowlight, data_overlight)
	final = torch.clamp(final, 0, 1)
	image_path = args.output+os.path.basename(lowlight_image_path)

	image_path = image_path.replace('.jpg','.png')
	image_path = image_path.replace('.JPG','.png')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')): 
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(final, result_path)
	
if __name__ == '__main__':
	with torch.no_grad():
		filePath_u = args.input_u
		filePath_o = args.input_o
		file_list_u = sorted(os.listdir(filePath_u))
		file_list_o = sorted(os.listdir(filePath_o))
  
		for file_name_u, file_name_o in zip(file_list_u, file_list_o):
			image_u=filePath_u+file_name_u
			image_o=filePath_o+file_name_o
			print(image_u)
			inference(image_u, image_o)

		

