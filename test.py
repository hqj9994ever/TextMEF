import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import argparse
import torch
import logging
from collections import OrderedDict

from model.network import FusionNet
from utils import utils_image as util
from utils import utils_logger


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='PyTorch implementation of TextMEF')
parser.add_argument('--input_u', help='directory of input folder', default='Dataset/test_data/test/under/')
parser.add_argument('--input_o', help='directory of input folder', default='Dataset/test_data/test/over/')
parser.add_argument('--gt', help='directory of gt folder', default='Dataset/test_data/test/gt/')
parser.add_argument('--output', help='directory of output folder', default='results/')
parser.add_argument('--need_H', help='have ground truth or not', action='store_true', default=False)
parser.add_argument('--model_G_path', help='test ckpt path', default='model_zoo/epoch_400_G.pth')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_G = FusionNet(base_filter=64)
model_G.load_state_dict(torch.load(args.model_G_path))
model_G.eval()
for k, v in model_G.named_parameters():
	v.requires_grad = False
model_G = model_G.to(device)

test_results = OrderedDict()
test_results['psnr'] = []
test_results['ssim'] = []



def inference(lowlight_image_path, overlight_image_path, normallight_image_path=None):
	
	data_U = util.imread_uint(lowlight_image_path, n_channels=3)
	data_O = util.imread_uint(overlight_image_path, n_channels=3)
	
	if args.need_H:
		img_H = util.imread_uint(normallight_image_path, n_channels=3)

	
	data_U = util.uint2tensor4(data_U).to(device)
	data_O = util.uint2tensor4(data_O).to(device)
	
	img_E = model_G(data_U, data_O)
	
	img_E = util.tensor2uint(img_E)

	if args.need_H:
		psnr = util.calculate_psnr(img_E, img_H, border=0)
		ssim = util.calculate_ssim(img_E, img_H, border=0)
		test_results['psnr'].append(psnr)
		test_results['ssim'].append(ssim)
		logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(os.path.basename(lowlight_image_path), psnr, ssim))
	else:
		logger.info('{:s}.'.format(os.path.basename(lowlight_image_path)))
	
	image_path = os.path.join(args.output, os.path.basename(lowlight_image_path))
	util.imsave(img_E, image_path)



if __name__ == '__main__':

	os.makedirs(args.output, exist_ok=True)

	logger_name = 'test'
	utils_logger.logger_info(logger_name, os.path.join(args.output, logger_name+'.log'))
	logger = logging.getLogger(logger_name)
	

	with torch.no_grad():
		filePath_u = args.input_u
		filePath_o = args.input_o
		file_list_u = sorted(os.listdir(filePath_u))
		file_list_o = sorted(os.listdir(filePath_o))

		if args.need_H:
			filePath_gt = args.gt
			file_list_gt = sorted(os.listdir(filePath_gt))
  
			for file_name_u, file_name_o, file_name_gt in zip(file_list_u, file_list_o, file_list_gt):
				image_u_path = os.path.join(filePath_u, file_name_u)
				image_o_path = os.path.join(filePath_o, file_name_o)
				image_gt_path = os.path.join(filePath_gt, file_name_gt)
				inference(image_u_path, image_o_path, image_gt_path)
		else:
			for file_name_u, file_name_o in zip(file_list_u, file_list_o):
				image_u_path = os.path.join(filePath_u, file_name_u)
				image_o_path = os.path.join(filePath_o, file_name_o)
				inference(image_u_path, image_o_path)

		if args.need_H:
			ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
			ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
			logger.info('Average PSNR/SSIM(RGB) - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(ave_psnr, ave_ssim))


		
	
		

