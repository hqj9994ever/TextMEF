import torch
import os
import random
import numpy as np
from PIL import Image
import glob
import torchvision
from torchvision import transforms
import logging
import torch.distributed as dist


def fusion(lowlight_image_path,overlight_image_path,image_list_path,result_list_path,DCE_net,size=256): 
	data_l = Image.open(lowlight_image_path)
	data_o = Image.open(overlight_image_path)
	data_l = data_l.resize((size,size), Image.LANCZOS)
	data_o = data_o.resize((size,size), Image.LANCZOS)
	data_l = (np.asarray(data_l)/255.0) 
	data_o = (np.asarray(data_o)/255.0)
     
	data_l = torch.from_numpy(data_l).float()
	data_o = torch.from_numpy(data_o).float()
     
	data_l = data_l.permute(2,0,1)
	data_o = data_o.permute(2,0,1)
	data_l = data_l.cuda().unsqueeze(0)
	data_o = data_o.cuda().unsqueeze(0)
	enhanced_image = torch.clamp(DCE_net(data_l, data_o)[0], 0, 1)
	
	image_path = lowlight_image_path.replace(image_list_path,result_list_path)
     
	image_path = image_path.replace('.JPG','.png')  #?
	output_path = image_path
	if not os.path.exists(output_path.replace('/'+image_path.split("/")[-1],'')): 
		os.makedirs(output_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(enhanced_image, output_path)

def inference(lowlight_image_list_path,overlight_image_list_path,result_list_path,DCE_net,size=256):
    with torch.no_grad():
        filePath_low = lowlight_image_list_path
        filePath_over = overlight_image_list_path
        file_list_low = sorted(os.listdir(filePath_low))
        file_list_over = sorted(os.listdir(filePath_over))
        
        print("Inferencing...")
        for file_name_a, file_name_b in zip(file_list_low,file_list_over):
            lowlight_image = glob.glob(filePath_low+file_name_a)[0]
            overlight_image = glob.glob(filePath_over+file_name_b)[0]
            fusion(lowlight_image,overlight_image,lowlight_image_list_path,result_list_path,DCE_net,size)


def Get_color_MEF_tensor(cb1, cr1, cb2, cr2):

      cb1s = [x.clone().squeeze().cpu().numpy()*255 for x in torch.split(cb1, 1, dim=0)]
      cr1s = [x.clone().squeeze().cpu().numpy()*255 for x in torch.split(cr1, 1, dim=0)]
      cb2s = [x.clone().squeeze().cpu().numpy()*255 for x in torch.split(cb2, 1, dim=0)]
      cr2s = [x.clone().squeeze().cpu().numpy()*255 for x in torch.split(cr2, 1, dim=0)]
      cb = []
      cr = []
      for Cb1, Cr1, Cb2, Cr2 in zip(cb1s, cr1s, cb2s, cr2s):
        H, W = Cb1.shape
        Cb = np.ones((H, W))
        Cr = np.ones((H, W))
        for k in range(H):
            for n in range(W):
                if abs(Cb1[k, n] - 128) == 0 and abs(Cb2[k, n] - 128) == 0:
                    Cb[k, n] = 128
                else:
                    middle_1 = Cb1[k, n] * abs(Cb1[k, n] - 128) + Cb2[k, n] * abs(Cb2[k, n] - 128)
                    middle_2 = abs(Cb1[k, n] - 128) + abs(Cb2[k, n] - 128)
                    Cb[k, n] = middle_1 / middle_2
                if abs(Cr1[k, n] - 128) == 0 and abs(Cr2[k, n] - 128) == 0:
                    Cr[k, n] = 128
                else:
                    middle_3 = Cr1[k, n] * abs(Cr1[k, n] - 128) + Cr2[k, n] * abs(Cr2[k, n] - 128)
                    middle_4 = abs(Cr1[k, n] - 128) + abs(Cr2[k, n] - 128)
                    Cr[k, n] = middle_3 / middle_4
        cb.append(transforms.ToTensor()(Cb.astype(np.uint8)).reshape(1, 1, H, W))
        cr.append(transforms.ToTensor()(Cr.astype(np.uint8)).reshape(1, 1, H, W))
      cb = torch.cat(cb, dim=0).cuda()
      cr = torch.cat(cr, dim=0).cuda()
      return cb, cr


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
            .transpose(1, 3)
            .transpose(2, 3)
    )
    return out
                

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_root_logger(logger_name='basicsr',
                    log_level=logging.INFO,
                    log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger

