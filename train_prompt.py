import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim

import argparse
from data import dataloader_prompt_add
from model.prompt import Prompts, TextEncoder

from CLIP import clip

task_name="train1"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
#load clip
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")#ViT-B/32
model.to(device)
for para in model.parameters():
    para.requires_grad = False


def main(config):
    # ----------------------------------------
    # initialize training parameters
    # ----------------------------------------

    total_iteration=0
    cur_iteration=0
    min_prompt_loss=10000
    rounds=0 
    
    # ----------------------------------------
    # load pretrained model weights
    # ----------------------------------------

    if config.load_pretrain_prompt == True: 
        learn_prompt = Prompts(config, config.prompt_pretrain_dir).to(device)
    else: 
        learn_prompt = Prompts(config, [" ".join(["X"]*(config.length_prompt))," ".join(["X"]*(config.length_prompt))," ".join(["X"]*(config.length_prompt))]).to(device) 
    learn_prompt =  torch.nn.DataParallel(learn_prompt)
    
    prompt_train_dataset = dataloader_prompt_add.dataloader(config.lowlight_images_path, config.overlight_images_path, config.normallight_images_path, config.patch_size)
    prompt_train_loader = torch.utils.data.DataLoader(prompt_train_dataset, batch_size=config.prompt_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    prompt_optimizer = torch.optim.Adam(learn_prompt.parameters(), lr=config.prompt_lr, weight_decay=config.weight_decay)

    # ----------------------------------------
    # start training
    # ----------------------------------------
    for epoch in range(config.num_epochs):
        if total_iteration < config.num_clip_pretrained_iters:
            for iteration, item in enumerate(prompt_train_loader):
                text_features, label = item['image_features'], item['label']    
                text_features = text_features.cuda()
                label = label.cuda()
                output = learn_prompt(text_features, 0) 
                loss = F.cross_entropy(output, label) 
                prompt_optimizer.zero_grad()
                loss.backward()
                prompt_optimizer.step()

                if ((total_iteration + 1) % config.prompt_display_iter) == 0:
                    if loss < min_prompt_loss:
                        min_prompt_loss = loss
                        print("find better prompt model")
                        torch.save(learn_prompt.state_dict(), config.prompt_snapshots_folder + "best_prompt_round" + str(rounds) + '.pth')
                    print("prompt current learning rate: ", prompt_optimizer.state_dict()['param_groups'][0]['lr'])
                    print("Loss at iteration", total_iteration + 1, ":", loss.item())
                    print("output",output.softmax(dim=-1), "label", label)
                    print("cross_entropy_loss", loss)
                    
                    print(cur_iteration + 1," ", total_iteration + 1)
                
                cur_iteration+=1
                total_iteration+=1 
        else: break    

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='PyTorch implementation of TextMEF')

    # Input Parameters
    parser.add_argument('-b','--lowlight_images_path', type=str, default="./Dataset/train_data/SICE/trainA/") 
    parser.add_argument('--overlight_images_path', type=str, default="./Dataset/train_data/SICE/trainB/")
    parser.add_argument('-r','--normallight_images_path', type=str, default="./Dataset/train_data/SICE/trainC/") 
    parser.add_argument('--length_prompt', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--prompt_lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=180) #3000
    parser.add_argument('--num_clip_pretrained_iters', type=int, default=10000) #8000
    parser.add_argument('--prompt_batch_size', type=int, default=16) #32
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prompt_display_iter', type=int, default=20)
    parser.add_argument('--train_snapshots_folder', type=str, default="./"+task_name+"/"+"snapshots_train_"+task_name+"/")
    parser.add_argument('--prompt_snapshots_folder', type=str, default="./"+task_name+"/"+"snapshots_prompt_"+task_name+"/")
    parser.add_argument('--load_pretrain', type=lambda x: (str(x).lower() == 'true'), default= False)
    parser.add_argument('--load_pretrain_prompt', type=lambda x: (str(x).lower() == 'true'), default= False)
    parser.add_argument('--prompt_pretrain_dir', type=str, default= './train0/snapshots_prompt_train0/best_prompt_round0.pth')
    
    config = parser.parse_args()

    os.makedirs(config.train_snapshots_folder, exist_ok=True)
    os.makedirs(config.prompt_snapshots_folder, exist_ok=True)


    main(config)
