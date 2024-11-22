import os
import torch
import torch.nn as nn
import torch.optim
import argparse
import dataloader_pairs 
import model.model_small as model_small
from utils import inference, setup_seed
import losses
from collections import OrderedDict
from torch.optim import lr_scheduler
from CLIP import clip


task_name="train1"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
#load clip
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")#ViT-B/32
model.to(device)
for para in model.parameters():
    para.requires_grad = False

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x

class Prompts(nn.Module):
    def __init__(self,initials=None):
        super(Prompts,self).__init__()
        print("The initial prompts are:",initials)
        self.text_encoder = TextEncoder(model)
        if isinstance(initials,list):
            text = clip.tokenize(initials).cuda()
            self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_()).cuda()
        elif isinstance(initials,str):
            prompt_path=initials

            state_dict = torch.load(prompt_path)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.embedding_prompt=nn.Parameter(new_state_dict['embedding_prompt']).cuda()
            self.embedding_prompt.requires_grad = True
        else:
            self.embedding_prompt=torch.nn.init.xavier_normal_(nn.Parameter(model.token_embedding([" ".join(["X"]*config.length_prompt)," ".join(["X"]*config.length_prompt)," ".join(["X"]*config.length_prompt)]).requires_grad_())).cuda()

    def forward(self,tensor,flag=1):
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*config.length_prompt)]]) #config.length_prompt
        text_features = self.text_encoder(self.embedding_prompt,tokenized_prompts) 
        for i in range(tensor.shape[0]):
            image_features=tensor[i]
            nor=torch.norm(text_features,dim=-1, keepdim=True)
            if flag==0:
                similarity = (100.0 * image_features @ (text_features/nor).T)#.softmax(dim=-1)
                if(i==0):
                    probs=similarity
                else:
                    probs=torch.cat([probs,similarity],dim=0)
            else:
                similarity = (100.0 * image_features @ (text_features/nor).T).softmax(dim=-1)#/nor
                if(i==0):
                    probs = similarity[:,0]
                else:
                    probs=torch.cat([probs,similarity[:,0]],dim=0)
        return probs

def weights_init(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):
    setup_seed(2022)
    #load model
    if config.load_pretrain_prompt == True: 
        learn_prompt=Prompts(config.prompt_pretrain_dir).cuda()
    else: 
        learn_prompt=Prompts([" ".join(["X"]*(config.length_prompt))," ".join(["X"]*(config.length_prompt))," ".join(["X"]*(config.length_prompt))]).cuda() 

    learn_prompt = torch.nn.DataParallel(learn_prompt)
    fusion_net = model_small.Net()
    model_small.print_network(fusion_net)
    #add pretrained model weights
    fusion_net.apply(weights_init)
    fusion_net.cuda()
    
    #dataset
    train_dataset = dataloader_pairs.dataloader(config.lowlight_images_path,config.overlight_images_path, config.normallight_images_path)    #dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    
    #loss
    text_encoder = TextEncoder(model)
    L_clip_cr = losses.L_clip_cr().cuda()
    L_clip_from_feature = losses.L_clip_from_feature().cuda()
    L_clip_MSE = nn.MSELoss().cuda()

    #learnable prompt
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*config.length_prompt)]])
    embedding_prompt=learn_prompt.module.embedding_prompt 
    embedding_prompt.requires_grad = False
    text_features = text_encoder(embedding_prompt, tokenized_prompts)

    #fixed prompt
    # text = ["under-exposed","over-exposed","well-exposed"]
    # tokenized_prompts = torch.cat([clip.tokenize(p) for p in text])
    # tokens = clip.tokenize(text).cuda()
    # embedding_prompt = nn.Parameter(model.token_embedding(tokens).requires_grad_()).cuda()
    # text_features = text_encoder(embedding_prompt, tokenized_prompts)

    #fix the prompt and train the enhancement model
    for _, param in learn_prompt.named_parameters():
        param.requires_grad_(False)
    
    #load gradient update strategy.
    train_optimizer = torch.optim.Adam(fusion_net.parameters(), lr=config.train_lr, betas=(0.5, 0.999))
    lr_scheduler_s = lr_scheduler.MultiStepLR(train_optimizer, milestones=[config.num_epochs/2, config.num_epochs/3*2], gamma=0.1)
    #initial parameters
    fusion_net.train()

    #Start training!
    for epoch in range(1, config.num_epochs+1):
        
        if (epoch % config.display_epoch) == 0:
            torch.save(fusion_net.state_dict(), config.train_snapshots_folder + "epoch_" + str(epoch) + '.pth') 
            inference(config.lowlight_images_path,config.overlight_images_path,'./'+task_name+'/result_'+task_name+'/FUSION/result_epoch_'+str(epoch)+'/',fusion_net,224)

        for iteration, item in enumerate(train_loader): 
    
            img_lowlight, img_overlight, img_normal_light=item
            img_lowlight = img_lowlight.cuda()
            img_overlight = img_overlight.cuda() 
            img_normal_light = img_normal_light.cuda()
            final = fusion_net(img_lowlight, img_overlight)
            Loss_clip_cr = L_clip_from_feature(final,text_features)
            Loss_MSE = L_clip_MSE(final,img_normal_light)
            if epoch <= 5:
                loss = Loss_MSE
            else:
                loss = Loss_MSE + 1e-2 * Loss_clip_cr
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()


            print("training current learning rate: ",train_optimizer.state_dict()['param_groups'][0]['lr'])
            print("Loss at epoch",epoch, "iteration",iteration+1, ":", loss.item())
            
        lr_scheduler_s.step(epoch=epoch)
        
                


if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='PyTorch implementation of TextMEF')

    # Input Parameters
    parser.add_argument('-b','--lowlight_images_path', type=str, default="../Dataset/train_data/train/trainA/") 
    parser.add_argument('--overlight_images_path', type=str, default="../Dataset/train_data/train/trainB/")
    parser.add_argument('-r','--normallight_images_path', type=str, default="../Dataset/train_data/train/trainC/") 
    parser.add_argument('--length_prompt', type=int, default=16)
    parser.add_argument('--train_lr', type=float, default=2e-4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--noTV_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--display_epoch', type=int, default=5)
    parser.add_argument('--train_snapshots_folder', type=str, default="./"+task_name+"/"+"snapshots_train_"+task_name+"/")
    parser.add_argument('--load_pretrain_prompt', type=lambda x: (str(x).lower() == 'true'), default= True)
    parser.add_argument('--prompt_pretrain_dir', type=str, default= './train1/snapshots_prompt_train1/best_prompt_round0.pth')
    config = parser.parse_args()

    if not os.path.exists(config.train_snapshots_folder):
        os.mkdir(config.train_snapshots_folder)

    train(config)
