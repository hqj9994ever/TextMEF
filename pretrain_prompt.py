import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim
import argparse
import dataloader_prompt_add
from collections import OrderedDict
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
        tokenized_prompts= torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*config.length_prompt)]]) 
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
                    probs = similarity
                else:
                    probs=torch.cat([probs,similarity],dim=0)
        return probs


def train(config):
    
    #add pretrained model weights
    if config.load_pretrain_prompt == True: 
        learn_prompt=Prompts(config.prompt_pretrain_dir).cuda()
    else: 
        learn_prompt=Prompts([" ".join(["X"]*(config.length_prompt))," ".join(["X"]*(config.length_prompt))," ".join(["X"]*(config.length_prompt))]).cuda() 
    learn_prompt =  torch.nn.DataParallel(learn_prompt)
    
    prompt_train_dataset = dataloader_prompt_add.dataloader(config.lowlight_images_path, config.overlight_images_path, config.normallight_images_path)
    prompt_train_loader = torch.utils.data.DataLoader(prompt_train_dataset, batch_size=config.prompt_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    prompt_optimizer = torch.optim.Adam(learn_prompt.parameters(), lr=config.prompt_lr, weight_decay=config.weight_decay)

    #initial parameters
    total_iteration=0
    cur_iteration=0
    min_prompt_loss=10000
    rounds=0 
    
    #Start training!
    for epoch in range(config.num_epochs):
            if total_iteration<config.num_clip_pretrained_iters:
                for iteration, item in enumerate(prompt_train_loader):
                    text_features, label=item    
                    text_features = text_features.cuda()
                    label = label.cuda()
                    output = learn_prompt(text_features, 0) 
                    loss = F.cross_entropy(output,label) 
                    prompt_optimizer.zero_grad()
                    loss.backward()
                    prompt_optimizer.step()

                    if ((total_iteration + 1) % config.prompt_display_iter) == 0:
                        if loss<min_prompt_loss:
                            min_prompt_loss = loss
                            print("find better prompt model")
                            torch.save(learn_prompt.state_dict(), config.prompt_snapshots_folder + "best_prompt_round" + str(rounds) + '.pth')
                        print("prompt current learning rate: ",prompt_optimizer.state_dict()['param_groups'][0]['lr'])
                        print("Loss at iteration", total_iteration+1, ":", loss.item())
                        print("output",output.softmax(dim=-1),"label",label)
                        print("cross_entropy_loss",loss)
                        
                        print(cur_iteration+1," ",total_iteration+1)
                   
                    cur_iteration+=1
                    total_iteration+=1 
            else: break    

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='PyTorch implementation of TextMEF')

    # Input Parameters
    parser.add_argument('-b','--lowlight_images_path', type=str, default="../Dataset/train_data/SICE/under/") 
    parser.add_argument('--overlight_images_path', type=str, default="../Dataset/train_data/SICE/over/")
    parser.add_argument('-r','--normallight_images_path', type=str, default="../Dataset/train_data/train/trainC/") 
    parser.add_argument('--length_prompt', type=int, default=16)
    parser.add_argument('--prompt_lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.001)
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

    if not os.path.exists(config.train_snapshots_folder.split('/')[1]):
        print(config.train_snapshots_folder.split('/')[1])
        os.mkdir(config.train_snapshots_folder.split('/')[1])
    if not os.path.exists(config.train_snapshots_folder):
        os.mkdir(config.train_snapshots_folder)
    if not os.path.exists(config.prompt_snapshots_folder):
        os.mkdir(config.prompt_snapshots_folder)


    train(config)
