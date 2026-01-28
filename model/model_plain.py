import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.nn.parallel import DataParallel

from collections import OrderedDict

from model.model_base import ModelBase
from model.network import define_G
from model.prompt import Prompts, TextEncoder

from utils.utils_losses import L_clip_from_feature, L_clip_cr

from CLIP import clip


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------------------------------------
# load CLIP model
# ----------------------------------------
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")
model.to(device)
for para in model.parameters():
    para.requires_grad = False


class ModelPlain(ModelBase):
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.netG = define_G(in_channels=opt.in_channels)
        self.netG = self.model_to_device(self.netG)


    def init_train(self, init_paths):
        self.load(init_paths)
        self.load_prompt()
        self.netG.train()
        self.define_loss()
        self.define_optimizer()
        self.define_scheduler()
        self.log_dict = OrderedDict()

    
    def model_to_device(self, net):
        net = net.to(self.device)
        net = DataParallel(net)
        return net


    def load(self, init_paths):
        load_path_G = init_paths[0]

        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=True, param_key='params')


    def load_prompt(self):
        if self.opt.load_pretrain_prompt == True: 
            learned_prompt = Prompts(self.opt, self.opt.prompt_pretrain_dir).to(self.device)
        else: 
            learned_prompt = Prompts(self.opt, [" ".join(["X"]*(self.opt.length_prompt))," ".join(["X"]*(self.opt.length_prompt))," ".join(["X"]*(self.opt.length_prompt))]).to(self.device) 

        learned_prompt = DataParallel(learned_prompt)
        
        for _, param in learned_prompt.named_parameters():
            param.requires_grad_(False)

        text_encoder = TextEncoder(model)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*self.opt.length_prompt)]])
        embedding_prompt = learned_prompt.module.embedding_prompt 
        embedding_prompt.requires_grad = False
        self.text_features = text_encoder(embedding_prompt, tokenized_prompts)



    def define_loss(self):
        self.L_clip = L_clip_from_feature().to(self.device)
        self.L_content = nn.MSELoss().to(self.device)


    def define_optimizer(self):
        
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        self.train_optimizerG = torch.optim.Adam(G_optim_params, lr=self.opt.train_lr, betas=(0.5, 0.999))
    

    def define_scheduler(self):
        self.lr_schedulerG = lr_scheduler.MultiStepLR(self.train_optimizerG, milestones=[self.opt.num_epochs/4, self.opt.num_epochs/2, self.opt.num_epochs/4*3], gamma=0.4)    


    def feed_data(self, data):
        self.data = data
        self.under = self.data['under'].to(self.device)
        self.over = self.data['over'].to(self.device)
        self.gt = self.data['gt'].to(self.device) if self.data['gt'] is not None else None
        self.image_path = self.data['img_path'] if self.data['img_path'] is not None else None

    
    
    def netG_forward(self):
        self.output = self.netG(self.under, self.over) 


    # ----------------------------------------
    # optimization
    # ----------------------------------------
    def optimize_parametersG(self):
        self.train_optimizerG.zero_grad()
        self.netG_forward()
        loss_content = self.L_content(self.output, self.gt)
        loss_clip = self.L_clip(self.output, self.text_features)
        loss_G = loss_content + 1e-2 * loss_clip
        loss_G.backward()
        self.train_optimizerG.step()

        self.log_dict['loss_content'] = loss_content.item()
        self.log_dict['loss_clip'] = loss_clip.item()
        self.log_dict['loss_G'] = loss_G.item()
        

    def optimize_parameters(self):
        self.optimize_parametersG()


    def update_learning_rate(self, epoch):
        self.lr_schedulerG.step(epoch)


    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()
    

    def current_log(self):
        return self.log_dict
    

    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['E'] = self.output.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.gt.detach()[0].float().cpu()
        return out_dict


    def print_network(self):
        num_params = 0
        for param in self.netG.parameters():
            num_params += param.numel()
        print('Total number of parameters: %d' % num_params)


    def save(self, epoch):
        self.save_network(self.save_dir, self.netG, 'G', epoch)

    
    def current_learning_rate(self):
        return self.lr_schedulerG.get_lr()[0]



