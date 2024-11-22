import torchvision.transforms as transforms
import torch
from CLIP import clip
import torch.nn as nn
from torch.nn import functional as F
from CLIP.clip import load

device = "cuda" if torch.cuda.is_available() else "cpu"
#load clip
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")#"ViT-B/32"
model.to(device)
for para in model.parameters():
	para.requires_grad = False

def get_clip_score(tensor,words):
	score=0
	for i in range(tensor.shape[0]):
		#image preprocess
		clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
		img_resize = transforms.Resize((224,224))
		image2=img_resize(tensor[i])
		image=clip_normalizer(image2).unsqueeze(0)
		#get probabilitis
		text = clip.tokenize(words).to(device)
		logits_per_image, logits_per_text = model(image, text)
		probs = logits_per_image.softmax(dim=-1)
		#2-word-compared probability
		# prob = probs[0][0]/probs[0][1]#you may need to change this line for more words comparison
		prob = probs[0][0]
		score =score + prob

	return score


class L_clip(nn.Module):
	def __init__(self):
		super(L_clip,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
  
	def forward(self, x, light):
		k1 = get_clip_score(x,["dark","normal light"])
		if light:
			k2 = get_clip_score(x,["noisy photo","clear photo"])
			return (k1+k2)/2
		return k1

class Prompts(nn.Module):
	def __init__(self,initials=None):
		super(Prompts,self).__init__()
		if initials!=None:
			text = clip.tokenize(initials).cuda()
			with torch.no_grad():
				self.text_features = model.encode_text(text).cuda()
		else:
			self.text_features=torch.nn.init.xavier_normal_(nn.Parameter(torch.cuda.FloatTensor(2,512))).cuda()

	def forward(self,tensor):
		for i in range(tensor.shape[0]):
			image_features=tensor[i]
			nor=torch.norm(self.text_features,dim=-1, keepdim=True)
			similarity = (model.logit_scale.exp() * image_features @ (self.text_features/nor).T).softmax(dim=-1)
			if(i==0):
				probs=similarity
			else:
				probs=torch.cat([probs,similarity],dim=0)
		return probs

learn_prompt=Prompts().cuda()
clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
img_resize = transforms.Resize((224,224))

def get_clip_score_from_feature(tensor, text_features):
	score=0
	for i in range(tensor.shape[0]):
		image2=img_resize(tensor[i])
		image=clip_normalizer(image2.reshape(1,3,224,224))
  
		image_features = model.encode_image(image)
		image_nor=image_features.norm(dim=-1, keepdim=True)
		nor= text_features.norm(dim=-1, keepdim=True)
		similarity = (100.0 * (image_features/image_nor) @ (text_features/nor).T).softmax(dim=-1) # @是矩阵乘法操作
		probs = similarity
		# print(probs)
		prob = 1-probs[0][-1]
		score = score + prob
	score=score/tensor.shape[0]
	return score


class L_clip_from_feature(nn.Module):
	def __init__(self):
		super(L_clip_from_feature,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
  
	def forward(self,x,text_features):
		k1 = get_clip_score_from_feature(x,text_features)
		return k1

class L_clip_cr(nn.Module):
	def __init__(self):
		super(L_clip_cr,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
		self.feature_loss = nn.MSELoss()
  
	def forward(self, x, positive, negative, text_features):
		k1 = get_clip_score_from_feature(x,text_features)
		k2 = get_clip_score_from_feature(positive,text_features)
		k3 = get_clip_score_from_feature(negative,text_features)
		loss = torch.clamp(self.feature_loss(k1,k2)/(self.feature_loss(k1,k3)+0.00001),0, 1.0) 
		loss.requires_grad_(True)
		return loss

#for clip reconstruction loss
res_model, res_preprocess = load("RN101", device=device, download_root="./clip_model/")
for para in res_model.parameters():
	para.requires_grad = False

def l2_layers(pred_conv_features, input_conv_features,weight):
	weight=torch.tensor(weight).type(pred_conv_features[0].dtype)
	return weight@torch.tensor([torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
			zip(pred_conv_features, input_conv_features)],requires_grad=True)/len(weight)

def get_clip_score_MSE(pred,inp,weight):
	score=0
	for i in range(pred.shape[0]):

		pred_img=img_resize(pred[i])
		pred_img=pred_img.reshape(1,3,224,224)
		pred_image_features = res_model.encode_image(pred_img)

		inp_img=img_resize(inp[i])
		inp_img=inp_img.reshape(1,3,224,224)
		inp_image_features = res_model.encode_image(inp_img)

		MSE_loss_per_img=0

		for feature_index in range(len(weight)):
				MSE_loss_per_img=MSE_loss_per_img+weight[feature_index]*F.mse_loss(pred_image_features[1][feature_index].squeeze(0),inp_image_features[1][feature_index].squeeze(0))

		score = score + MSE_loss_per_img
	return score


class L_clip_MSE(nn.Module):
	def __init__(self):
		super(L_clip_MSE,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
		
	def forward(self,pred,inp,weight=[1.0,1.0,1.0,1.0,0.5]):
		res = get_clip_score_MSE(pred,inp,weight)
		return res
	
		


