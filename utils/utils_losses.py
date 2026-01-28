import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from CLIP import clip

from utils import utils_image as util


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
#load clip
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")
model.to(device)
for para in model.parameters():
    para.requires_grad = False


clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
img_resize = transforms.Resize((224,224))


def get_clip_score(tensor, words):
    score=0
    text = clip.tokenize(words).to(device)
    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img_resize = transforms.Resize((224, 224))
    for i in range(tensor.shape[0]):
        #image preprocess
        image2 = img_resize(tensor[i])
        image = clip_normalizer(image2).unsqueeze(0)
        #get probabilitis
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1)
        print(probs)
        # 2-word-compared probability
        # prob = probs[0][0]/probs[0][1] # you may need to change this line for more words comparison
        prob = probs[0][0]
        score = score + prob

    return score


class L_clip(nn.Module):
    def __init__(self):
        super(L_clip, self).__init__()
        for param in self.parameters(): 
            param.requires_grad = False
  
    def forward(self, x, light):
        k1 = get_clip_score(x, ["dark", "normal light"])
        if light:
            k2 = get_clip_score(x, ["noisy photo", "clear photo"])
            return (k1 + k2)/2
        return k1



def get_clip_score_from_feature(tensor, text_features):
	score=0
	for i in range(tensor.shape[0]):
		image_in = img_resize(tensor[i])
		image = clip_normalizer(image_in.reshape(1, 3, 224, 224))

		image_features = model.encode_image(image)
		image_nor = image_features.norm(dim=-1, keepdim=True)
		nor= text_features.norm(dim=-1, keepdim=True)
		similarity = (100.0 * (image_features/image_nor) @ (text_features/nor).T).softmax(dim=-1) # @是矩阵乘法操作
		probs = similarity
		# print(probs)
		prob = 1 - probs[0][-1]
		score = score + prob
	score=score / tensor.shape[0]
	return score



class L_clip_from_feature(nn.Module):
	def __init__(self):
		super(L_clip_from_feature,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
  
	def forward(self,x,text_features):
		k1 = get_clip_score_from_feature(x, text_features)
		return k1



class L_clip_cr(nn.Module):
	def __init__(self):
		super(L_clip_cr,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
		self.feature_loss = nn.MSELoss()
  
	def forward(self, x, positive, negative, text_features):
		k1 = get_clip_score_from_feature(x, text_features)
		k2 = get_clip_score_from_feature(positive, text_features)
		k3 = get_clip_score_from_feature(negative, text_features)
		loss = torch.clamp(self.feature_loss(k1, k2)/(self.feature_loss(k1, k3) + 1e-5), 0, 1.0) 
		loss.requires_grad_(True)
		return loss









