from torch.autograd import Variable
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from models import Generator , Discriminator
from dataset import transform_hl_pair
from dataset import arrange_data
from dataset import WIDER
from skimage.measure import compare_ssim as ssim
import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import matplotlib 
import pytorch_ssim
from nuaa import Done,Dst
import os







def transform_hl_pair(hr_height, hr_width):

    tf = [  #transforms.Resize((hr_height // 4, hr_width // 4), Image.BICUBIC),
            transforms.ToTensor(),                            
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  

    
    return transforms.Compose(tf)


high_resolution = 128
trans_lr = transform_hl_pair(high_resolution, high_resolution)




weights_ = torch.load("./models/generator_102.pth")
weights = OrderedDict()
#discriminator = Discriminator()
for k, v in weights_.items():
    weights[k.split('module.')[-1]] = v
gen_srgan = Generator()
gen_srgan.load_state_dict(weights)
gen_srgan = gen_srgan.eval()

i = 0
imgs = r"D:\2018NUAA\small classification via GAN\gan图片\PSNR SSIM\down"
for child_dir in os.listdir(imgs):
    i = i + 1
    child_path = os.path.join(imgs, child_dir)
    img = Image.open(child_path) #?*?*3
    img = img.convert('RGB')
    img_lr = trans_lr(img).unsqueeze(0)
    g1, g2 = gen_srgan(img_lr)
    f = lambda x:np.transpose(x.squeeze(), (1, 2, 0)) 
    g2_recover = torch.clamp(g2*0.5 + 0.5, 0, 1)    
    g2rp=f(g2_recover.detach())   
    g2rp = g2rp.numpy()
    matplotlib.image.imsave(r"D:\2018NUAA\small classification via GAN\gan图片\PSNR SSIM\ours\%d.png"%(i),g2rp)
