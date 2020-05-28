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


high_resolution = 128
trans_lr = transform_hl_pair(high_resolution, high_resolution)




weights_ = torch.load("./models/generator_409.pth")
weights = OrderedDict()
#discriminator = Discriminator()
for k, v in weights_.items():
    weights[k.split('module.')[-1]] = v
gen_srgan = Generator()
gen_srgan.load_state_dict(weights)
gen_srgan = gen_srgan.eval()

img = Image.open("./imgs/80.png") #?*?*3
#imgg= np.array(img)
img = img.convert('RGB')
img_lr = trans_lr(img).unsqueeze(0)  #1 3 32 32
#print(img_lr)
#print(img_lr.size())
img_hr = trans_lr(img).unsqueeze(0)  #1 3 128 128
#print(img_lr.size())
g1, g2 = gen_srgan(img_lr)    #1 3 128 128
f = lambda x:np.transpose(x.squeeze(), (1, 2, 0)) #tensor 2 pilimg
g1_recover = torch.clamp(g1*0.5 + 0.5, 0, 1)    #1 3 128 128  [-1 1] 2 [0 1]
g2_recover = torch.clamp(g2*0.5 + 0.5, 0, 1)    #1 3 128 128
img_lr = torch.clamp(img_lr*0.5 + 0.5, 0, 1)    #


#dl = discriminator(gen_srgan(img_lr)[1].detach())

#print(dl)



#mse = nn.MSELoss(reduction='sum')


#ssim_loss1 = pytorch_ssim.ssim(g1, img_hr).item()


#ssim_loss2 = pytorch_ssim.ssim(g2, img_hr).item()

#print(ssim_loss1)

#print(ssim_loss2)

#mse_loss1 = mse(g1,img_hr)

#mse_loss2 = mse(g2,img_hr)


#print(mse_loss1)

#print(mse_loss2)

#a = Done(dl,1)
#b = Dst(dl,1)
#print(dl)
#print(a)
#print(b)









#print(dl[0])
#print(img_lr.size())   #1 3 32 32
lrp=f(img_lr.detach())
#lrp = lrp.numpy()
#print(lrp.size())    #32*32*3
#lrp.show()
plt.imshow(lrp)
#plt.savefig('l_256.png')
plt.show()
matplotlib.image.imsave("01.jpg",lrp)



#print(g2_recover.size())   #1*3*128*128
g2rp=f(g2_recover.detach())#128*128*3
#g2rp= g2rp.numpy()
#print(g2rp.size())   
plt.imshow(g2rp)#plt.savefig('G_256.png')
plt.show()
#g2rp.save(os.path.join("./imgs/00.jpg"))
matplotlib.image.imsave("00.jpg",g2rp)




































#hr=wangxu(img_hr)
#g2=wangxu(g2_recover.detach())
#print(g2.shape)
#print(g2)
#img_hr=f(img_hr)
#hr=hr.numpy()
#g2=g2.numpy()
#print(g2.shape)
#print(g2)
#g2=g2 * 255
#hr=hr * 255
#print(g2)
#print("---------")
#print(g2.shape)
#print(g2[0])
#ssim = ssim(g2, hr, multichannel=True)
 
#print(ssim)

















































































































#print(img.size)
#plt.imshow(img)
#plt.show()

#print(img_lr.size())   #1 3 32 32
#lrp=f(img_lr.detach())  
#print(lrp.size())    #32*32*3
#plt.imshow(lrp)
#plt.savefig('l_256.png')
#plt.show()


#print(img_hr.size())   # 1 3 128 128
#hrp=f(img_hr.detach())  
#print(hrp.size())    #128*128*3
#plt.imshow(hrp)
#plt.show()

#print(g1.size())   #1*3*128*128
#g1p=f(g1.detach())   #128*128*3
#print(g1p.size())   
#plt.imshow(g1p)
#plt.show()



#print(g2.size())   #1*3*128*128
#g2p=f(g2.detach())   #128*128*3
#print(g2p.size())   
#plt.imshow(g2p)
#plt.show()


#print(g1_recover.size())   #1*3*128*128
#g1rp=f(g1_recover.detach())   #128*128*3
#print(g1rp.size())   
#plt.imshow(g1rp)
#plt.show()

#print(g2_recover.size())   #1*3*128*128
##g2rp=f(g2_recover.detach())   #128*128*3
#print(g2rp.size())   
##plt.imshow(g2rp)
#plt.savefig('G_256.png')
##plt.show()



