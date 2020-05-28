import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from os.path import join
import numpy as np
import os


def transform_hl_pair(hr_height, hr_width):

    tf = [  transforms.ToTensor(),                            
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  

    
    return transforms.Compose(tf)


#def tf():

    
    #transforms = [                                                                                        V 2.0
    #                 transforms.ToTensor(),                            #[0 1]
    #                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  #[-1 1]

    
    #return transforms.Compose(transforms)






def arrange_data(path):
    flags=[]
    for child_dir in os.listdir(path):
        flags.append(child_dir)


    #path = np.array(data)[flags].tolist()
    path=flags
    return path





class WIDER(Dataset):

    def __init__(self, base1, base2, base3, base4, base5, base6, path1, path2, path3, path4, path5, path6, high_resolution=(128, 128)):
        self.base1 = base1
        self.base2 = base2
        self.base3 = base3
        self.base4 = base4
        self.base5 = base5
        self.base6 = base6
        self.path1 = path1
        self.path2 = path2
        self.path3 = path3
        self.path4 = path4
        self.path5 = path5
        self.path6 = path6
        self.lr = transform_hl_pair(*high_resolution)
        #self.ft = tf()
    def __len__(self):
        return len(self.path1)

    def __getitem__(self, idx):
        carl = Image.open(join(self.base1, self.path1[idx]))
        carh = Image.open(join(self.base2, self.path2[idx]))
        vanl = Image.open(join(self.base3, self.path3[idx]))
        vanh = Image.open(join(self.base4, self.path4[idx]))
        bgl = Image.open(join(self.base5, self.path5[idx]))
        bgh = Image.open(join(self.base6, self.path6[idx]))
        
        carl= carl.convert('RGB')
        carh= carh.convert('RGB')
        bgl= bgl.convert('RGB')
        bgh= bgh.convert('RGB')
        vanl=vanl.convert('RGB')
        vanh=vanh.convert('RGB')

        return {"lr_car": self.lr(carl), "lr_van": self.lr(vanl),
                "hr_car": self.lr(carh), "hr_van": self.lr(vanh),
                "lr_background": self.lr(bgl), "hr_background": self.lr(bgh)}









