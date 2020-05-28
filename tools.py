from torch import nn
import torch
from torchvision import utils
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim 
from nuaa import wangxu
import pytorch_ssim

mse = nn.MSELoss(reduction='sum')
bce = nn.BCELoss(reduction='none')




def G_loss(g1, g2, pixel_label,dl, dh,trade_off, batch_size):
    content_loss = ((mse(g1, pixel_label) + mse(g2, pixel_label)) / (10 * batch_size))

    adv_cls_loss = (torch.sum(bce(dl, dh) * trade_off) / batch_size) * 15000

    ssim_loss = pytorch_ssim.ssim(g2, pixel_label)
    ssim_loss =2 * (1000 -(ssim_loss * 1000))

    return content_loss, ssim_loss, adv_cls_loss

def D_loss1(outputs, target, factor):
    adv_loss = torch.sum(bce(outputs, target) * factor)
    return adv_loss

def D_loss2(outputs, target):
    cls_loss = torch.sum(bce(outputs, target))
    return cls_loss


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, .0, .1)
        torch.nn.init.constant_(m.bias.data, .0)





