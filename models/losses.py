import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.math_utils as pmath

class Loss_Function(torch.nn.Module):
    def __init__(self):
        super(Loss_Function, self).__init__()
    
    def forward(self, input, y):
        x = input
        return ((x - y) ** 2).sum(dim=2)

class OneHot_CrossEntropy(Loss_Function):
    def __init__(self):
        super(OneHot_CrossEntropy, self).__init__()

    def forward(self, input, y):
        x = input
        P_i = torch.nn.functional.softmax(x, dim=-1)
        loss = y * torch.log(P_i + 0.0000001)
        loss = -torch.sum(torch.sum(loss, dim=-1))
        return loss
    
class Image_Loss(Loss_Function):
    def __init__(self):
        super(Image_Loss, self).__init__()
    
    def forward(self, input, y):
        encoded_x, decoded_x = input
        P_i = torch.nn.functional.softmax(encoded_x, dim=2)
        rec_loss = y * torch.log(P_i + 0.0000001)
        rec_loss = -torch.sum(torch.sum(rec_loss, dim=2))


    
