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


class SDE_Loss(Loss_Function):
    def __init__(self):
        super(SDE_Loss, self).__init__()

    def forward(self, input, y):
        socre_model = input
        x = y[..., 1:]
        #sample time
        t = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device) * (1 - 1e-4) + 1e-4

        int_beta = t  # integral of beta, can be changed to allow for a variance shcedule instead
        mu_t = x * torch.exp(-0.5 * int_beta)
        var_t = -torch.expm1(-int_beta)
        x_t = torch.randn_like(x) * var_t ** 0.5 + mu_t
        grad_log_p = -(x_t - mu_t) / var_t
    
        xt = torch.cat((x_t, t), dim=-1) 
        score = socre_model(xt)

        loss = (score - grad_log_p) ** 2
        lmbda_t = var_t
        weighted_loss = lmbda_t * loss
        return torch.mean(weighted_loss)