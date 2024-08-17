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
        return torch.mean((x - y) ** 2)

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
        x = y
        #sample time
        t = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device) * (1 - 1e-4) + 1e-4

        int_beta = 1 - t  # integral of beta, can be changed to allow for a variance shcedule instead
        mu_t = x * torch.exp(-0.5 * int_beta)
        var_t = -torch.expm1(-int_beta)
        x_t = torch.randn_like(x) * var_t ** 0.5 + mu_t
        grad_log_p = -(x_t - mu_t) / var_t
    
        xt = torch.cat((t, x_t), dim=-1) 
        score = socre_model(xt)
        loss = (score - grad_log_p) ** 2
        lmbda_t = var_t
        weighted_loss = lmbda_t * loss
        return torch.mean(weighted_loss)

class Lorentz_SDE_Loss(Loss_Function):
    def __init__(self):
        super(SDE_Loss, self).__init__()

    def forward(self, input, y):
        f = input
        x = y[..., 1:]
        #sample time
        t = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device).clamp(1e-4, 1 - 1e-4)

        # int_beta = t  # integral of beta, can be changed to allow for a variance shcedule instead
        # mu_t = x * torch.exp(-0.5 * (t))
        # var_t = -torch.expm1(-t)
        xe_t = torch.randn_like(x) * (1 - torch.exp(-(1-t))).sqrt() + x * torch.exp(-0.5 * (1-t))

        x_time = ((xe_t ** 2).sum(-1, keepdims=True) + 1.).clamp_min(1e-6).sqrt()
        x_t = torch.cat([x_time, xe_t], dim=-1)
        f_star = f(x_t, t)

        sq_norm = (torch.sum(xe_t ** 2, dim = -1, keepdim=True) + 1).clamp_min(1e-6).sqrt()
        grad_0 = xe_t / (sq_norm.clamp_min(1e-6))
        grad_rest = torch.eye(xe_t.size(1), device=xe_t.device).expand(xe_t.size(0), -1, -1)
        grad_proj = torch.cat([grad_0.unsqueeze(1), grad_rest], dim=1)

        cb_norm = sq_norm * sq_norm * sq_norm
        tr_hess_rest = torch.zeros_like(xe_t)
        tr_hess_0 = ((xe_t.size(1) - 1) * (sq_norm * sq_norm)  + 1) / cb_norm
        tr_hess = torch.cat([tr_hess_0, tr_hess_rest], dim=-1)

        a = -((xe_t - x * torch.exp(-0.5 * (1 - t))) / (torch.exp(-(1-t)) - 1)) * torch.exp((xe_t - x * torch.exp(-0.5 * (1 - t))) ** 2 / (torch.exp(-(1-t)) - 1))
        target = grad_proj.bmm((0.5 * xe_t + 2 * a).unsqueeze(dim=-1)).squeeze(dim=-1) + 0.5 * tr_hess
        # dim = -1
        # d = target.size(dim) - 1
        # uv = target * x_t
        # print(-uv.narrow(dim, 0, 1).squeeze(dim) + uv.narrow(
        #     dim, 1, d
        # ).sum(dim=dim, keepdim=False))
        # assert(1==2)
        loss = (target - f_star) ** 2
        lmbda_t = -torch.expm1(-(1-t))
        weighted_loss = lmbda_t * loss
        return torch.mean(weighted_loss)
        