import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import manifolds
import models.encoder as Encoders
import models.decoder as Decoders
import models.losses as Loss
import utils.math_utils as pmath
from tqdm import trange

class Autoencoder(nn.Module):
    def __init__(self, args):
        super(Autoencoder, self).__init__()
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        if args.manifold in ['Lorentz', 'Hyperboloid']:
            args.feat_dim = args.feat_dim + 1
        self.encoder = Encoders.get_encoder(args)
        self.decoder = Decoders.get_decoder(args)
        self.manifold = getattr(manifolds, args.manifold)()
        self.loss_fn = getattr(Loss, args.AE_loss_fn)()

    def forward(self, x):
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
            x = self.manifold.expmap0(x)
        encoded_emb = self.encoder(x)
        decoded_emb = self.decoder(encoded_emb)
        decoded_emb = self.manifold.logmap0(decoded_emb)[..., 1:]
        output = (encoded_emb, decoded_emb)
        return output

class Score_Model(nn.Module):
    def __init__(self, args):
        super(Score_Model, self).__init__()
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        #account for time as well
        args.feat_dim = args.feat_dim + 1

        self.layers = Encoders.get_encoder(args)
        self.manifold = getattr(manifolds, args.sde_manifold)()
        self.manifold_name = args.sde_manifold
        self.loss_fn = getattr(Loss, args.score_loss_fn)()

    def forward(self, x):
        if self.manifold_name != 'Euclidean':
            xt = x.clone()
            x = self.layers(x)
            output = self.manifold.logmap(xt, x, self.c)
        else:
            output = self.layers(x)
        return output

class Sampler(object):
    def __init__(self, args, score_model):
        super(Sampler, self).__init__()
        self.args = args
        self.device = args.device
        self.manifold_name = args.manifold
        self.manifold = getattr(manifolds, args.manifold)()
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.score_model = score_model
    def sample(self, independent=True):
        xe_t = torch.randn((self.args.num_samples, self.args.dim), device=self.args.device)
        if self.manifold_name == 'Lorentz':
            x_time = ((xe_t ** 2).sum(-1, keepdims=True) + self.c.reciprocal()).clamp_min(1e-6).sqrt()
            x_t = torch.cat([x_time, xe_t], dim=-1)
        else:
            x_t = 0

        time_pts = torch.linspace(1, 0, self.args.num_time_pts, device=xe_t.device)
        beta = 1 # can be changed later to allow for variance scheduling
        for i in trange(0, (self.args.num_time_pts), desc = '[Sampling]', position = 1, leave=False):
            x_ori = x_t
            t = time_pts[i]
            dt = time_pts[i + 1] - t
            if self.manifold_name == 'Lorentz':
                xe_t = x_t[..., 1:]
            # fxt = -0.5 * beta * xe_t
            # gt = beta ** 0.5
            score = self.score_model(torch.cat((xe_t, t.expand(xe_t.shape[0], 1)), dim=-1))
            # drift = fxt - gt * gt * score
            # diffusion = gt
            # dW = torch.randn_like(xe_t, device=xe_t.device)
            # xe_t = xe_t + drift * dt + diffusion * dW * torch.abs(dt) ** 0.5
            
            if self.manifold_name == 'Lorentz':
                sq_norm = torch.sum(xe_t ** 2, dim = -1, keepdim=True)
                grad_0 = xe_t / sq_norm
                grad_rest = torch.eye(xe_t.size(1), device=xe_t.device).expand(xe_t.size(0), -1, -1)
                grad_proj = torch.cat([grad_0.unsqueeze(1), grad_rest], dim=1)

                cb_norm = torch.pow(sq_norm + 1, 1.5)
                I = torch.eye(xe_t.size(1), device=xe_t.device).expand(xe_t.size(0), -1, -1)
                outer = torch.bmm(xe_t.unsqueeze(2), xe_t.unsqueeze(1))
                hess_0 = I / torch.sqrt(1 + sq_norm).unsqueeze(dim=-1)
                hess_0 = hess_0 - outer / cb_norm.unsqueeze(dim=-1)
                hess = torch.zeros(xe_t.size(0), xe_t.size(1) + 1, xe_t.size(1), xe_t.size(1), device = xe_t.device)
                hess[:, 0, :, :]  = hess_0
                h_drift = grad_proj.bmm((0.5 * xe_t + 2 * score).unsqueeze(dim=-1)).squeeze(dim=-1) + torch.diagonal(hess, offset=0, dim1=-2, dim2=-1).sum(dim=-1)
                # x_t = h_drift * dt + grad_proj.bmm(dW.unsqueeze(dim=-1)).squeeze(dim=-1) * (torch.abs(dt) ** 0.5)
                h_diffusion = (grad_proj * grad_proj).sum(dim=-1)
                dW = torch.randn_like(x_t, device = x_t.device)
                x_h = h_drift * dt + h_diffusion * dW * (torch.abs(dt) ** 0.5)
                print("lalalalla")
                print(self.manifold.inner(x_h, x_ori))
                torch.cuda.empty_cache()
                assert(1==2)
        return x_t