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
            x = torch.cat([torch.zeros_like(x)[..., 0:1], x], dim=-1)
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
        t_layer = []
        t_dims = [args.feat_dim, 16, 8, args.feat_dim - 1]
        for i in range(len(t_dims) - 2):
                t_layer.append(torch.nn.Linear(t_dims[i], t_dims[i+1]))
                t_layer.append(nn.ELU())
        t_layer.append(torch.nn.Linear(t_dims[-2], t_dims[-1]))
        self.t_layers = nn.Sequential(*t_layer )
        self.manifold = getattr(manifolds, args.sde_manifold)()
        self.manifold_name = args.sde_manifold
        self.loss_fn = getattr(Loss, args.score_loss_fn)()

    def forward(self, x, t=None):
        if self.manifold_name != 'Euclidean':
            xt = x.clone()
            t = self.t_layers(t)
            t = torch.cat([torch.zeros_like(t)[..., 0:1], t], dim=-1)
            t = self.manifold.expmap0(t)
            x = self.layers(x)
            
            # "add" time and feature embeddings
            ave = (t + x)
            denom = (-self.manifold.inner(ave, ave, keepdim=True)).abs().clamp_min(1e-8).sqrt() * self.c.sqrt()
            output = ave / denom

            output = self.manifold.logmap(xt, output)
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

        time_pts = torch.linspace(0, 1, self.args.num_time_pts, device=xe_t.device)
        # beta = 1 # can be changed later to allow for variance scheduling
        for i in trange(0, (self.args.num_time_pts - 1), desc = '[Sampling]', position = 1, leave=False):
            x_ori = x_t.clone()
            t = torch.ones(xe_t.shape[0], 1, dtype=x_t.dtype, device=x_t.device) * time_pts[i]
            dt = time_pts[i + 1] - t
            if self.manifold_name == 'Lorentz':
                xe_t = x_t[..., 1:]
            score = self.score_model(xe_t, t)
            dW = torch.randn_like(xe_t, device=xe_t.device).unsqueeze(dim=-1)            
            if self.manifold_name == 'Lorentz':
                sq_norm = (torch.sum(xe_t ** 2, dim = -1, keepdim=True) + 1).clamp_min(1e-6).sqrt()
                grad_0 = xe_t / (sq_norm.clamp_min(1e-6))
                grad_rest = torch.eye(xe_t.size(1), device=xe_t.device).expand(xe_t.size(0), -1, -1)
                grad_proj = torch.cat([grad_0.unsqueeze(1), grad_rest], dim=1)

                cb_norm = sq_norm * sq_norm * sq_norm
                tr_hess_rest = torch.zeros_like(xe_t)
                tr_hess_0 = ((xe_t.size(1) - 1) * (sq_norm * sq_norm)  + 1) / cb_norm
                tr_hess = torch.cat([tr_hess_0, tr_hess_rest], dim=-1)

                h_diffusion = grad_proj
                x_h = (grad_proj.bmm((0.5 * xe_t + 2 * score).unsqueeze(dim=-1)).squeeze(dim=-1) + 0.5 * tr_hess) * torch.abs(dt) + h_diffusion.bmm(dW).squeeze(dim=-1) * (torch.abs(dt) ** 0.5)
                # dim = -1
                # d = x_h.size(dim) - 1
                # uv = x_h * x_t
                # print(-uv.narrow(dim, 0, 1).squeeze(dim) + uv.narrow(
                #     dim, 1, d
                # ).sum(dim=dim, keepdim=False))
                # assert(1==2)
                x_t = self.manifold.expmap(x_ori, x_h)
                assert(not x_t.isnan().any())
                assert(not x_t.isinf().any())
        return x_t