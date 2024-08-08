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
        self.manifold = getattr(manifolds, args.sde_manifold)()
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.score_model = score_model
    def sample(self, independent=True):
        xe_t = torch.randn((self.args.num_samples, self.args.feat_dim))
        if self.manifold_name == 'Lorentz':
            x_t = torch.cat([(xe_t * xe_t + 1 / self.c).sqrt(), xe_t], dim=-1)
        else:
            x_t = 0

        time_pts = torch.linspace(1, 0, self.args.num_time_pts)
        beta = 1 # can be changed later to allow for variance scheduling
        for i in range(len(time_pts) - 1):
            t = time_pts[i]
            dt = time_pts[i + 1] - t
            if self.manifold_name == 'Lorentz':
                xe_t = x_t[..., 1:]
            fxt = -0.5 * beta * xe_t
            gt = beta ** 0.5
            score = self.score_model(torch.cat((xe_t, t.expand(xe_t.shape[0], 1)), dim=-1)).detach()
            drift = fxt - gt * gt * score
            diffusion = gt
            xe_t = xe_t + drift * dt + diffusion * torch.randn_like(xe_t) * torch.abs(dt) ** 0.5

            if self.manifold_name == 'Lorentz':
                