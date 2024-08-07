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

class Score_Model(nn.Module):
    def __init__(self, args):
        super(Score_Model, self).__init__()
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