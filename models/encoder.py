import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
import layers.hyp_layers as hyp_layers
import utils.math_utils as pmath

from geoopt import ManifoldParameter

class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x):
        output = self.layers.forward(x)
        return output


class LorentzMLP(Encoder):
    """
    HyboNet.
    """

    def __init__(self, c, args):
        super(LorentzMLP, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        # self.curvatures.append(self.c)
        c = self.curvatures
        hgc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.LorentzLinear(
                            self.manifold, in_dim, out_dim, c, args.bias, args.dropout, nonlin=act if i != 0 else None
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)

def get_encoder(args):
    assert args.num_layers > 1
    dims, acts, curvatures = hyp_layers.get_dim_act_curv(args)
    c = curvatures
    if args.encoder == 'LorentzMLP':
        manifold = getattr(manifolds, args.manifold)()
        hgc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.LorentzLinear(
                        manifold, in_dim, out_dim, c, args.bias, args.dropout, nonlin=act if i != 0 else None
                )
        )
        layers = nn.Sequential(*hgc_layers)
        return layers
    if args.encoder == 'MLP':
        hgc_layers = []
        for i in range(len(dims) - 1):
                hgc_layers.append(torch.nn.Linear(dims[i], dims[i+1]))
                act = acts[i]
                hgc_layers.append(nn.LogSigmoid())
        layers = nn.Sequential(*hgc_layers)
        return layers
    return 0