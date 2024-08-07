import math
import manifolds
import torch
import torch.nn as nn
import torch.nn.functional as F
import layers.hyp_layers as hyp_layers

class Decoder(nn.Module):

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x):
        probs = self.layers.forward(x)
        return probs
    
class LorentzMLP(Decoder):
    """
    HyboNet.
    """

    def __init__(self, c, args):
        super(LorentzMLP, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        dims.reverse()
        acts.reverse()
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

def get_decoder(args):
    manifold = getattr(manifolds, args.manifold)()
    assert args.num_layers > 1
    dims, acts, curvatures = hyp_layers.get_dim_act_curv(args)
    dims.reverse()
    acts.reverse()
    c = curvatures
    if args.decoder == 'LorentzMLP':
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
    return 0