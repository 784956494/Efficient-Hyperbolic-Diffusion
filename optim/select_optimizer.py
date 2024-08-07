from geoopt import ManifoldParameter
import torch
from optim.radam import RiemannianAdam
from models.autoencoder import Autoencoder
class Optimizer(object):
    def __init__(self, model, euc_lr, hyp_lr, euc_weight_decay, hyp_weight_decay):
        euc_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and not isinstance(p, ManifoldParameter)]

        hyp_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and isinstance(p, ManifoldParameter)]
        
        optimizer_euc = torch.optim.Adam(euc_params, lr=euc_lr, weight_decay=euc_weight_decay)
        optimizer_hyp = RiemannianAdam(hyp_params, lr=hyp_lr, stabilize=10, weight_decay=hyp_weight_decay)
        self.optimizer = [optimizer_euc, optimizer_hyp]
    def step(self):
        for optimizer in self.optimizer:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizer:
            optimizer.zero_grad()

def select(args, model):
    optimizer = None
    lr_scheduler = None
    if args.manifold == 'Eulicdean' or isinstance(model, Autoencoder):
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    else:
        optimizer = Optimizer(model, args.lr, args.hyp_lr, args.weight_decay, args.hyp_weight_decay)
    
    if args.lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=int(
                                                       args.lr_reduce_freq),
                                                   gamma=float(args.gamma))
    return (optimizer, lr_scheduler)