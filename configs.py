import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (1e-4, 'learning rate'),
        'hyp-lr':(1e-4, 'learning rate for hyperbolic optimizer'),
        'dropout': (0.0, 'dropout probability'),
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'hyp-weight-decay': (0., 'l2 regularization strength for hyperbolic optimizer'),
        'optimizer': ('radam', 'which optimizer to use, can be any of [rsgd, radam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (100, 'patience for early stopping'),
        'seed': (1234, 'seed for training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': ('log/run/', 'path to save training logs and model weights'),
        'checkpoint_path':('mnist', 'path to save the checkpoint model'),
        'sweep-c': (0, ''),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (1, 'do not early stop before min-epochs'),
        'lr_scheduler': (False, 'whether to use learning rate scheduler'),
        'train_save_freq':(10, 'how often to save the model'),
        'encoder_loss':(False, 'is the autoencoder loss dependent on the encoded embedding')
    },
    'model_config': {
        'task': ('nc', 'which tasks to train on, can be any of [lp, nc]'),
        'model': ('GCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HyperGCN, HyboNet]'),
        'encoder':('LorentzMLP', 'which encoder to use'),
        'decoder':('LorentzMLP', 'which decoder to use'),
        'dim': (128, 'embedding dimension'),
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall, Lorentz]'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'margin': (2., 'margin of MarginLoss'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('None', 'which activation function to use (or None for no activation)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'double-precision': ('0', 'whether to use double precision'),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not'),
        'AE_loss_fn':('OneHot_CrossEntropy', 'what function to use to the autoencoder'),
        'score_loss_fn':('SDE_Loss', 'objective function for score model'),
        'diffusion-type':('Lorentz', 'hyperbolic or Euclidean diffusion to run'),
        'ae_path':('Ae', 'path to the autoencoder model'),
        'sde_manifold': ('Euclidean', 'whcih manifold to use for score training'),
        'ema':(0.999, 'decay for exponential moving average'),
        'use_ema':(False, 'whether to use moving average for sampling'),
        'num-samples':(1000, 'the number of samples to generate'),
        'num-time-pts': (1000, 'the number of time steps during sampling')
    },
    'data_config': {
        'dataset': ('mnist', 'which dataset to use'),
        'feat-dim':(784, 'the dimension of the dataset'),
        'batch-size':(32, 'batch size to train for data'),
        'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'use-feats': (1, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
