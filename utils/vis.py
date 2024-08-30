import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cartopy
import cartopy.crs as ccrs
import pyvista as pv
import warnings; warnings.filterwarnings("ignore", category=UserWarning)

from torch.distributions.multivariate_normal import MultivariateNormal
from geomstats.geometry.special_orthogonal import _SpecialOrthogonal3Vectors
from geomstats.geometry.hyperbolic import Hyperbolic

def make_disk_grid(dim, npts, eps=1e-3, device='cpu'):
    poincare_ball = Hyperbolic(dim=dim, default_coords_type="ball")
    R=1.0
    R = R - eps
    bp = torch.linspace(-R, R, npts, device=device)
    xx, yy = torch.meshgrid((bp, bp))
    twodim = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    mask = torch.linalg.norm(twodim, axis=-1) < 1.0 - eps
    lambda_x = poincare_ball.metric.lambda_x(twodim) ** 2 * mask.float()
    volume = (2 * R) ** dim
    return twodim, volume, lambda_x

def plot_hyperbolic(test_ds, log_prob=None, npts=150):
    if test_ds.manifold.coords_type == 'extrinsic': #Hyperboloid
        coord_map = Hyperbolic._ball_to_extrinsic_coordinates
    else: #Poincare ball
        coord_map = lambda x: x
        
    size=10
    device = test_ds.device
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(0.6 * size, 0.6 * size),
        sharex=False,
        sharey=True,
        tight_layout=True,
    )
    cmap = sns.cubehelix_palette(
        light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
    )
    xs, volume, lambda_x = make_disk_grid(test_ds.manifold.dim, npts, 1e-2, device)
    idx = torch.nonzero(lambda_x).squeeze()
    ys = xs[idx]
    idx = idx.detach().cpu().numpy()
    lambda_x = lambda_x.detach().cpu().numpy()[idx]
    ys = coord_map(ys)
    xs = xs.detach().cpu().numpy()

    for i, log_prob_fn in enumerate([test_ds.log_prob, log_prob]):
        if log_prob_fn is None:
            continue
        prob = np.exp(log_prob_fn(ys))
        # print(f"{prob.min():.4f} | {prob.mean():.4f} | {prob.max():.4f}")
        idx_not_nan = np.nonzero(~np.isnan(prob))[0]
        nb = len(np.nonzero(np.isnan(prob))[0])
        tot = prob.shape[0]
        # print(f"prop nan in prob: {nb / tot * 100:.1f}%")

        measure = np.zeros((npts * npts))
        measure[idx] = prob * lambda_x

        xs = xs.reshape(npts, npts, 2)
        measure = measure.reshape(npts, npts)
        ax[i].pcolormesh(
            xs[:, :, 0],
            xs[:, :, 1],
            measure,
            cmap=cmap,
            linewidth=0,
            rasterized=True,
            shading="gouraud",
        )   
        ax[i].set_xlim([-1.01, 1.01])
        ax[i].set_ylim([-1.01, 1.01])

        ax[i].add_patch(Circle((0, 0), 1.0, color="black", fill=False, linewidth=2, zorder=10))
        ax[i].set_aspect("equal")
        ax[i].axis("off")
    plt.close(fig)
    return fig