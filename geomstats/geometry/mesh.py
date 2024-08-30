import torch
import math
import numpy as np
import igl
import scipy

import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric


class _Mesh(Manifold):
    """Private class for general closed manifold described by triangular mesh.
    ----------
    dim : int
        Dimension of the manifold.
    """

    def __init__(self, dim, v, f, trunc=200):
        super(_Mesh, self).__init__(
            dim=dim,
            metric=None,
            default_point_type="vector",
            default_coords_type="intrinsic",
        )
        assert dim == 3
        self.device = v.device
        self.vt = v
        self.ft = f
        self.vn = v.detach().cpu().numpy()
        self.fn = f.detach().cpu().numpy().astype(np.int32)
        self.per_face_normals = igl.per_face_normals(self.vn, self.fn, np.array([], dtype=np.float32))
        self.nt = torch.from_numpy(self.per_face_normals).float().to(self.device)

        areas = torch.tensor(igl.doublearea(self.vn, self.fn)).reshape(-1) / 2
        self.areas = areas.to(self.device)
        self.eig_val, self.eig_vec = self.get_eig(trunc=trunc)
        
    
    def project_edge(self, p, e1, e2):
        x = p - e1
        ev = e2 - e1
        r = torch.sum(x * ev, dim=-1, keepdim=True) / torch.sum(ev * ev, dim=-1, keepdim=True)
        r = r.clamp(max=1.0, min=0.0)
        projx = ev * r
        return projx + e1

    def barycenter_coordinates(self, p, a, b, c):
        """Assumes inputs are (N, D).
        Follows https://ceng2.ktu.edu.tr/~cakir/files/grafikler/Texture_Mapping.pdf
        """
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = torch.sum(v0 * v0, dim=-1)
        d01 = torch.sum(v0 * v1, dim=-1)
        d11 = torch.sum(v1 * v1, dim=-1)
        d20 = torch.sum(v2 * v0, dim=-1)
        d21 = torch.sum(v2 * v1, dim=-1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return torch.stack([u, v, w], dim=-1)

    def closest_point(self, p):
        """Returns the point on the mesh closest to the query point p.
        Algorithm follows https://www.youtube.com/watch?v=9MPr_XcLQuw&t=204s.

        Inputs:
            p : (#query, 3)
            v : (#vertices, 3)
            f : (#faces, 3)

        Return:
            A projected tensor of size (#query, 3) and an index (#query,) indicating the closest triangle.
        """
        orig_p = p.reshape(-1, 3)

        nq = p.shape[0]
        nf = self.ft.shape[0]

        vs = self.vt[self.ft]
        a, b, c = vs[:, 0], vs[:, 1], vs[:, 2]

        n = self.nt.reshape(1, nf, 3)
        # p = p.reshape(nq, 1, 3)
        p = p.reshape(-1, 1, 3)

        a = a.reshape(1, nf, 3)
        b = b.reshape(1, nf, 3)
        c = c.reshape(1, nf, 3)

        # project onto the plane of each triangle
        p = p + (n * (a - p)).sum(-1, keepdim=True) * n

        # if barycenter coordinate is negative,
        # then point is outside of the edge on the opposite side of the vertex.
        bc = self.barycenter_coordinates(p, a, b, c)

        # for each outside edge, project point onto edge.
        p = torch.where((bc[..., 0] < 0)[..., None], self.project_edge(p, b, c), p)
        p = torch.where((bc[..., 1] < 0)[..., None], self.project_edge(p, c, a), p)
        p = torch.where((bc[..., 2] < 0)[..., None], self.project_edge(p, a, b), p)

        # compute distance to all points and take the closest one
        fidx = torch.argmin(torch.linalg.norm(orig_p[:, None] - p, dim=-1), dim=-1)
        p_idx = torch.vmap(lambda p_, idx_: torch.index_select(p_, 0, idx_))(
            p, fidx.reshape(-1, 1)
        ).reshape(-1, 3)
        return p_idx, fidx

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        proj = self.closest_point(point)[0]
        return torch.isclose(point, proj, atol).all(dim=-1)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        fidx = self.closest_point(base_point)[1]
        normals = self.nt[fidx]
        inner_prod = self.metric.inner_product(vector, normals)
        return gs.isclose(inner_prod, torch.zeros_like(inner_prod), atol)

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        # TODO: how to compute normal from base_ponit with continuous grad?
        with torch.no_grad():
            fidx = self.closest_point(base_point)[1]
            normals = self.nt[fidx]
        coeff = self.metric.inner_product(vector, normals)
        tangent_vec = vector - gs.einsum("...,...j->...j", coeff, normals)
        return tangent_vec

    def projection(self, point):
        """Project a point on the mesh.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in embedding Euclidean space.

        Returns
        -------
        projected_point : array-like, shape=[..., dim]
            Point projected on the mesh.
        """
        projected_point = self.closest_point(point)[0]
        return projected_point

    def random_point(self, n_samples=1, bound=1.0, device='cpu'):
        """Sample random points on the manifold.

        If the manifold is compact, a uniform distribution is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        samples : array-like, shape=[..., {dim, [n, n]}]
            Points sampled on the hypersphere.
        """
        return self.random_uniform(n_samples, device)

    def sample_simplex_uniform(self, K, shape=(), dtype=torch.float32, device="cpu"):
        x = torch.sort(torch.rand(shape + (K,), dtype=dtype, device=device))[0]
        x = torch.cat(
            [
                torch.zeros(*shape, 1, dtype=dtype, device=device),
                x,
                torch.ones(*shape, 1, dtype=dtype, device=device),
            ],
            dim=-1,
        )
        diffs = x[..., 1:] - x[..., :-1]
        return diffs

    def random_uniform(self, n_samples=1, device='cpu'):
        """Sample in the mesh from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., dim]
            Points sampled on the mesh.
        """
        fidx = torch.multinomial(self.areas, n_samples, replacement=True)
        barycoords = self.sample_simplex_uniform(
            2, (n_samples,), dtype=self.vt.dtype, device=device
        )
        return torch.sum(self.vt[self.ft[fidx]] * barycoords[..., None], axis=1)

    def random_normal_tangent(self, base_point, n_samples=1):
        """Sample in the tangent space from the standard normal distribution.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        ambiant_noise = torch.randn((n_samples, self.dim), device=base_point.device)
        return self.to_tangent(vector=ambiant_noise, base_point=base_point)

    def get_eig(self, trunc):
        l = -igl.cotmatrix(self.vn, self.fn)
        m = igl.massmatrix(self.vn, self.fn, igl.MASSMATRIX_TYPE_VORONOI)
        # generalized eigenvalue problem
        eig_val, eig_vec = scipy.sparse.linalg.eigsh(
            l, trunc+1, m, sigma=0, which="LM", maxiter=100000
        )
        # Only the non-zero eigenvalues and its eigenvectors
        eig_val, eig_vec = eig_val[1:trunc+1], eig_vec[:,1:trunc+1]

        # Should we normalize the eigenfunction?
        # eig_vec = eig_vec / np.sqrt((eig_vec**2).sum(0))

        eig_val = torch.from_numpy(eig_val).float().to(self.device)
        eig_vec = torch.from_numpy(eig_vec).float().to(self.device)
        return eig_val, eig_vec

    def eig_fn(self, x):
        fidx = self.closest_point(x)[1]
        vfx = self.vt[self.ft[fidx]]
        vfx_a, vfx_b, vfx_c = vfx[..., 0, :], vfx[..., 1, :], vfx[..., 2, :]
        bc_x = self.barycenter_coordinates(x, vfx_a, vfx_b, vfx_c)[..., None]  # (N, 3, 1)

        # compute interpolated eigenfunction
        eigfns = torch.sum(bc_x * self.eig_vec[self.ft[fidx]], dim=-2)

        return eigfns

    @property
    def log_volume(self):
        return torch.log(self.areas.sum()).detach().cpu().numpy()


class MeshMetric(RiemannianMetric):
    """Class for the Mesh Metric.

    Parameters
    ----------
    dim : int
        Dimension of the Mesh.
    """

    def __init__(self, dim, v, f, trunc):
        super(MeshMetric, self).__init__(dim=dim, signature=(dim, 0))
        self.embedding_metric = EuclideanMetric(dim)
        self._space = _Mesh(dim=dim, v=v, f=f, trunc=trunc)

    def metric_matrix(self, base_point=None):
        """Metric matrix at the tangent space at a base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        return gs.eye(self.dim)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., dim], optional
            Point on the mesh.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        inner_prod = self.embedding_metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        ).float()

        return inner_prod

    def squared_norm(self, vector, base_point=None):
        """Compute the squared norm of a vector.

        Squared norm of a vector associated with the inner-product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector on the tangent space of the mesh at base point.
        base_point : array-like, shape=[..., dim], optional
            Point on the mesh.

        Returns
        -------
        sq_norm : array-like, shape=[..., 1]
            Squared norm of the vector.
        """
        sq_norm = self.embedding_metric.squared_norm(vector)
        return sq_norm

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., dim]
            Point on the mesh.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point on the mesh equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        # First project the vector to the tangent space
        fidx = self._space.closest_point(base_point)[1]
        normals = self._space.nt[fidx]
        coeff = self.inner_product(tangent_vec, normals)
        tangent_vec = tangent_vec - gs.einsum("...,...j->...j", coeff, normals)
        # NOTE: approx. exp using projection
        return self._space.projection(base_point + tangent_vec)

    def log(self, point, base_point, **kwargs):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point on the mesh.
        base_point : array-like, shape=[..., dim]
            Point on the mesh.

        Returns
        -------
        log : array-like, shape=[..., dim]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        raise NotImplementedError(f'Mesh: log not implemented.')

    def grad(self, func):
        return self.embedding_metric.grad(func)


class Mesh(_Mesh):
    def __init__(self, dim, v, f, trunc=200):
        super(Mesh, self).__init__(dim=dim, v=v, f=f, trunc=trunc)
        self.metric = MeshMetric(dim, v, f, trunc)
