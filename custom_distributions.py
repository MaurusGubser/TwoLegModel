from __future__ import division, print_function

import warnings
from particles.distributions import ProbDist, MvNormal
from collections import OrderedDict  # see prior
import numpy as np
import numpy.random as random
import scipy.stats as stats
from scipy.linalg import solve_triangular, inv, cholesky

HALFLOG2PI = 0.5 * np.log(2. * np.pi)


class MvNormalMultiDimCov(ProbDist):
    """Multivariate Normal distribution with change of shapes.

    Parameters
    ----------
    loc: (Np, d) ndarray
        location parameter (see below)
    scale: ndarray
        scale parameter (see below)
    cov: (Np, d, d) ndarray
        covariance matrix (see below)

    Note
    ----
    The parametrisation used here is slightly unusual. In short,
    the following line::

        x = dists.MvNormal(loc=m, scale=s, cov=Sigma).rvs(size=30)

    is equivalent to::

        x = m + s * dists.MvNormal(cov=Sigma).rvs(size=30)

    The idea is that they are many cases when we may want to pass
    varying means and scales (but a fixed correlation matrix).

    dx (dimension of vectors x) is determined by matrix cov; for rvs,
    size must be (N, ), otherwise an error is raised.

    Notes:
    * if du<dx, fill the remaining dimensions by location
        (i.e. scale should be =0.)
    * cov does not need to be a correlation matrix; more generally
    > mvnorm(loc=x, scale=s, cor=C)
    correspond to N(m,diag(s)*C*diag(s))

    In addition, note that x and s may be (N, d) vectors;
    i.e for each n=1...N we have a different mean, and a different scale.
    """

    def __init__(self, loc=0., scale=1., cov=None):
        self.loc = loc
        self.scale = scale
        self.cov = cov
        err_msg = 'mvnorm: argument cov must be a Nxdxd ndarray, with d>1, defining a symmetric positive matrix'
        try:
            self.L = np.linalg.cholesky(cov)  # L*L.T = cov
            self.halflogdetcor = np.array([np.sum(np.log(np.diag(l))) for l in self.L])
        except:
            raise ValueError(err_msg)
        assert cov[0, :, :].shape == (self.dim, self.dim), err_msg

    @property
    def dim(self):
        return self.cov.shape[1]

    def linear_transform(self, z):
        z_transformed = np.empty_like(z)
        for i in range(0, z.shape[0]):
            z_transformed[i, :] = self.loc[i] + self.scale * np.dot(z[i], self.L[i].T)
        return z_transformed

    def logpdf(self, x):
        z_solved = np.empty_like(x.T)
        for i in range(0, x.shape[0]):
            z_solved[:, i] = solve_triangular(self.L[i], np.transpose((x - self.loc)[i] / self.scale), lower=True)
        # z is dxN, not Nxd
        if np.asarray(self.scale).ndim == 0:
            # handle as array of dimension N? To handle with logdet...
            logdet = self.dim * np.log(self.scale)
        else:
            # This case is not adapted to cov (N, d, d) form
            logdet = np.sum(np.log(self.scale), axis=-1)
        logdet += self.halflogdetcor
        return - 0.5 * np.sum(z_solved * z_solved, axis=0) - logdet - self.dim * HALFLOG2PI

    def rvs(self, size=None):
        if size is None:
            sh = np.broadcast(self.loc, self.scale).shape
            # sh=() when both loc and scale are scalars
            N = 1 if len(sh) == 0 else sh[0]
        else:
            N = size
        z = stats.norm.rvs(size=(N, self.dim))
        return self.linear_transform(z)

    def ppf(self, u):
        """
        Note: if dim(u) < self.dim, the remaining columns are filled with 0
        Useful in case the distribution is partly degenerate
        """
        N, du = u.shape
        if du < self.dim:
            z = np.zeros((N, self.dim))
            z[:, :du] = stats.norm.ppf(u)
        else:
            z = stats.norm.ppf(u)
        return self.linear_transform(z)

    def posterior(self, x, Sigma=None):
        """Posterior for model: X1, ..., Xn ~ N(theta, Sigma).

        Parameters
        ----------
        x: (n, d) ndarray
            data
        Sigma: (d, d) ndarray
            (fixed) covariance matrix in the model
        """
        n = x.shape[0]
        Sigma = np.eye(self.dim) if Sigma is None else Sigma
        Siginv = inv(Sigma)
        Qpost = inv(self.cov) + n * Siginv
        Sigpost = inv(Qpost)

        mupost = (np.matmul(Siginv, self.loc) +
                  np.matmul(Siginv, np.sum(x, axis=0)))

        return MvNormalMultiDimCov(loc=mupost, cov=Sigpost)


class MvStudent(ProbDist):

    def __init__(self, loc=0., shape=1, df=1.0):
        self.loc = loc
        self.locs = [loc[i, :] for i in range(0, loc.shape[0])]
        self.shape = shape
        self.df = df

    @property
    def dim(self):
        return self.df.shape[1]

    def logpdf(self, x):
        nb_particles, _ = x.shape
        return np.array(
            [stats.multivariate_t(loc=self.locs[i], shape=self.shape, allow_singular=True).logpdf(x[i, :]) for i in
             range(0, nb_particles)])

    def rvs(self, size=1):
        return np.array(
            [stats.multivariate_t(loc=mu, shape=self.shape, allow_singular=True).rvs(size=1) for mu in self.locs])

    def ppf(self, u):
        warnings.warn('Using ppf of MvStudent', UserWarning)
        pass


class MvNormalMissingObservations(MvNormal):

    def __init__(self, loc=0., scale=1., cov=None):
        MvNormal.__init__(self, loc=loc, scale=scale, cov=cov)

    def logpdf(self, x):
        nb_part, _ = x.shape
        assert nb_part == 1, 'x is expected to be 2-dimensional, got {} array instead'.format(x.shape)
        mask_not_nan = np.invert(np.isnan(x))[0]
        x_not_nan = x[:, mask_not_nan]
        loc_not_nan = self.loc[:, mask_not_nan]
        mask_2d = np.outer(mask_not_nan, mask_not_nan)
        nb_non_nan = np.sum(mask_not_nan)
        L_not_nan = np.reshape(self.L[mask_2d], (nb_non_nan, nb_non_nan))

        z = solve_triangular(L_not_nan, np.transpose((x_not_nan - loc_not_nan) / self.scale), lower=True)
        # z is dxN, not Nxd
        if np.asarray(self.scale).ndim == 0:
            logdet = self.dim * np.log(self.scale)
        else:
            logdet = np.sum(np.log(self.scale), axis=-1)
        logdet += self.halflogdetcor
        return - 0.5 * np.sum(z * z, axis=0) - logdet - self.dim * HALFLOG2PI
