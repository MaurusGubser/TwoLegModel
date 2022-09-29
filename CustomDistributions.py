from __future__ import division, print_function

from particles.distributions import MvNormal
import numpy as np
from scipy.linalg import solve_triangular

HALFLOG2PI = 0.5 * np.log(2.0 * np.pi)


class MvNormalMissingObservations(MvNormal):
    def __init__(self, loc=0.0, scale=1.0, cov=None):
        MvNormal.__init__(self, loc=loc, scale=scale, cov=cov)

    def logpdf(self, x):
        nb_part, _ = x.shape
        assert (nb_part == 1), "x is expected to be 2-dimensional, got {} array instead".format(x.shape)
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
        return -0.5 * np.sum(z * z, axis=0) - logdet - self.dim * HALFLOG2PI
