from __future__ import division, print_function

import itertools
import numpy as np
from numpy import random
from scipy import stats
from scipy.linalg import cholesky, LinAlgError

import particles
from particles import smc_samplers as ssp
from particles.state_space_models import Bootstrap
from particles import utils
from particles.mcmc import PMMH
from particles.collectors import Moments, LogLts


class TruncatedPMMH(PMMH):
    def __init__(self, niter=10, verbose=0, ssm_cls=None,
                 smc_cls=particles.SMC, prior=None, data=None, smc_options=None,
                 fk_cls=Bootstrap, Nx=100, theta0=None, adaptive=True, scale=1.,
                 rw_cov=None, t_start=0):
        """
        Parameters
        ----------
        niter: int
            number of iterations
        verbose: int (default=0)
            print some info every `verbose` iterations (never if 0)
        ssm_cls: StateSpaceModel class
            the considered parametric class of state-space models
        smc_cls: class (default: particles.SMC)
            SMC class
        prior: StructDist
            the prior
        data: list-like
            the data
        smc_options: dict
            options to pass to class SMC
        fk_cls: (default=Bootstrap)
            FeynmanKac class associated to the model
        Nx: int
            number of particles (for the particle filter that evaluates the
            likelihood)
        theta0: structured array of length=1
            starting point (generated from prior if =None)
        adaptive: bool
            whether to use the adaptive version
        scale: positive scalar (default = 1.)
            in the adaptive case, covariance of the proposal is scale^2 times
            (2.38 / d) times the current estimate of the target covariance
        rw_cov: (d, d) array
            covariance matrix of the random walk proposal (set to I_d if None)
        t_start: int
            start time, on which likelihood is conditioned, i.e. p(y_{t_start:T}|y_{0:t_start})
        """
        self.t_start = t_start
        PMMH.__init__(self, niter=niter, verbose=verbose, ssm_cls=ssm_cls,
                      smc_cls=smc_cls, prior=prior, data=data, smc_options=smc_options,
                      fk_cls=fk_cls, Nx=Nx, theta0=theta0, adaptive=adaptive, scale=scale,
                      rw_cov=rw_cov)
        self.smc_options.update({'collect': [LogLts()]})

    def alg_instance(self, theta):
        return self.smc_cls(fk=self.fk_cls(ssm=self.ssm_cls(**theta), data=self.data), N=self.Nx,
                            **self.smc_options)

    def compute_post(self):
        self.prop.lpost[0] = self.prior.logpdf(self.prop.theta)
        if np.isfinite(self.prop.lpost[0]):
            pf = self.alg_instance(ssp.rec_to_dict(self.prop.theta[0]))
            pf.run()
            self.prop.lpost[0] += pf.summaries.logLts[-1] - pf.summaries.logLts[self.t_start]
