import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import particles
from collections import OrderedDict

from particles import distributions as dists
from particles.collectors import Moments, LogLts
from particles import state_space_models as ssm
from particles import mcmc
from particles import smc_samplers as ssp

from DataReaderWriter import DataReaderWriter
from TwoLegSMCModel import TwoLegModel
from Plotter import Plotter
from CustomMCMC import TruncatedPMMH


def set_prior(add_Q, add_H, add_legs, add_imus, add_alphas):
    prior_dict = {}
    if add_Q:
        prior_Q = {'scale_x': dists.LinearD(dists.InvGamma(3.0, 2.0), a=100.0, b=0.0),
                   'scale_y': dists.LinearD(dists.InvGamma(3.0, 2.0), a=100.0, b=0.0),
                   'scale_phi': dists.LinearD(dists.InvGamma(3.0, 2.0), a=250.0, b=0.0)}
        prior_Q = {'scale_x': dists.Uniform(20.0, 200.0),
                   'scale_y': dists.Uniform(20.0, 200.0),
                   'scale_phi': dists.Uniform(50.0, 500.0)}
        prior_dict.update(prior_Q)
    if add_H:
        prior_H = {'sigma_imu_acc': dists.LinearD(dists.InvGamma(3.0, 2.0), a=0.1, b=0.0),
                   'sigma_imu_gyro': dists.LinearD(dists.InvGamma(3.0, 2.0), a=0.01, b=0.0),
                   'sigma_press_velo': dists.LinearD(dists.InvGamma(3.0, 2.0), a=0.1, b=0.0),
                   'sigma_press_acc': dists.LinearD(dists.InvGamma(3.0, 2.0), a=1000.0, b=0.0)}
        prior_H = {'sigma_imu_acc': dists.Uniform(0.0001, 0.01),
                   'sigma_imu_gyro': dists.Uniform(0.0001, 0.01),
                   'sigma_press_velo': dists.Uniform(0.0001, 0.01),
                   'sigma_press_acc': dists.Uniform(0.001, 0.1)}
        prior_dict.update(prior_H)
    if add_legs:
        prior_legs = {'femur_left': dists.Uniform(0.3, 0.7), 'fibula_left': dists.Uniform(0.4, 0.8),
                      'femur_right': dists.Uniform(0.3, 0.7), 'fibula_right': dists.Uniform(0.4, 0.8)}
        prior_dict.update(prior_legs)
    if add_imus:
        prior_imus = {'pos_imu0': dists.TruncNormal(mu=0.25, sigma=0.3, a=0.0, b=0.5),
                      'pos_imu1': dists.TruncNormal(mu=0.3, sigma=0.3, a=0.0, b=0.6),
                      'pos_imu2': dists.TruncNormal(mu=0.25, sigma=0.3, a=0.0, b=0.5),
                      'pos_imu3': dists.TruncNormal(mu=0.3, sigma=0.3, a=0.0, b=0.6)}
        prior_imus = {'pos_imu0': dists.TruncNormal(mu=0.3, sigma=0.3, a=0.0, b=0.5)}
        prior_dict.update(prior_imus)
    if add_alphas:
        prior_alphas = {'alpha_0': dists.TruncNormal(mu=0.0, sigma=0.5, a=-1.57, b=1.57),
                        'alpha_1': dists.TruncNormal(mu=0.0, sigma=0.5, a=-1.57, b=1.57),
                        'alpha_2': dists.TruncNormal(mu=0.0, sigma=0.5, a=-1.57, b=1.57),
                        'alpha_3': dists.TruncNormal(mu=0.0, sigma=0.5, a=-1.57, b=1.57)}
        prior_alphas = {'alpha_0': dists.TruncNormal(mu=0.0, sigma=1.0, a=-2.0, b=2.0),
                        'alpha_2': dists.TruncNormal(mu=0.0, sigma=1.0, a=-2.0, b=2.0)}
        prior_dict.update(prior_alphas)
    return prior_dict, dists.StructDist(prior_dict)


def prepare_data(generation_type, max_timesteps, dim_observations):
    path_truth = 'GeneratedData/' + generation_type + '/truth.dat'
    path_obs = 'GeneratedData/' + generation_type + '/noised_observations.dat'
    data_reader = DataReaderWriter()
    data_reader.read_states_as_arr(path_truth, max_timesteps=max_timesteps)
    data_reader.read_observations_as_arr(path_obs, max_timesteps=max_timesteps)
    data_reader.prepare_lists()
    states = data_reader.states_list
    observations = data_reader.observations_list
    if dim_observations == 20:
        observations = [obs[:, (0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34)] for obs in
                        observations]
    return states, observations


def learn_model_parameters(prior_dict, my_prior, learning_alg, Nx, N, t_start, niter):
    if learning_alg == 'pmmh':
        alg = mcmc.PMMH(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.GuidedPF,
                        smc_options={'ESSrmin': 0.5}, data=y, Nx=Nx, niter=niter, verbose=niter,
                        adaptive=True, scale=1.0)
    elif learning_alg == 'cpmmh':
        alg = TruncatedPMMH(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.GuidedPF,
                            smc_options={'ESSrmin': 0.5}, data=y, Nx=Nx, niter=niter, verbose=niter,
                            adaptive=True, scale=1.0, t_start=t_start)
    elif learning_alg == 'gibbs':
        alg = mcmc.ParticleGibbs(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.GuidedPF, data=y, Nx=Nx, niter=niter,
                                 verbose=niter)
    elif learning_alg == 'smc2':
        fk_smc2 = ssp.SMC2(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.GuidedPF, data=y, init_Nx=Nx,
                           ar_to_increase_Nx=-1.0, smc_options={'verbose': True})
        alg = particles.SMC(fk=fk_smc2, N=N)
    else:
        raise ValueError("learning_alg has to be one of 'pmmh', 'gibbs', 'smc2'; got {} instead.".format(learning_alg))
    start_user, start_process = time.time(), time.process_time()
    alg.run()  # Warning: takes a few seconds
    end_user, end_process = time.time(), time.process_time()
    print('Time user {:.1f}s; time processor {:.1f}s'.format(end_user - start_user, end_process - start_process))

    if learning_alg == 'pmmh' or learning_alg == 'cpmmh' or learning_alg == 'gibbs':
        burnin = 0  # discard the __ first iterations
        for i, param in enumerate(prior_dict.keys()):
            plt.figure()
            sb.histplot(alg.chain.theta[param][burnin:], bins=10)
            plt.title(param)
            plt.savefig(learning_alg + '_' + param + '.pdf')
        plt.show()
    else:
        for i, param in enumerate(prior_dict.keys()):
            plt.figure()
            sb.histplot([t[i] for t in alg.X.theta], bins=10)
            plt.title(param)
            plt.savefig(learning_alg + '_' + param + '.pdf')
        plt.show()

    return None


if __name__ == '__main__':
    # ---------------------------- data ----------------------------
    generation_type = 'Missingdata005'
    nb_timesteps = 500
    dim_obs = 20  # 20 or 36
    x, y = prepare_data(generation_type, nb_timesteps, dim_obs)

    # ---------------------------- parameter learning ----------------------------
    add_Q = False
    add_H = False
    add_legs = False
    add_imu = False
    add_alphas = True
    prior_dict, my_prior = set_prior(add_Q, add_H, add_legs, add_imu, add_alphas)
    Nx = 1000
    N = 20
    t_start = 100
    niter = 100
    learning_alg = 'cpmmh'  # cpmmh, pmmh, gibbs, smc2
    learn_model_parameters(prior_dict, my_prior, learning_alg, Nx, N, t_start, niter)