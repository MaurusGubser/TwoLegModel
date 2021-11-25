import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import particles
from collections import OrderedDict
from particles import distributions as dists
from particles.collectors import Moments
from particles import state_space_models as ssm
from particles import mcmc

from ReadData import DataReaderWriter
from TwoLegSMCModel import TwoLegModel
from Plotting import Plotter


def set_prior(add_Q, add_H, add_legs, add_imus):
    prior = {}
    if add_Q:
        prior_Q = {'scale_x': dists.Uniform(40.0, 200.0),
                   'scale_y': dists.Uniform(40.0, 200.0),
                   'scale_phi': dists.Uniform(50.0, 500.0)}
        prior_Q = {'scale_x': dists.LinearD(dists.InvGamma(3.0, 2.0), a=100.0, b=0.0),
                   'scale_y': dists.LinearD(dists.InvGamma(3.0, 2.0), a=100.0, b=0.0),
                   'scale_phi': dists.LinearD(dists.InvGamma(3.0, 2.0), a=250.0, b=0.0)}
        prior.update(prior_Q)
    if add_H:
        prior_H = {'sigma_imu_acc': dists.LinearD(dists.InvGamma(3.0, 2.0), a=0.1, b=0.0),
                   'sigma_imu_gyro': dists.LinearD(dists.InvGamma(3.0, 2.0), a=0.01, b=0.0),
                   'sigma_press_velo': dists.LinearD(dists.InvGamma(3.0, 2.0), a=0.1, b=0.0),
                   'sigma_press_acc': dists.LinearD(dists.InvGamma(3.0, 2.0), a=1000.0, b=0.0)}
        prior.update(prior_H)
    if add_legs:
        dist_femur = dists.Uniform(0.4, 0.6)
        dist_fibula = dists.Uniform(0.5, 0.7)
        prior_legs = {'len_legs': dists.IndepProd(*[dist_femur, dist_fibula, dist_femur, dist_fibula])}
        prior.update(prior_legs)
    if add_imus:
        dist_imu0 = dists.Uniform(0.29, 0.39)
        dist_imu1 = dists.Uniform(0.24, 0.34)
        dist_imu2 = dists.Uniform(0.265, 0.365)
        dist_imu3 = dists.Uniform(0.28, 0.38)
        prior_imus = {'pos_imus': [dist_imu0, dist_imu1, dist_imu2, dist_imu3]}
        prior.update(prior_imus)
    return prior, dists.StructDist(prior)


if __name__ == '__main__':
    dt = 0.01
    dim_states = 18
    dim_observations = 20
    length_legs = np.array([0.5, 0.6, 0.5, 0.6])
    position_imus = np.array([0.34, 0.29, 0.315, 0.33])
    a = np.array([5.6790e-03, 1.0575e+00, -1.2846e-01, -2.4793e-01, 3.6639e-01, -1.8980e-01,
                  5.6790e-01, 9.6320e-02, 2.5362e+00, -3.7986e+00, -7.8163e-02, -8.1819e-01,
                  -4.0705e-11, 5.0517e-03, -1.7762e+00, 3.3158e+00, -2.9528e-01, 5.3581e-01])
    P = 0.01 * np.eye(dim_states)

    cov_step = dt  # 0.01
    scale_x = 100.0  # 100.0
    scale_y = 100.0  # 100.0
    scale_phi = 250.0  # 250.0
    factor_Q = 1000.0  # 1000.0
    diag_Q = False
    sigma_imu_acc = 0.1  # 0.1
    sigma_imu_gyro = 0.01  # 0.01
    sigma_press_velo = 0.1  # 0.1
    sigma_press_acc = 1000.0  # 1000.0
    factor_H = 0.01  # 0.01

    factor_proposal = 1.1

    my_model = TwoLegModel(dt=dt,
                           dim_states=dim_states,
                           dim_observations=dim_observations,
                           len_legs=length_legs,
                           pos_imus=position_imus,
                           a=a,
                           P=P,
                           cov_step=cov_step,
                           scale_x=scale_x,
                           scale_y=scale_y,
                           scale_phi=scale_phi,
                           factor_Q=factor_Q,
                           diag_Q=diag_Q,
                           sigma_imu_acc=sigma_imu_acc,
                           sigma_imu_gyro=sigma_imu_gyro,
                           sigma_press_velo=sigma_press_velo,
                           sigma_press_acc=sigma_press_acc,
                           factor_H=factor_H,
                           factor_proposal=factor_proposal
                           )

    # simulated data from weto
    path_truth = 'GeneratedData/Normal/truth_normal.dat'    # 'GeneratedData/sandbox/truth_sandbox.dat'
    path_obs = 'GeneratedData/Normal/noised_observations_normal.dat'    # 'GeneratedData/sandbox/noised_observations_sandbox.dat'
    data_reader = DataReaderWriter()
    max_timesteps = 100
    data_reader.read_states_as_arr(path_truth, max_timesteps=max_timesteps)
    data_reader.read_observations_as_arr(path_obs, max_timesteps=max_timesteps)
    data_reader.prepare_lists()
    x = data_reader.states_list
    y = data_reader.observations_list
    if dim_observations == 20:
        y = [obs[:, (0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34)] for obs in y]
    # simulate data from this model
    # x_sim, y_sim = my_model.simulate(max_timesteps)

    # feynman-kac model
    nb_particles = 100
    fk_boot = ssm.Bootstrap(ssm=my_model, data=y)
    fk_guided = ssm.GuidedPF(ssm=my_model, data=y)
    pf = particles.SMC(fk=fk_guided, N=nb_particles, ESSrmin=0.5, store_history=True, collect=[Moments()], verbose=True)
    """
    # filter and plot
    start_user, start_process = time.time(), time.process_time()
    pf.run()
    end_user, end_process = time.time(), time.process_time()
    print('Time user {:.1f}s; time processor {:.1f}s'.format(end_user - start_user, end_process - start_process))
    print('Resampled {} of totally {} steps.'.format(np.sum(pf.summaries.rs_flags), max_timesteps))
    print('Log likelihood: {}'.format(pf.summaries.logLts))
    plotter = Plotter(true_states=np.array(x), true_obs=np.array(y), delta_t=dt)
    export_name = 'GF_{}steps_{}particles'.format(max_timesteps, nb_particles)
    # plotter.plot_observations(np.array(pf.hist.X), model=my_model, export_name=export_name)
    # plotter.plot_particles_trajectories(np.array(pf.hist.X), export_name=export_name)
    particles_mean = np.array([m['mean'] for m in pf.summaries.moments])
    particles_var = np.array([m['var'] for m in pf.summaries.moments])
    plotter.plot_ESS(pf.summaries.ESSs)
    plotter.plot_particle_moments(particles_mean=particles_mean, particles_var=particles_var,
                                  X_hist=None, export_name=export_name)  # X_hist = np.array(pf.hist.X)
    """
    """
    # compare MC and QMC method
    results = particles.multiSMC(fk=fk_guided, N=100, nruns=30, qmc={'SMC': False, 'SQMC': True})
    plt.figure()
    sb.boxplot(x=[r['output'].logLt for r in results], y=[r['qmc'] for r in results])
    plt.show()
    """
    """
    # smoothing
    smooth_trajectories = pf.hist.backward_sampling(5, linear_cost=False)
    data_reader.export_trajectory(np.array(smooth_trajectories), dt, export_name)
    plotter.plot_smoothed_trajectories(samples=np.array(smooth_trajectories), export_name=export_name)
    """
    # learning parameters
    add_Q = True
    add_H = False
    add_legs = False
    add_imu = False
    prior_dict, my_prior = set_prior(add_Q, add_H, add_legs, add_imu)
    pmmh = mcmc.PMMH(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.Bootstrap, data=y, Nx=100, niter=200,
                     verbose=10)
    start_user, start_process = time.time(), time.process_time()
    pmmh.run()  # Warning: takes a few seconds
    end_user, end_process = time.time(), time.process_time()
    print('Time user {:.1f}s; time processor {:.1f}s'.format(end_user - start_user, end_process - start_process))

    burnin = 0  # discard the __ first iterations
    for i, param in enumerate(prior_dict.keys()):
        plt.subplot(2, 2, i + 1)
        sb.distplot(pmmh.chain.theta[param][burnin:], 10)
        plt.title(param)
    plt.show()
