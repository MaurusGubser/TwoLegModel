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
from particles import smc_samplers as ssp

from DataReaderWriter import DataReaderWriter
from TwoLegSMCModel import TwoLegModel
from Plotter import Plotter


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
        prior_imus = {'pos_imu0': dists.Normal(loc=0.25, scale=0.3), 'pos_imu1': dists.Normal(loc=0.3, scale=0.3),
                      'pos_imu2': dists.Normal(loc=0.25, scale=0.3), 'pos_imu3': dists.Normal(loc=0.3, scale=0.3)}
        prior_dict.update(prior_imus)
    if add_alphas:
        prior_alphas = {'alpha_0': dists.Normal(loc=0.0, scale=0.3),
                        'alpha_1': dists.Normal(loc=0.0, scale=0.3),
                        'alpha_2': dists.Normal(loc=0.0, scale=0.3),
                        'alpha_3': dists.Normal(loc=0.0, scale=0.3)}
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


def run_particle_filter(fk_model):
    pf = particles.SMC(fk=fk_model, N=nb_particles, ESSrmin=0.5, store_history=True, collect=[Moments()],
                       verbose=True)
    start_user, start_process = time.time(), time.process_time()
    pf.run()
    end_user, end_process = time.time(), time.process_time()
    print('Time user {:.1f}s; time processor {:.1f}s'.format(end_user - start_user, end_process - start_process))
    print('Resampled {} of totally {} steps.'.format(np.sum(pf.summaries.rs_flags), nb_timesteps))
    loglikelihood = np.sum(pf.summaries.logLts)
    print('Log likelihood = {:.3E}'.format(loglikelihood))
    return pf


def plot_results(pf, x, y, dt, export_name, plt_smthng=False):
    plotter = Plotter(true_states=np.array(x), true_obs=np.array(y), delta_t=dt, export_name=export_name)

    plotter.plot_observations(np.array(pf.hist.X), model=my_model)
    # plotter.plot_particles_trajectories(np.array(pf.hist.X))
    particles_mean = np.array([m['mean'] for m in pf.summaries.moments])
    particles_var = np.array([m['var'] for m in pf.summaries.moments])
    plotter.plot_particle_moments(particles_mean=particles_mean, particles_var=particles_var)
    plotter.plot_ESS(pf.summaries.ESSs)
    plotter.plot_logLts(pf.summaries.logLts)
    if plt_smthng:
        smooth_trajectories = pf.hist.backward_sampling(5, linear_cost=False, return_ar=False)
        data_reader = DataReaderWriter()
        data_reader.export_trajectory(np.array(smooth_trajectories), dt, export_name)
        plotter.plot_smoothed_trajectories(samples=np.array(smooth_trajectories))
    return None


def compute_loglikelihood_stats(fk_model, nb_particles, nb_runs):
    results = particles.multiSMC(fk=fk_model, N=nb_particles, nruns=nb_runs, nprocs=-1)
    for N in nb_particles:
        loglts = [r['output'].logLt for r in results if r['N'] == N]
        print('N={}, Mean loglikelihood={}, Variance loglikelihood={}'.format(N, np.mean(loglts), np.var(loglts)))
    plt.figure()
    sb.boxplot(x=[r['output'].logLt for r in results], y=[str(r['N']) for r in results])
    plt.xlabel('Log likelihood')
    plt.ylabel('Number of particles')
    plt.show()
    return None


def learn_model_parameters(prior_dict, my_prior, learning_alg):
    if learning_alg == 'pmmh':
        alg = mcmc.PMMH(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.GuidedPF, smc_options={'ESSrmin': 0.5},
                        data=y, Nx=50, niter=100, verbose=50, adaptive=True, scale=1.0)
    elif learning_alg == 'gibbs':
        alg = mcmc.ParticleGibbs(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.GuidedPF, data=y, Nx=100, niter=10,
                                 verbose=5)
    elif learning_alg == 'smc2':
        fk_smc2 = ssp.SMC2(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.GuidedPF, data=y, init_Nx=50,
                           ar_to_increase_Nx=-1.0, smc_options={'verbose': True})
        alg = particles.SMC(fk=fk_smc2, N=20)
    else:
        raise ValueError("learning_alg has to be one of 'pmmh', 'gibbs', 'smc2'; got {} instead.".format(learning_alg))
    start_user, start_process = time.time(), time.process_time()
    alg.run()  # Warning: takes a few seconds
    end_user, end_process = time.time(), time.process_time()
    print('Time user {:.1f}s; time processor {:.1f}s'.format(end_user - start_user, end_process - start_process))

    burnin = 0  # discard the __ first iterations
    for i, param in enumerate(prior_dict.keys()):
        plt.subplot(2, 2, i + 1)
        sb.histplot(alg.chain.theta[param][burnin:], bins=10)
        plt.title(param)
    plt.show()

    return None


if __name__ == '__main__':
    # ---------------------------- data ----------------------------
    generation_type = 'Missingdata005'
    nb_timesteps = 1000
    dim_obs = 20  # 20 or 36
    x, y = prepare_data(generation_type, nb_timesteps, dim_obs)

    # ---------------------------- model ----------------------------
    dt = 0.01
    dim_states = 18
    dim_observations = 20
    femur_left = 0.5    # 0.5
    fibula_left = 0.6   # 0.6
    femur_right = 0.5   # 0.5
    fibula_right = 0.6  # 0.6
    pos_imu0 = 0.34     # 0.34
    pos_imu1 = 0.29     # 0.29
    pos_imu2 = 0.315    # 0.315
    pos_imu3 = 0.33     # 0.33
    alpha_0 = 0.0
    alpha_1 = 0.0
    alpha_2 = 0.0
    alpha_3 = 0.0

    a = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    factor_init = 0.1  # 0.1

    cov_step = dt  # 0.01
    scale_x = 10000.0  # 10000.0
    scale_y = 1000.0  # 1000.0
    scale_phi = 10000000.0  # 10000000.0
    factor_Q = 1.0  # 1.0
    diag_Q = False
    sigma_imu_acc = 0.01  # 0.01
    sigma_imu_gyro = 0.01  # 0.01
    sigma_press_velo = 0.01  # 0.01
    sigma_press_acc = 0.1  # 0.1
    factor_H = 10.0  # 1.0

    factor_proposal = 1.2  # 1.2

    my_model = TwoLegModel(dt=dt,
                           dim_states=dim_states,
                           dim_observations=dim_observations,
                           femur_left=femur_left,
                           fibula_left=fibula_left,
                           femur_right=femur_right,
                           fibula_right=fibula_right,
                           pos_imu0=pos_imu0,
                           pos_imu1=pos_imu1,
                           pos_imu2=pos_imu2,
                           pos_imu3=pos_imu3,
                           alpha_0=alpha_0,
                           alpha_1=alpha_1,
                           alpha_2=alpha_2,
                           alpha_3=alpha_3,
                           a=a,
                           factor_init=factor_init,
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

    # ---------------------------- particle filter ----------------------------
    nb_particles = 500
    fk_boot = ssm.Bootstrap(ssm=my_model, data=y)
    fk_guided = ssm.GuidedPF(ssm=my_model, data=y)
    # pf = run_particle_filter(fk_model=fk_guided)

    # ---------------------------- plot results ----------------------------
    export_name = 'GF_{}_steps{}_particles{}_factorP{}_factorQ{}_factorH{}_factorProp{}'.format(
        generation_type,
        nb_timesteps,
        nb_particles,
        factor_init,
        factor_Q,
        factor_H,
        factor_proposal)
    # plot_results(pf, x, y, dt, export_name, plt_smthng=True)

    # ---------------------------- loglikelihood stats ----------------------------
    Ns = [5000, 10000]
    nb_runs = 30
    compute_loglikelihood_stats(fk_model=fk_guided, nb_particles=Ns, nb_runs=nb_runs)

    # ---------------------------- loglikelihood stats ----------------------------
    add_Q = False
    add_H = False
    add_legs = False
    add_imu = False
    add_alphas = True
    prior_dict, my_prior = set_prior(add_Q, add_H, add_legs, add_imu, add_alphas)
    learning_alg = 'smc2'  # pmmh, gibbs, smc2
    # learn_model_parameters(prior_dict, my_prior, learning_alg)
