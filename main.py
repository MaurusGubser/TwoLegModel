import os
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
from CustomMCMC import CustomPMMH


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
        prior_alphas = {'alpha_0': dists.Normal(loc=0.0, scale=0.3),
                        'alpha_1': dists.Normal(loc=0.0, scale=0.3),
                        'alpha_2': dists.Normal(loc=0.0, scale=0.3),
                        'alpha_3': dists.Normal(loc=0.0, scale=0.3)}
        prior_alphas = {'alpha_0': dists.Normal(loc=0.0, scale=0.5)}
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


def run_particle_filter(fk_model, nb_particles, ESSrmin=0.5):
    pf = particles.SMC(fk=fk_model, N=nb_particles, ESSrmin=ESSrmin, store_history=True, collect=[Moments()],
                       verbose=True)
    start_user, start_process = time.time(), time.process_time()
    pf.run()
    end_user, end_process = time.time(), time.process_time()
    print('Time user {:.1f}s; time processor {:.1f}s'.format(end_user - start_user, end_process - start_process))
    print('Resampled {} of totally {} steps.'.format(np.sum(pf.summaries.rs_flags), nb_timesteps))
    loglikelihood = pf.summaries.logLts[-1]
    print('Log likelihood = {:.3E}'.format(loglikelihood))
    return pf


def plot_results(pf, x, y, dt, export_name, show_fig, plt_smthng=False):
    plotter_single_run = Plotter(true_states=np.array(x), true_obs=np.array(y), delta_t=dt, show_fig=show_fig,
                                 export_name=export_name)

    plotter_single_run.plot_observations(np.array(pf.hist.X), model=my_model)
    # plotter_single_run.plot_particles_trajectories(np.array(pf.hist.X))
    particles_mean = np.array([m['mean'] for m in pf.summaries.moments])
    particles_var = np.array([m['var'] for m in pf.summaries.moments])
    plotter_single_run.plot_particle_moments(particles_mean=particles_mean, particles_var=particles_var)
    plotter_single_run.plot_ESS(pf.summaries.ESSs)
    plotter_single_run.plot_logLts_one_run(pf.summaries.logLts)
    if plt_smthng:
        smooth_trajectories = pf.hist.backward_sampling(5, linear_cost=False, return_ar=False)
        data_reader = DataReaderWriter()
        data_reader.export_trajectory(np.array(smooth_trajectories), dt, export_name)
        plotter_single_run.plot_smoothed_trajectories(samples=np.array(smooth_trajectories))
    return None


def get_extremal_cases(output_multismc, N, t_start):
    logLts = []
    mean_X = []
    var_X = []
    for r in output_multismc:
        if r['N'] == N:
            logLts.append(r['output'].summaries.logLts[-1] - r['output'].summaries.logLts[t_start])
            mean_X.append([stats['mean'] for stats in r['output'].summaries.moments])
            var_X.append([stats['var'] for stats in r['output'].summaries.moments])
    mean, sd = np.mean(logLts, axis=0), np.std(logLts, axis=0)
    res = np.abs(logLts - mean)
    idx_bad, idx_middle = np.argmax(res), np.argmin(res)
    print('Worst run has likelihood={}'.format(logLts[idx_bad]))
    print('Middle run has likelihood={}'.format(logLts[idx_middle]))
    bad_run = {'mean': np.array(mean_X[idx_bad]), 'var': np.array(var_X[idx_bad])}
    middle_run = {'mean': np.array(mean_X[idx_middle]), 'var': np.array(var_X[idx_middle])}
    return bad_run, middle_run


def analyse_likelihood(fk_model, true_states, data, dt, nb_particles, nb_runs, t_start, show_fig, export_name=None):
    start_user, start_process = time.time(), time.process_time()
    results = particles.multiSMC(fk=fk_model, N=nb_particles, nruns=nb_runs, collect=[Moments], nprocs=-1)
    end_user, end_process = time.time(), time.process_time()
    print('Time user {:.1f}s; time processor {:.1f}s'.format(end_user - start_user, end_process - start_process))
    assert t_start < results[0]['output'].fk.T

    plotter_multismc = Plotter(np.array(true_states), np.array(data), dt, export_name, show_fig=show_fig)
    for N in nb_particles:
        logLts = [r['output'].summaries.logLts[-1] - r['output'].summaries.logLts[t_start] for r in results if
                  r['N'] == N]
        mean, var = np.mean(logLts, axis=0), np.var(logLts, axis=0)
        print('N={:.5E}, Mean loglhd={:.5E}, Variance loglhd={:.5E}'.format(N, mean, var))
        bad_run, middle_run = get_extremal_cases(output_multismc=results, N=N, t_start=t_start)
        plotter_multismc.plot_particle_moments(bad_run['mean'], bad_run['var'], name_suffix='_bad_N{}_'.format(N))
        plotter_multismc.plot_particle_moments(middle_run['mean'], middle_run['var'],
                                               name_suffix='_middle_N{}_'.format(N))

    plotter_multismc.plot_logLts_multiple_runs(results, nb_particles, nb_runs, t_start)

    return None


def learn_model_parameters(prior_dict, my_prior, learning_alg, t_start):
    if learning_alg == 'pmmh':
        alg = mcmc.PMMH(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.GuidedPF,
                        smc_options={'ESSrmin': 0.5}, data=y, Nx=100, niter=100, verbose=100,
                        adaptive=True, scale=1.0)
    elif learning_alg == 'cpmmh':
        alg = CustomPMMH(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.GuidedPF,
                         smc_options={'ESSrmin': 0.5}, data=y, Nx=100, niter=100, verbose=100,
                         adaptive=True, scale=1.0, t_start=t_start)
    elif learning_alg == 'gibbs':
        alg = mcmc.ParticleGibbs(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.GuidedPF, data=y, Nx=100, niter=10,
                                 verbose=5)
    elif learning_alg == 'smc2':
        fk_smc2 = ssp.SMC2(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.GuidedPF, data=y, init_Nx=20,
                           ar_to_increase_Nx=-1.0, smc_options={'verbose': True})
        alg = particles.SMC(fk=fk_smc2, N=10)
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
    nb_timesteps = 100
    dim_obs = 20  # 20 or 36
    x, y = prepare_data(generation_type, nb_timesteps, dim_obs)

    # ---------------------------- model ----------------------------
    dt = 0.01
    dim_states = 18  # 18
    dim_observations = dim_obs  # 20
    femur_left = 0.5  # 0.5
    fibula_left = 0.6  # 0.6
    femur_right = 0.5  # 0.5
    fibula_right = 0.6  # 0.6
    pos_imu0 = 0.34  # 0.34
    pos_imu1 = 0.29  # 0.29
    pos_imu2 = 0.315  # 0.315
    pos_imu3 = 0.33  # 0.33
    alpha_0 = 0.0
    alpha_1 = 0.0
    alpha_2 = 0.0
    alpha_3 = 0.0

    a = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    factor_init = 0.01  # 0.1

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
    factor_H = 10.0  # 10.0

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
    nb_particles = 1000
    # fk_boot = ssm.Bootstrap(ssm=my_model, data=y)
    fk_guided = ssm.GuidedPF(ssm=my_model, data=y)
    # pf = run_particle_filter(fk_model=fk_guided, nb_particles=nb_particles, ESSrmin=0.5)

    # ---------------------------- plot results ----------------------------
    export_name_single = 'SingleRun_{}_steps{}_particles{}_factorP{}_factorQ{}_factorH{}_factorProp{}'.format(
        generation_type,
        nb_timesteps,
        nb_particles,
        factor_init,
        factor_Q,
        factor_H,
        factor_proposal)
    # plot_results(pf, x, y, dt, export_name_single, plt_smthng=False)

    # ---------------------------- loglikelihood stats ----------------------------
    Ns = [10, 20]
    nb_runs = 20
    t_start = 0
    show_fig = False
    export_name_multi = 'MultiRun_{}_steps{}_Ns{}_nbruns{}_tstart{}_factorP{}_factorQ{}_factorH{}_factorProp{}'.format(
        generation_type,
        nb_timesteps, Ns,
        nb_runs,
        t_start,
        factor_init,
        factor_Q,
        factor_H,
        factor_proposal)
    # analyse_likelihood(fk_guided, x, y, dt, Ns, nb_runs, t_start, show_fig=show_fig, export_name=export_name_multi)

    # ---------------------------- loglikelihood stats ----------------------------
    add_Q = False
    add_H = False
    add_legs = False
    add_imu = False
    add_alphas = True
    prior_dict, my_prior = set_prior(add_Q, add_H, add_legs, add_imu, add_alphas)
    t_start = 500
    learning_alg = 'cpmmh'  # cpmmh, pmmh, gibbs, smc2
    learn_model_parameters(prior_dict, my_prior, learning_alg, t_start)
