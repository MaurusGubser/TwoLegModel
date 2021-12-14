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

from ReadData import DataReaderWriter
from TwoLegSMCModel import TwoLegModel
from Plotting import Plotter


def set_prior(add_Q, add_H, add_legs, add_imus, add_a, add_alphas):
    prior = {}
    if add_Q:
        prior_Q = {'scale_x': dists.LinearD(dists.InvGamma(3.0, 2.0), a=100.0, b=0.0),
                   'scale_y': dists.LinearD(dists.InvGamma(3.0, 2.0), a=100.0, b=0.0),
                   'scale_phi': dists.LinearD(dists.InvGamma(3.0, 2.0), a=250.0, b=0.0)}
        prior_Q = {'scale_x': dists.Uniform(20.0, 200.0),
                   'scale_y': dists.Uniform(20.0, 200.0),
                   'scale_phi': dists.Uniform(50.0, 500.0)}
        prior.update(prior_Q)
    if add_H:
        prior_H = {'sigma_imu_acc': dists.LinearD(dists.InvGamma(3.0, 2.0), a=0.1, b=0.0),
                   'sigma_imu_gyro': dists.LinearD(dists.InvGamma(3.0, 2.0), a=0.01, b=0.0),
                   'sigma_press_velo': dists.LinearD(dists.InvGamma(3.0, 2.0), a=0.1, b=0.0),
                   'sigma_press_acc': dists.LinearD(dists.InvGamma(3.0, 2.0), a=1000.0, b=0.0)}
        prior_H = {'sigma_imu_acc': dists.Uniform(0.01, 1.0),
                   'sigma_imu_gyro': dists.Uniform(0.0001, 0.1),
                   'sigma_press_velo': dists.Uniform(0.01, 1.0),
                   'sigma_press_acc': dists.Uniform(100.0, 1000.0)}
        prior.update(prior_H)
    if add_legs:
        dist_femur = dists.Uniform(0.3, 0.7)
        dist_fibula = dists.Uniform(0.4, 0.8)
        prior_legs = {'len_legs': dists.IndepProd(*[dist_femur, dist_fibula, dist_femur, dist_fibula])}
        prior.update(prior_legs)
    if add_imus:
        dist_imu0 = dists.Uniform(0.0, 0.5)
        dist_imu1 = dists.Uniform(0.0, 0.6)
        dist_imu2 = dists.Uniform(0.0, 0.5)
        dist_imu3 = dists.Uniform(0.0, 0.6)
        prior_imus = {'pos_imus': dists.IndepProd(*[dist_imu0, dist_imu1, dist_imu2, dist_imu3])}
        prior.update(prior_imus)
    if add_a:
        mean = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        covar = np.diag([0.01, 0.05, 0.01, 0.01, 0.01, 0.01,
                         0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                         0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        prior_a = {'a': dists.MvNormal(loc=mean, cov=covar)}
        prior.update(prior_a)
    if add_alphas:
        prior_alphas = {'alpha_0': dists.Normal(loc=0.0, scale=0.2),
                        'alpha_1': dists.Normal(loc=0.0, scale=0.2),
                        'alpha_2': dists.Normal(loc=0.0, scale=0.2),
                        'alpha_3': dists.Normal(loc=0.0, scale=0.2)}
        prior.update(prior_alphas)
    return prior, dists.StructDist(prior)


if __name__ == '__main__':
    dt = 0.01
    dim_states = 18
    dim_observations = 20
    length_legs = np.array([0.5, 0.6, 0.5, 0.6])  # [0.5, 0.6, 0.5, 0.6]
    position_imus = np.array([0.34, 0.29, 0.315, 0.33])  # [0.34, 0.29, 0.315, 0.33]
    alpha_0 = 0.0
    alpha_1 = 0.0
    alpha_2 = 0.0
    alpha_3 = 0.0
    a = np.array([5.6790e-03, 1.0575e+00, -1.2846e-01, -2.4793e-01, 3.6639e-01, -1.8980e-01,
                  5.6790e-01, 9.6320e-02, 2.5362e+00, -3.7986e+00, -7.8163e-02, -8.1819e-01,
                  -4.0705e-11, 5.0517e-03, -1.7762e+00, 3.3158e+00, -2.9528e-01, 5.3581e-01])

    a = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    factor_init = 1.0

    cov_step = dt  # 0.01
    scale_x = 10000.0  # 100.0
    scale_y = 1000.0  # 100.0
    scale_phi = 10000000.0  # 250.0
    factor_Q = 1.0  # 1000.0
    diag_Q = False
    sigma_imu_acc = 0.1  # 0.1
    sigma_imu_gyro = 0.01  # 0.01
    sigma_press_velo = 0.1  # 0.1
    sigma_press_acc = 1000.0  # 1000.0
    factor_H = 0.01  # 0.01

    factor_proposal = 1.2

    my_model = TwoLegModel(dt=dt,
                           dim_states=dim_states,
                           dim_observations=dim_observations,
                           len_legs=length_legs,
                           pos_imus=position_imus,
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

    # simulated data from weto
    path_truth = 'GeneratedData/Missingdata005/truth_missingdata.dat'  # 'GeneratedData/RotatedFemurLeft/truth_rotatedfemurleft.dat'    # 'GeneratedData/Missingdata/truth_missingdata.dat'
    path_obs = 'GeneratedData/Missingdata005/noised_observations_missingdata.dat'  # 'GeneratedData/RotatedFemurLeft/noised_observations_rotatedfemurleft.dat'    # 'GeneratedData/Missingdata/noised_observations_missingdata.dat'
    data_reader = DataReaderWriter()
    max_timesteps = 1200
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
    nb_particles = 500
    fk_boot = ssm.Bootstrap(ssm=my_model, data=y)
    fk_guided = ssm.GuidedPF(ssm=my_model, data=y)
    pf = particles.SMC(fk=fk_guided, N=nb_particles, ESSrmin=0.2, store_history=True, collect=[Moments()], verbose=True)

    # filter and plot
    start_user, start_process = time.time(), time.process_time()
    pf.run()
    end_user, end_process = time.time(), time.process_time()
    print('Time user {:.1f}s; time processor {:.1f}s'.format(end_user - start_user, end_process - start_process))
    print('Resampled {} of totally {} steps.'.format(np.sum(pf.summaries.rs_flags), max_timesteps))
    print('Log likelihood: {}'.format(pf.summaries.logLts))
    plotter = Plotter(true_states=np.array(x), true_obs=np.array(y), delta_t=dt)
    export_name = 'GF_AllData_steps{}_particles{}_factorP{}_factorQ{}_factorH{}_factorProp'.format(max_timesteps,
                                                                                                   nb_particles,
                                                                                                   factor_init,
                                                                                                   factor_Q,
                                                                                                   factor_H,
                                                                                                   factor_proposal)
    plotter.plot_observations(np.array(pf.hist.X), model=my_model, export_name=export_name)
    # plotter.plot_particles_trajectories(np.array(pf.hist.X), export_name=export_name)
    particles_mean = np.array([m['mean'] for m in pf.summaries.moments])
    particles_var = np.array([m['var'] for m in pf.summaries.moments])
    plotter.plot_ESS(pf.summaries.ESSs)
    plotter.plot_particle_moments(particles_mean=particles_mean, particles_var=particles_var,
                                  X_hist=None, export_name=export_name)  # X_hist = np.array(pf.hist.X)

    """
    # compare MC and QMC method
    results = particles.multiSMC(fk=fk_guided, N=100, nruns=30, qmc={'SMC': False, 'SQMC': True})
    plt.figure()
    sb.boxplot(x=[r['output'].logLt for r in results], y=[r['qmc'] for r in results])
    plt.show()
    """

    # smoothing
    smooth_trajectories = pf.hist.backward_sampling(5, linear_cost=False)
    data_reader.export_trajectory(np.array(smooth_trajectories), dt, export_name)
    plotter.plot_smoothed_trajectories(samples=np.array(smooth_trajectories), export_name=export_name)

    """
    # learning parameters
    add_Q = False
    add_H = False
    add_legs = False
    add_imu = True
    add_a = False
    add_alphas = False
    prior_dict, my_prior = set_prior(add_Q, add_H, add_legs, add_imu, add_a, add_alphas)
    pmmh = mcmc.PMMH(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.GuidedPF, smc_options={'ESSrmin': 0.2},
                     data=y, Nx=100, niter=100, verbose=20)
    pg = mcmc.ParticleGibbs(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.Bootstrap, data=y, Nx=100, niter=10,
                            verbose=5)
    fk_smc2 = ssp.SMC2(ssm_cls=TwoLegModel, prior=my_prior, fk_cls=ssm.GuidedPF, data=y, init_Nx=100,
                       ar_to_increase_Nx=0.1)
    smc2 = particles.SMC(fk=fk_smc2, N=200)

    start_user, start_process = time.time(), time.process_time()
    pmmh.run()  # Warning: takes a few seconds
    # pg.run() need to define update_theta method for a mcmc.ParticleGibbs subclass
    # smc2.run()
    end_user, end_process = time.time(), time.process_time()
    print('Time user {:.1f}s; time processor {:.1f}s'.format(end_user - start_user, end_process - start_process))

    burnin = 0  # discard the __ first iterations
    for i, param in enumerate(prior_dict.keys()):
        plt.subplot(2, 2, i + 1)
        sb.histplot(pmmh.chain.theta[param][burnin:], bins=10)
        plt.title(param)
    plt.show()
    """
