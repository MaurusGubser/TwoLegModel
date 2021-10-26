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

from ReadData import DataReader
from TwoLegSMCModel import TwoLegModel, TwoLegModelGuided
from Plotting import Plotter

if __name__ == '__main__':
    dt = 0.01
    dim_states = 18
    dim_observations = 36
    leg_constants = np.array([0.5, 0.6, 0.5, 0.6])
    imu_position = np.array([0.34, 0.29, 0.315, 0.33])
    a = np.array([5.6790e-03, 1.0575e+00, -1.2846e-01, -2.4793e-01, 3.6639e-01, -1.8980e-01,
                  5.6790e-01, 9.6320e-02, 2.5362e+00, -3.7986e+00, -7.8163e-02, -8.1819e-01,
                  -4.0705e-11, 5.0517e-03, -1.7762e+00, 3.3158e+00, -2.9528e-01, 5.3581e-01])

    P = 0.01 * np.eye(dim_states)

    cov_step = 0.01
    scale_x = 0.01
    scale_y = 1.0
    scale_phi = 100.0
    sigma_imu_acc = 0.1
    sigma_imu_gyro = 0.01
    sigma_press_velo = 0.1
    sigma_press_acc = 10.0

    my_model = TwoLegModel(dt=dt,
                           dim_states=dim_states,
                           dim_observations=dim_observations,
                           leg_constants=leg_constants,
                           imu_position=imu_position,
                           a=a,
                           P=P,
                           cov_step=cov_step,
                           scale_x=scale_x,
                           scale_y=scale_y,
                           scale_phi=scale_phi,
                           sigma_imu_acc=sigma_imu_acc,
                           sigma_imu_gyro=sigma_imu_gyro,
                           sigma_press_velo=sigma_press_velo,
                           sigma_press_acc=sigma_press_acc
                           )

    my_model_prop = TwoLegModelGuided(dt=dt,
                                      dim_states=dim_states,
                                      dim_observations=dim_observations,
                                      leg_constants=leg_constants,
                                      imu_position=imu_position,
                                      a=a,
                                      P=P,
                                      cov_step=cov_step,
                                      scale_x=scale_x,
                                      scale_y=scale_y,
                                      scale_phi=scale_phi,
                                      sigma_imu_acc=sigma_imu_acc,
                                      sigma_imu_gyro=sigma_imu_gyro,
                                      sigma_press_velo=sigma_press_velo,
                                      sigma_press_acc=sigma_press_acc
                                      )

    # simulated data from weto
    path_truth = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Normal/truth_normal.dat'
    path_obs = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Normal/noised_observations_normal.dat'
    data_reader = DataReader()
    max_timesteps = 300
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
    fk_boot = ssm.Bootstrap(ssm=my_model, data=y)
    fk_guided = ssm.GuidedPF(ssm=my_model_prop, data=y)
    pf = particles.SMC(fk=fk_guided, N=20, qmc=False, resampling='stratified', ESSrmin=0.5,
                       store_history=True, collect=[Moments()])
    pf.run()

    plotter = Plotter(truth=np.array(x), delta_t=0.01)
    plotter.plot_particles_trajectories(np.array(pf.hist.X), export_name='trajectories_9_Q1_H1_Prop100')

    particles_mean = np.array([m['mean'] for m in pf.summaries.moments])
    particles_var = np.array([m['var'] for m in pf.summaries.moments])
    plotter.plot_particle_moments(particles_mean=particles_mean, particles_var=particles_var,
                                  X_hist=np.array(pf.hist.X), export_name='trajectories_9_Q1_H1_Prop100')

    """
    # compare MC and QMC method
    results = particles.multiSMC(fk=fk_model, N=100, nruns=30, qmc={'SMC': False, 'SQMC': True})
    plt.figure()
    sb.boxplot(x=[r['output'].logLt for r in results], y=[r['qmc'] for r in results])
    #plt.show()
    """

    # smoothing
    #smooth_trajectories = pf.hist.backward_sampling(5)
    #plotter.plot_samples_detail(samples=np.array(smooth_trajectories), export_name='trajectories_0_Q1_H1')

    """
    # learning parameters
    prior_dict = {'sigma_x': dists.Uniform(0.0001, 1.0),
                  'sigma_y': dists.Uniform(0.0001, 1.0),
                  'sigma_phi': dists.Uniform(0.0001, 1.0)}
    my_prior = dists.StructDist(prior_dict)
    pmmh = mcmc.PMMH(ssm_cls=TwoLegModel, prior=my_prior, data=y, Nx=50, niter=1000)
    pmmh.run()  # Warning: takes a few seconds

    burnin = 100  # discard the 100 first iterations
    for i, param in enumerate(prior_dict.keys()):
        plt.subplot(2, 2, i + 1)
        sb.distplot(pmmh.chain.theta[param][burnin:], 40)
        plt.title(param)
    plt.show()
    """
