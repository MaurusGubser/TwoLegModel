import cProfile
import pstats
import time
import numpy as np
import particles

from particles.collectors import Moments, LogLts
from particles import state_space_models as ssm

from DataReaderWriter import DataReaderWriter
from TwoLegSMCModel import TwoLegModel
from Plotter import Plotter


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


if __name__ == '__main__':
    # ---------------------------- data ----------------------------
    generation_type = 'Missingdata005'
    nb_timesteps = 250
    dim_obs = 36  # 20 or 36
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
    sigma_imu_acc = 0.1  # 0.1
    sigma_imu_gyro = 0.1  # 0.1
    sigma_press_velo = 0.1  # 0.1
    sigma_press_acc = 1.0  # 1.0
    factor_H = 1.0  # 1.0

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
    nb_particles = 800
    fk_boot = ssm.Bootstrap(ssm=my_model, data=y)
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
    show_fig = True

    # plot_results(pf, x, y, dt, export_name_single, show_fig=show_fig, plt_smthng=False)

    # ---------------------------- profiling ----------------------------
    cProfile.run('run_particle_filter(fk_model=fk_guided, nb_particles=nb_particles, ESSrmin=0.5)', 'output.dat')

    with open('output_time_jit_{}particles.dat'.format(nb_particles), 'w') as f:
        p = pstats.Stats('output.dat', stream=f)
        p.sort_stats('time').print_stats()
