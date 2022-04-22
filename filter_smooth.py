import time
import numpy as np
import particles

from particles.collectors import Moments, LogLts
from particles import state_space_models as ssm

from DataReaderWriter import DataReaderWriter
from TwoLegModelSMC import TwoLegModel
from Plotter import Plotter


def run_particle_filter(fk_model, nb_particles, ESSrmin=0.5):
    pf = particles.SMC(fk=fk_model, N=nb_particles, ESSrmin=ESSrmin, store_history=True, collect=[Moments()],
                       verbose=True)
    start_user, start_process = time.time(), time.process_time()
    pf.run()
    end_user, end_process = time.time(), time.process_time()
    s_user = end_user - start_user
    s_process = end_process - start_process
    print('Time user {:.0f}min {:.0f}s; time processor {:.0f}min {:.0f}s'.format(s_user // 60, s_user % 60,
                                                                                 s_process // 60, s_process % 60))
    print('Resampled {} of totally {} steps.'.format(np.sum(pf.summaries.rs_flags), nb_timesteps))
    loglikelihood = pf.summaries.logLts[-1]
    print('Log likelihood = {:.3E}'.format(loglikelihood))
    return pf


def plot_results(pf, obs_map, x, y, dt, export_name, show_fig, plt_smthng=False):
    plotter = Plotter(true_states=np.array(x), true_obs=np.array(y), delta_t=dt, show_fig=show_fig,
                      export_name=export_name)

    plotter.plot_observations(np.array(pf.hist.X), observation_map=obs_map)
    # plotter.plot_particles_trajectories(np.array(pf.hist.X))
    particles_mean = np.array([m['mean'] for m in pf.summaries.moments])
    particles_var = np.array([m['var'] for m in pf.summaries.moments])
    plotter.plot_particle_moments(particles_mean=particles_mean, particles_var=particles_var)
    plotter.plot_ESS(pf.summaries.ESSs)
    plotter.plot_logLts_one_run(pf.summaries.logLts)
    if plt_smthng:
        smooth_trajectories = pf.hist.backward_sampling(10, linear_cost=False, return_ar=False)
        data_writer = DataReaderWriter()
        data_writer.export_trajectory(np.array(smooth_trajectories), dt, export_name)
        plotter.plot_smoothed_trajectories(samples=np.array(smooth_trajectories))
    return None


if __name__ == '__main__':
    # ---------------------------- data ----------------------------
    generation_type = 'Missingdata005'
    nb_timesteps = 1000
    dim_obs = 20  # 20 or 36
    data_reader = DataReaderWriter()
    x, y = data_reader.get_data_as_lists(generation_type, nb_timesteps, dim_obs)

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
    alpha0 = 0.0
    alpha1 = 0.0
    alpha2 = 0.0
    alpha3 = 0.0

    b0 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    factor_Q0 = 0.1  # 0.01

    lambda_x = 0.5*10000.0  # 10000.0
    lambda_y = 1000.0  # 1000.0
    lambda_phi = 10000000.0  # 10000000.0

    sigma_imu_acc = 0.1  # 0.1
    sigma_imu_gyro = 0.1  # 0.1
    sigma_press_velo = 0.1  # 0.1
    sigma_press_acc = 1.0  # 1.0

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
                           alpha0=alpha0,
                           alpha1=alpha1,
                           alpha2=alpha2,
                           alpha3=alpha3,
                           b0=b0,
                           factor_Q0=factor_Q0,
                           lambda_x=lambda_x,
                           lambda_y=lambda_y,
                           lambda_phi=lambda_phi,
                           sigma_imu_acc=sigma_imu_acc,
                           sigma_imu_gyro=sigma_imu_gyro,
                           sigma_press_velo=sigma_press_velo,
                           sigma_press_acc=sigma_press_acc,
                           factor_proposal=factor_proposal
                           )

    # ---------------------------- particle filter ----------------------------
    nb_particles = 1000
    ESSrmin = 0.5
    fk_boot = ssm.Bootstrap(ssm=my_model, data=y)
    fk_guided = ssm.GuidedPF(ssm=my_model, data=y)
    pf = run_particle_filter(fk_model=fk_guided, nb_particles=nb_particles, ESSrmin=ESSrmin)

    # ---------------------------- plot results ----------------------------
    export_name = 'SingleRun_{}_steps{}_particles{}_factorQ0{}_lambdax{}_lambday{}_lambdaphi{}_simuacc{}_simugyro{}_spressvelo{}_spressacc{}_factorProp{}'.format(
        generation_type,
        nb_timesteps,
        nb_particles,
        factor_Q0,
        lambda_x,
        lambda_y,
        lambda_phi,
        sigma_imu_acc,
        sigma_imu_gyro,
        sigma_press_velo,
        sigma_press_acc,
        factor_proposal)
    show_fig = True
    plt_smoothed = True
    plot_results(pf, my_model.state_to_observation, x, y, dt, export_name, show_fig, plt_smoothed)
