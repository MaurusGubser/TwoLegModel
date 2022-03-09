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
    results = particles.multiSMC(fk=fk_model, N=nb_particles, nruns=nb_runs, collect=[Moments()], nprocs=-1)
    end_user, end_process = time.time(), time.process_time()
    print('Time user {:.1f}s; time processor {:.1f}s'.format(end_user - start_user, end_process - start_process))
    assert t_start < results[0]['output'].fk.T, 'Start time should be shorter than number of steps.'

    plotter_multismc = Plotter(np.array(true_states), np.array(data), dt, export_name, show_fig=show_fig)
    for N in nb_particles:
        logLts = [r['output'].summaries.logLts[-1] - r['output'].summaries.logLts[t_start] for r in results if
                  r['N'] == N]
        mean, var = np.mean(logLts, axis=0), np.var(logLts, axis=0)
        print('N={:.5E}, Mean loglhd={:.5E}, Variance loglhd={:.5E}'.format(N, mean, var))
        """
        bad_run, middle_run = get_extremal_cases(output_multismc=results, N=N, t_start=t_start)
        plotter_multismc.plot_particle_moments(bad_run['mean'], bad_run['var'], name_suffix='_bad_N{}_'.format(N))
        plotter_multismc.plot_particle_moments(middle_run['mean'], middle_run['var'],
                                               name_suffix='_middle_N{}_'.format(N))
        """
    plotter_multismc.plot_logLts_multiple_runs(results, nb_particles, nb_runs, t_start)
    return None


if __name__ == '__main__':
    # ---------------------------- data ----------------------------
    generation_type = 'Missingdata005'
    nb_timesteps = 200
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

    # ---------------------------- loglikelihood stats ----------------------------

    # fk_boot = ssm.Bootstrap(ssm=my_model, data=y)
    fk_guided = ssm.GuidedPF(ssm=my_model, data=y)

    Ns = [100, 200]
    nb_runs = 10
    t_start = 100
    show_fig = True
    export_name_multi = 'MultiRun_{}_steps{}_Ns{}_nbruns{}_tstart{}_factorP{}_factorQ{}_factorH{}_factorProp{}'.format(
        generation_type,
        nb_timesteps, Ns,
        nb_runs,
        t_start,
        factor_init,
        factor_Q,
        factor_H,
        factor_proposal)
    analyse_likelihood(fk_guided, x, y, dt, Ns, nb_runs, t_start, show_fig=show_fig, export_name=export_name_multi)