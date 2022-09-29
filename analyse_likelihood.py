import time
import numpy as np
import particles

from particles.collectors import Moments, LogLts
from particles import state_space_models as ssm

from DataReaderWriter import DataReaderWriter
from TwoLegModelSMC import TwoLegModel
from Plotter import Plotter


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


def analyse_likelihood(fk_model, true_states, data, dt, nb_particles, nb_runs,
                       t_start, show_fig, export_name=None):
    start_user, start_process = time.time(), time.process_time()
    results = particles.multiSMC(fk=fk_model, N=nb_particles, nruns=nb_runs, collect=[Moments()], nprocs=-1)
    end_user, end_process = time.time(), time.process_time()
    s_user = end_user - start_user
    s_process = end_process - start_process
    print('Time user {:.0f}min {:.0f}s; time processor {:.0f}min {:.0f}s'.format(s_user // 60, s_user % 60,
                                                                                 s_process // 60, s_process % 60))
    assert t_start < results[0]['output'].fk.T, 'Start time should be shorter than number of steps.'
    plotter = Plotter(np.array(true_states), np.array(data), dt, export_name, show_fig=show_fig)
    for N in nb_particles:
        logLts = [r['output'].summaries.logLts[-1] - r['output'].summaries.logLts[t_start] for r in results if
                  r['N'] == N]
        mean, var = np.mean(logLts, axis=0), np.var(logLts, axis=0)
        print('N={:.5E}, Mean loglhd={:.5E}, Variance loglhd={:.5E}'.format(N, mean, var))
        """
        bad_run, middle_run = get_extremal_cases(output_multismc=results, N=N, t_start=t_start)
        plotter.plot_particle_moments(bad_run['mean'], bad_run['var'], name_suffix='_bad_N{}_'.format(N))
        plotter.plot_particle_moments(middle_run['mean'], middle_run['var'],
                                               name_suffix='_middle_N{}_'.format(N))
        """
    plotter.plot_logLts_multiple_runs(results, nb_particles, nb_runs, t_start)
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
    alpha_0 = 0.0
    alpha_1 = 0.0
    alpha_2 = 0.0
    alpha_3 = 0.0

    b0 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    factor_init = 0.01  # 0.01

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
    factor_S = 1.0  # 1.0

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
                           b0=b0,
                           factor_init=factor_init,
                           cov_step=cov_step,
                           lambda_x=scale_x,
                           lambda_y=scale_y,
                           lambda_phi=scale_phi,
                           factor_Q=factor_Q,
                           diag_Q=diag_Q,
                           sigma_imu_acc=sigma_imu_acc,
                           sigma_imu_gyro=sigma_imu_gyro,
                           sigma_press_velo=sigma_press_velo,
                           sigma_press_acc=sigma_press_acc,
                           factor_S=factor_S,
                           factor_proposal=factor_proposal
                           )

    # ---------------------------- loglikelihood stats ----------------------------

    # fk_boot = ssm.Bootstrap(ssm=my_model, data=y)
    fk_guided = ssm.GuidedPF(ssm=my_model, data=y)

    Ns = [500, 1000]
    nb_runs = 5
    t_start = 500
    show_fig = True
    export_name_multi = 'MultiRun_{}_steps{}_Ns{}_nbruns{}_tstart{}_factorP{}_factorQ{}_factorH{}_factorProp{}'.format(
        generation_type,
        nb_timesteps, Ns,
        nb_runs,
        t_start,
        factor_init,
        factor_Q,
        factor_S,
        factor_proposal)
    analyse_likelihood(fk_guided, x, y, dt, Ns, nb_runs, t_start, show_fig=show_fig, export_name=export_name_multi)
