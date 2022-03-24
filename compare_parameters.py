import time
import numpy as np
import particles
from matplotlib import pyplot as plt
import seaborn as sb

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


def compare_parameters(fk_models, nb_timesteps, nb_particles, nb_runs, t_start, show_fig, export_name=None):
    start_user, start_process = time.time(), time.process_time()
    results = particles.multiSMC(fk=fk_models, N=nb_particles, nruns=nb_runs, collect=[LogLts()], nprocs=-1)
    end_user, end_process = time.time(), time.process_time()
    print('Time user {:.1f}s; time processor {:.1f}s'.format(end_user - start_user, end_process - start_process))

    logLts_truncated = [r['output'].summaries.logLts[-1] - r['output'].summaries.logLts[t_start] for r in results]
    mean, var = np.mean(logLts_truncated, axis=0), np.var(logLts_truncated, axis=0)
    print('N={:.5E}, Mean loglhd truncated={:.5E}, Variance loglhd truncated={:.5E}'.format(N, mean, var))
    sb.boxplot(x=[r['output'].summaries.logLts[-1] - r['output'].summaries.logLts[t_start] for r in results],
               y=[r['fk'] for r in results])
    plt.title('Boxplots for truncated likelihood')
    plt.show()

    t_vals = np.arange(nb_timesteps)
    fig = plt.figure(figsize=(15, 8))
    for fk_model in fk_models.keys():
        logLts = np.array([r['output'].summaries.logLts for r in results if r['fk'] == fk_model])
        mean, std = np.mean(logLts, axis=0), np.std(logLts, axis=0)
        plt.plot(t_vals, mean, label=fk_model)
        plt.fill_between(t_vals, mean - std, mean + std, alpha=0.5)
        plt.xlabel('Timesteps')
        plt.ylabel('$p(y_{0:t})$')
        plt.legend()
        plt.title('Loglikelihood, not truncated')
    plt.show()
    return None


if __name__ == '__main__':
    # ---------------------------- data ----------------------------
    generation_type = 'Missingdata005'
    nb_timesteps = 150
    dim_obs = 20  # 20 or 36
    x, y = prepare_data(generation_type, nb_timesteps, dim_obs)

    # ---------------------------- model ----------------------------
    # ! model works with parameters defined below; other parameters are chosen as the standard params in TwoLegModelSMC.py
    dt = 0.01
    N = 100
    t_start = 0
    nb_runs = 3
    show_fig = False
    export_name = 'pos_imu0'

    parameters = [{'pos_imu0': 0.2},
                  {'pos_imu0': 0.25, 'pos_imu1': 0.2}]  # , {'pos_imu0': 0.3}, {'pos_imu0': 0.35}, {'pos_imu0': 0.4}]
    fk_models = {}
    for param in parameters:
        fk_models[str(param)] = ssm.GuidedPF(ssm=TwoLegModel(**param), data=y)
    compare_parameters(fk_models=fk_models, nb_timesteps=nb_timesteps, nb_particles=N, nb_runs=nb_runs,
                       t_start=t_start, show_fig=show_fig, export_name=export_name)
