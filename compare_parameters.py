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


def compare_parameters(fk_models, true_states, data, dt, nb_particles, nb_runs, t_start, show_fig, export_name=None):
    start_user, start_process = time.time(), time.process_time()
    results = particles.multiSMC(fk=fk_models, N=nb_particles, nruns=nb_runs, collect=[LogLts()], nprocs=-1)
    end_user, end_process = time.time(), time.process_time()
    print('Time user {:.1f}s; time processor {:.1f}s'.format(end_user - start_user, end_process - start_process))

    plotter = Plotter(np.array(true_states), np.array(data), dt, export_name, show_fig)
    plotter.plot_likelihood_parameters(results, fk_models.keys(), t_start)

    return None


if __name__ == '__main__':
    # ---------------------------- data ----------------------------
    generation_type = 'Missingdata005'
    nb_timesteps = 1000
    dim_obs = 20  # 20 or 36
    x, y = prepare_data(generation_type, nb_timesteps, dim_obs)

    # ---------------------------- model ----------------------------
    # ! model works with parameters defined below; other parameters are chosen as the standard params in TwoLegModelSMC.py
    dt = 0.01
    N = 2000
    t_start = 500
    nb_runs = 50
    show_fig = True
    params = 'pos_imu0'
    export_name = 'MultiRun_{}_steps{}_N{}_nbruns{}_tstart{}_params{}'.format(
        generation_type,
        nb_timesteps,
        N,
        nb_runs,
        t_start,
        params)

    parameters = [{'pos_imu0': 0.15}, {'pos_imu0': 0.20}, {'pos_imu0': 0.25}, {}, {'pos_imu0': 0.35}, {'pos_imu0': 0.4},
                  {'pos_imu0': 0.45}]
    # parameters = [{'pos_imu0': 0.15, 'pos_imu2': 0.15}, {'pos_imu0': 0.2, 'pos_imu2': 0.2}, {'pos_imu0': 0.25, 'pos_imu2': 0.25}, {}, {'pos_imu0': 0.35, 'pos_imu2': 0.35}, {'pos_imu0': 0.4, 'pos_imu2': 0.4}, {'pos_imu0': 0.45, 'pos_imu2': 0.45}]
    # parameters = [{'alpha_0': -0.2}, {'alpha_0': -0.1}, {'alpha_0': 0.0}, {'alpha_0': 0.1}, {'alpha_0': 0.2}, {'alpha_0': 0.3}]
    # parameters = [{'alpha_0': -0.2, 'alpha_2': -0.2}, {'alpha_0': -0.1, 'alpha_2': -0.1}, {'alpha_0': 0.0, 'alpha_2': 0.0}, {'alpha_0': 0.1, 'alpha_2': 0.1}, {'alpha_0': 0.2, 'alpha_2': 0.2}, {'alpha_0': 0.3, 'alpha_2': 0.3}]
    fk_models = {}
    for param in parameters:
        fk_models[str(param)] = ssm.GuidedPF(ssm=TwoLegModel(**param), data=y)
    compare_parameters(fk_models=fk_models, true_states=x, data=y, dt=dt, nb_particles=N, nb_runs=nb_runs,
                       t_start=t_start, show_fig=show_fig, export_name=export_name)
