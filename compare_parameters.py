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


def compare_parameters(fk_models, true_states, data, dt, nb_particles, nb_runs, t_start, show_fig, export_name=None):
    start_user, start_process = time.time(), time.process_time()
    results = particles.multiSMC(fk=fk_models, N=nb_particles, nruns=nb_runs, collect=[LogLts()], nprocs=-1)
    end_user, end_process = time.time(), time.process_time()
    s_user = end_user - start_user
    s_process = end_process - start_process
    print('Time user {:.0f}min {:.0f}s; time processor {:.0f}min {:.0f}s'.format(s_user // 60, s_user % 60,
                                                                                 s_process // 60, s_process % 60))

    plotter = Plotter(np.array(true_states), np.array(data), dt, export_name, show_fig)
    plotter.plot_likelihood_parameters(results, fk_models.keys(), t_start)

    return None


if __name__ == '__main__':
    # ---------------------------- data ----------------------------
    generation_type = 'Missingdata005'
    nb_timesteps = 1000
    dim_obs = 20  # 20 or 36
    data_reader = DataReaderWriter()
    x, y = data_reader.get_data_as_lists(generation_type, nb_timesteps, dim_obs)

    # ---------------------------- model ----------------------------
    # ! model works with parameters defined below; other parameters are chosen as the standard params in TwoLegModelSMC.py
    dt = 0.01
    N = 2000
    t_start = 500
    nb_runs = 50
    show_fig = True
    params = 'legs'
    export_name = 'MultiRun_{}_steps{}_N{}_nbruns{}_tstart{}_params{}'.format(
        generation_type,
        nb_timesteps,
        N,
        nb_runs,
        t_start,
        params)

    parameters = [{'femur_left': 0.3, 'femur_right': 0.3}, {'femur_left': 0.4, 'femur_right': 0.4}, {'femur_left': 0.5, 'femur_right': 0.5}, {'femur_left': 0.6, 'femur_right': 0.6}, {'femur_left': 0.7, 'femur_right': 0.7}]
    # parameters = [{'femur_left': 0.3}, {'femur_left': 0.4}, {'femur_left': 0.5}, {'femur_left': 0.6}, {'femur_left': 0.7}]
    # parameters = [{'factor_proposal': 1.0}, {'factor_proposal': 1.05}, {'factor_proposal': 1.1}, {'factor_proposal': 1.15}, {'factor_proposal': 1.2}, {'factor_proposal': 1.25}, {'factor_proposal': 1.3}, {'factor_proposal': 1.35}, {'factor_proposal': 1.4}]
    # parameters = [{'pos_imu0': 0.15}, {'pos_imu0': 0.20}, {'pos_imu0': 0.25}, {}, {'pos_imu0': 0.35}, {'pos_imu0': 0.4}, {'pos_imu0': 0.45}]
    # parameters = [{'pos_imu0': 0.15, 'pos_imu2': 0.15}, {'pos_imu0': 0.2, 'pos_imu2': 0.2}, {'pos_imu0': 0.25, 'pos_imu2': 0.25}, {}, {'pos_imu0': 0.35, 'pos_imu2': 0.35}, {'pos_imu0': 0.4, 'pos_imu2': 0.4}, {'pos_imu0': 0.45, 'pos_imu2': 0.45}]
    # parameters = [{'alpha_0': -0.2}, {'alpha_0': -0.1}, {'alpha_0': 0.0}, {'alpha_0': 0.1}, {'alpha_0': 0.2}, {'alpha_0': 0.3}]
    # parameters = [{'alpha_0': -0.2, 'alpha_2': -0.2}, {'alpha_0': -0.1, 'alpha_2': -0.1}, {'alpha_0': 0.0, 'alpha_2': 0.0}, {'alpha_0': 0.1, 'alpha_2': 0.1}, {'alpha_0': 0.2, 'alpha_2': 0.2}, {'alpha_0': 0.3, 'alpha_2': 0.3}]
    fk_models = {}
    for param in parameters:
        fk_models[str(param)] = ssm.GuidedPF(ssm=TwoLegModel(**param), data=y)
    compare_parameters(fk_models=fk_models, true_states=x, data=y, dt=dt, nb_particles=N, nb_runs=nb_runs,
                       t_start=t_start, show_fig=show_fig, export_name=export_name)
