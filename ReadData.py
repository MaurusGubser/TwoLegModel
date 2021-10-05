import numpy as np
import pandas as pd
import csv
import re


class DataReader():

    def __init__(self):
        self.observations = np.empty(0)
        self.true_states = np.empty(0)

    def read_true_states(self, path_true_states, nb_timesteps):
        truth = np.genfromtxt(path_true_states,
                              dtype=np.float32,
                              comments='#',
                              delimiter=',',
                              skip_header=6,
                              usecols=tuple(range(1, 19)),
                              max_rows=nb_timesteps)
        self.true_states = truth
        return None

    def read_observations(self, path_observations):
        obs = np.genfromtxt(path_observations,
                            dtype=np.float32,
                            comments='#',
                            delimiter=',',
                            skip_header=1,
                            usecols=tuple(range(1, 7)))
        if obs.shape[0] % 6 != 0:
            raise ArithmeticError('Data array should have 6 measurement series; number of entries {} was not divisible'
                                  'by 6.'.format(obs.shape[0]))
        # reshape observations
        nb_timesteps = obs.shape[0] // 6
        observations_array = np.empty((nb_timesteps, 36))
        for i in range(0, 6):
            observations_array[:, 6 * i:6 * (i + 1)] = obs[nb_timesteps * i: nb_timesteps * (i + 1), :]
        self.observations = observations_array

path_truth = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Normal/truth_normal.dat'
path_obs = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Normal/noised_observations_normal.dat'
reader = DataReader()
reader.read_true_states(path_truth, nb_timesteps=1014)
reader.read_observations(path_obs)
dummy=0