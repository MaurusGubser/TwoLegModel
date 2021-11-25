import numpy as np
import pandas as pd
from more_itertools import pairwise


class DataReader:

    def __init__(self):
        self.observations = np.empty(0)
        self.observations_list = []
        self.true_states = np.empty(0)
        self.states_list = []

    def read_states_as_arr(self, path_true_states, max_timesteps):
        truth = np.genfromtxt(path_true_states,
                              dtype=np.float32,
                              comments='#',
                              delimiter=',',
                              skip_header=6,
                              usecols=tuple(range(1, 19)),
                              max_rows=max_timesteps)
        self.true_states = truth
        return None

    def read_observations_as_arr(self, path_observations, max_timesteps):
        col_names = ['time', 'obs1', 'obs2', 'obs3', 'obs4', 'obs5', 'obs6']
        df = pd.read_csv(path_observations, index_col=None, names=col_names)
        df = df[df['time'] != '#time'].reset_index(drop=True)

        pattern_obs = '[#] Observations'
        sensor_idxs = df[df['time'].str.contains(pattern_obs)].index
        sensors_measurements = []
        for i, j in pairwise(sensor_idxs):
            df_sensor = df.iloc[i + 1:j]
            sensors_measurements.append(df_sensor)
        last_idx = sensor_idxs[-1] + 1
        sensors_measurements.append(df[last_idx:])

        observation_df = sensors_measurements[0]
        for df_obs in sensors_measurements[1:]:
            observation_df = pd.merge(observation_df, df_obs, how='left', on='time')

        measurement_arr = observation_df.drop('time', axis=1).to_numpy(dtype=float)
        self.observations = measurement_arr[0:max_timesteps, :]
        return None

    def prepare_lists(self):
        if self.observations.shape[0] != self.true_states.shape[0]:
            raise AssertionError('Number of time steps in observations is {}; number of time steps in truth is {};'
                                 'should be the same.'.format(self.observations.shape[0], self.true_states.shape[0]))
        nb_timesteps = self.true_states.shape[0]
        for time_step in range(0, nb_timesteps):
            self.states_list.append(np.reshape(self.true_states[time_step, :], (1, -1)))
            self.observations_list.append(np.reshape(self.observations[time_step, :], (1, -1)))
        return None
