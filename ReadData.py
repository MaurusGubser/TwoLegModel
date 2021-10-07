import numpy as np


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
        observations_array = observations_array[0:max_timesteps, :]
        self.observations = observations_array

    def prepare_lists(self):
        if self.observations.shape[0] != self.true_states.shape[0]:
            raise AssertionError('Number of time steps in observations is {}; number of time steps in truth is {};'
                                 'should be the same.'.format(self.observations.shape[0], self.true_states.shape[0]))
        nb_timesteps = self.true_states.shape[0]
        for time_step in range(0, nb_timesteps):
            self.states_list.append(np.reshape(self.true_states[time_step, :], (1, -1)))
            self.observations_list.append(np.reshape(self.observations[time_step, :], (1, -1)))
        return None
