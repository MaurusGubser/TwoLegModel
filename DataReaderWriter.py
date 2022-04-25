import os
import numpy as np
import pandas as pd
from more_itertools import pairwise


class DataReaderWriter:

    def __init__(self):
        self.observations_arr = np.empty(0)
        self.observations_list = []
        self.true_states_arr = np.empty(0)
        self.states_list = []

    def read_states_as_arr(self, path_true_states, max_timesteps):
        truth = np.genfromtxt(path_true_states,
                              dtype=np.float32,
                              comments='#',
                              delimiter=',',
                              skip_header=6,
                              usecols=tuple(range(1, 19)),
                              max_rows=max_timesteps)
        self.true_states_arr = truth
        return None

    def read_observations_as_arr(self, path_observations, max_timesteps):
        col_names = ['time', 'obs1', 'obs2', 'obs3', 'obs4', 'obs5', 'obs6', 'last']
        df = pd.read_csv(path_observations, index_col=None, names=col_names)
        df.drop(columns=col_names[-1], inplace=True)
        df = df[df['time'] != '#time'].reset_index(drop=True)

        pattern_obs = '[#] Observations'
        sensor_idxs = df[df['time'].str.contains(pattern_obs)].index
        sensors_measurements = []
        for i, j in pairwise(sensor_idxs):
            df_sensor = df.iloc[i + 1:j]
            df_sensor.columns = ['time'] + [col_name + '_' + str(i) for col_name in df_sensor.columns[1:]]
            sensors_measurements.append(df_sensor)
        last_idx = sensor_idxs[-1] + 1
        sensors_measurements.append(df[last_idx:])

        observation_df = sensors_measurements[0]
        for df_obs in sensors_measurements[1:]:
            observation_df = pd.merge(observation_df, df_obs, how='left', on='time')

        measurement_arr = observation_df.drop('time', axis=1).to_numpy(dtype=float)
        self.observations_arr = measurement_arr[0:max_timesteps, :]
        return None

    def data_arr_to_list(self):
        if self.observations_arr.shape[0] != self.true_states_arr.shape[0]:
            raise AssertionError('Number of time steps in observations is {}; number of time steps in truth is {};'
                                 'should be the same.'.format(self.observations_arr.shape[0],
                                                              self.true_states_arr.shape[0]))
        nb_timesteps = self.true_states_arr.shape[0]
        for time_step in range(0, nb_timesteps):
            self.states_list.append(np.reshape(self.true_states_arr[time_step, :], (1, -1)))
            self.observations_list.append(np.reshape(self.observations_arr[time_step, :], (1, -1)))
        return None

    def reduce_observations(self):
        self.observations_arr = self.observations_arr[:, (0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34)]
        return None

    def get_data_as_lists(self, generation_type, max_timesteps, dim_observations):
        path_truth = 'GeneratedData/' + generation_type + '/truth.dat'
        path_obs = 'GeneratedData/' + generation_type + '/noised_observations.dat'
        self.read_states_as_arr(path_truth, max_timesteps)
        self.read_observations_as_arr(path_obs, max_timesteps)
        if dim_observations == 20:
            self.reduce_observations()
        self.data_arr_to_list()
        return self.states_list, self.observations_list

    def get_data_as_arr(self, generation_type, max_timesteps, dim_observations):
        path_truth = 'GeneratedData/' + generation_type + '/truth.dat'
        path_obs = 'GeneratedData/' + generation_type + '/noised_observations.dat'
        self.read_states_as_arr(path_truth, max_timesteps)
        self.read_observations_as_arr(path_obs, max_timesteps)
        if dim_observations == 20:
            self.reduce_observations()
        return self.true_states_arr, self.observations_arr

    @staticmethod
    def export_trajectory(data_states, dt, file_name):
        nb_timesteps, nb_samples, _ = data_states.shape
        time_arr = dt * np.arange(1, nb_timesteps + 1)
        header = '\nnb_samples: {}\n'.format(nb_timesteps) + 4 * 'nb_times: {}\n'.format(nb_timesteps)
        header = header + '# time,x_0,x_1,phi femur_l,phi fibula_l,phi femur_r,phi fibula_r,dx_0,dx_1,dphi femur_l,dphi fibula_l,dphi femur_r,dphi fibula_r,ddx_0,ddx_1,ddphi femur_l,ddphi fibula_l,ddphi femur_r,ddphi fibula_r\n'

        if not os.path.exists('AnimationSamples'):
            os.mkdir('AnimationSamples')
        if not os.path.exists('AnimationSamples/{}'.format(file_name)):
            os.mkdir('AnimationSamples/{}'.format(file_name))

        for i in range(0, nb_samples):
            export_name = 'AnimationSamples/{}/sample_{}.dat'.format(file_name, i)
            with open(export_name, 'w') as f_write:
                f_write.write(header)
            sample = data_states[:, i, :]
            for j in range(0, nb_timesteps):
                data_list = [str(time_arr[j])] + [str(state) for state in sample[j, :]]
                data_str = ','.join(data_list) + ',\n'
                with open(export_name, 'a') as f_write:
                    f_write.write(data_str)
            time_list = 4 * [str(time) for time in time_arr]
            time_str = '\n'.join(time_list)
            with open(export_name, 'a') as f_write:
                f_write.write(time_str)
        print('Sampled states exported to AnimationSamples/{}'.format(file_name))
        return None

    @staticmethod
    def export_parameters(alg, prior_dict, file_name):
        data = np.array([alg.chain.theta[param] for param in prior_dict.keys()])
        row_names = list(prior_dict.keys())
        if not os.path.exists('ParameterLearning'):
            os.mkdir('ParameterLearning')
        export_path = 'ParameterLearning/' + file_name + '.csv'
        df = pd.DataFrame(data=data, index=row_names)
        df.to_csv(export_path)
        print('Learned parameters exported to ParameterLearning/{}'.format(file_name))
        return None
