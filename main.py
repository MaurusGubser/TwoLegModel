import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import particles

from collections import OrderedDict
from particles.collectors import Moments
from particles import state_space_models as ssm

from ReadData import DataReader
from TwoLegModel import TwoLegModel

if __name__ == '__main__':
    dt = 0.01
    leg_constants = np.array([0.5, 0.6, 0.5, 0.6])
    imu_position = np.array([0.34, 0.29, 0.315, 0.33])
    a = np.array([5.6790326970162603e-03, 1.0575269992136509e+00, -1.2846265995420103e-01, -2.4793110500096724e-01,
                  3.6639668719776680e-01, -1.8980094976695036e-01, 5.6790326966931337e-01, 9.6320242403311385e-02,
                  2.5362910623050072e+00, -3.7986101392570708e+00, -7.8163469474606714e-02, -8.1819333353243029e-01,
                  -4.0705228907187886e-11, 5.0517984683954827e-03, -1.7762296102838229e+00, 3.3158529817439670e+00,
                  -2.9528844960512168e-01, 5.3581371545316991e-01])
    P = 10.0 * np.eye(18)
    Q = 0.01 * np.eye(18)
    H = 0.1 * np.eye(36)

    my_model = TwoLegModel(dt=dt, leg_constants=leg_constants, imu_position=imu_position, a=a, P=P, Q=Q, H=H)

    path_truth = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Normal/truth_normal.dat'
    path_obs = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Normal/noised_observations_normal.dat'
    data_reader = DataReader()
    nb_timesteps = 1014
    data_reader.read_states_as_arr(path_truth, nb_timesteps=nb_timesteps)
    data_reader.read_observations_as_arr(path_obs)
    data_reader.prepare_lists()
    x = data_reader.states_list
    y = data_reader.observations_list

    # simulate data from model
    x_sim, y_sim = my_model.simulate(100)

    plotting_states = {'x_H': 0, 'y_H': 1, 'phi_0': 2, 'phi_1': 3}
    """
    for name, idx in plotting_states.items():
        plotting_data = [np.array([step[0, idx]]) for step in y]
        plt.figure()
        plt.plot(plotting_data, label=name)
        plt.legend()
    plt.show()
    """

    # particle filter
    fk_model = ssm.Bootstrap(ssm=my_model, data=y)  # we use the Bootstrap filter
    pf = particles.SMC(fk=fk_model, N=100, resampling='stratified', store_history=True)  # the algorithm
    # results = particles.multiSMC(fk=models, N=100, nruns=10, nprocs=1, collect=[Moments()], store_history=True)   # neuro variant, multiSMC for comparing SMCs with different parameters
    pf.run()  # actual computation
    """
    for name, idx in plotting_states.items():
        plotting_data = [np.array([step[0, idx]]) for step in y]
        plt.figure()
        plt.plot([yt ** 2 for yt in plotting_data], label='data-squared')
        plt.plot(pf.summaries.logLts, label='filtered volatility') #[m['mean'] for m in pf.summaries.logLts]
        plt.legend()
        plt.suptitle(name)
    plt.show()
    """
    smooth_trajectories = pf.hist.backward_sampling(10)
    # print('acceptance rate was %1.3f' % acc_rate)
    for name, idx in plotting_states.items():
        plotting_data = [state[:, idx] for state in smooth_trajectories]
        plt.plot(plotting_data, label=name)
        plt.legend()
        plt.show()
