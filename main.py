import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import particles

from collections import OrderedDict
from particles.collectors import Moments
from particles import state_space_models as ssm

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
    P = 0.1 * np.eye(18)
    A = np.eye(18)
    for row in range(0, 18):
        for col in range(0, 18):
            if row + 6 == col:
                A[row, col] = dt
            elif row + 12 == col:
                A[row, col] = dt**2 / 2.0
    Q = 0.1 * np.eye(18)
    H = 1.0 * np.eye(36)

    my_model = TwoLegModel(dt=dt, legs=leg_constants, cst=imu_position, a=a, P=P, A=A, Q=Q, H=H, g=9.81)

    # simulate data from model
    x, y = my_model.simulate(100)

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
    more_trajectories = pf.hist.backward_sampling(10, linear_cost=False, return_ar=True)
    #print('acceptance rate was %1.3f' % acc_rate)
    for name, idx in plotting_states.items():
        plotting_data = [state[:, idx] for state in more_trajectories]
        plt.plot(plotting_data, label=name)
        plt.legend()
        plt.show()
