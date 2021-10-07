import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import particles

from collections import OrderedDict
from particles import distributions as dists
from particles.collectors import Moments
from particles import state_space_models as ssm
from particles import mcmc

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

    P = 0.01 * np.eye(18)

    cov_step = 0.5
    scale_x = 1.0
    scale_y = 2.0
    scale_phi = 5.0

    sf_H = 0.1
    H = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01,
                 0.1, 0.1, 0.1, 0.01, 0.01, 0.01,
                 0.1, 0.1, 0.1, 0.01, 0.01, 0.01,
                 0.1, 0.1, 0.1, 0.01, 0.01, 0.01,
                 0.1, 0.1, 0.1, 1.0, 1.0, 1.0,
                 0.1, 0.1, 0.1, 1.0, 1.0, 1.0])

    my_model = TwoLegModel(dt=dt,
                           leg_constants=leg_constants,
                           imu_position=imu_position,
                           a=a,
                           P=P,
                           cov_step=cov_step,
                           scale_x=scale_x,
                           scale_y=scale_y,
                           scale_phi=scale_phi,
                           sf_H=sf_H,
                           H=H)

    # simulated data from weto
    path_truth = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Normal/truth_normal.dat'
    path_obs = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Normal/noised_observations_normal.dat'
    data_reader = DataReader()
    max_timesteps = 200
    data_reader.read_states_as_arr(path_truth, max_timesteps=max_timesteps)
    data_reader.read_observations_as_arr(path_obs, max_timesteps=max_timesteps)
    data_reader.prepare_lists()
    x_tom = data_reader.states_list
    y_tom = data_reader.observations_list

    # simulate data from this model
    x, y = my_model.simulate(max_timesteps)

    fk_model = ssm.Bootstrap(ssm=my_model, data=y)
    pf = particles.SMC(fk=fk_model, N=200, resampling='stratified', collect=[Moments()], store_history=True)
    pf.run()

    plotting_states = {'x_H': 0, 'y_H': 1, 'phi_0': 2, 'phi_1': 3,
                       'x_H_dot': 6, 'y_H_dot': 7, 'phi_0_dot': 8, 'phi_1_dot': 9,
                       'x_H_ddot': 12, 'y_H_ddot': 13, 'phi_0_ddot': 14, 'phi_1_ddot': 15}
    """
    # plot filtered observations and moments
    for name, idx in plotting_states.items():
        plt.figure()
        plt.plot([yt[0, idx] ** 2 for yt in y], label=name)
        plt.plot([m['mean'][idx] for m in pf.summaries.moments], label='moments')
        plt.legend()
    #plt.show()
    
    # compare MC and QMC method
    results = particles.multiSMC(fk=fk_model, N=100, nruns=30, qmc={'SMC': False, 'SQMC': True})
    plt.figure()
    sb.boxplot(x=[r['output'].logLt for r in results], y=[r['qmc'] for r in results])
    #plt.show()
    
    # smoothing
    smooth_trajectories = pf.hist.backward_sampling(10)
    for name, idx in plotting_states.items():
        plt.figure()
        samples = [time_step[:, idx] for time_step in smooth_trajectories]
        plt.plot(samples)
        truth = [t[0, idx] for t in x]
        plt.plot(truth, label='Truth')
        plt.legend()
        plt.title(name)
    plt.show()
    """
    prior_dict = {'cov_step': dists.Gamma(a=0.8, b=3.0), 'sf_H': dists.Uniform(0.0001, 1.0)}
    my_prior = dists.StructDist(prior_dict)
    pmmh = mcmc.PMMH(ssm_cls=TwoLegModel(dt=dt,
                                         leg_constants=leg_constants,
                                         imu_position=imu_position,
                                         a=a,
                                         P=P,
                                         scale_x=scale_x,
                                         scale_y=scale_y,
                                         scale_phi=scale_phi,
                                         H=H),
                     prior=my_prior, data=y, Nx=50, niter=1000)
    pmmh.run()  # Warning: takes a few seconds

    burnin = 100  # discard the 100 first iterations
    for i, param in enumerate(prior_dict.keys()):
        plt.subplot(2, 2, i + 1)
        sb.distplot(pmmh.chain.theta[param][burnin:], 40)
        plt.title(param)
    plt.show()
