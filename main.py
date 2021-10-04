import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import particles

from collections import OrderedDict
from particles.collectors import Moments
from particles import state_space_models as ssm

from TwoLegModel import TwoLegModel

# Press the green button in the gutter to run the script.
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
    Q = 0.1 * np.eye(18)
    H = 0.1 * np.eye(36)

    my_model = TwoLegModel(dt=dt, leg_constants=leg_constants, imu_position=imu_position, a=a, P=P, Q=Q, H=H)
    x, y = my_model.simulate(100)
    data = [np.array([step[0, 0]]) for step in y]

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(data)
    plt.show()

    models = OrderedDict()
    models['boot'] = ssm.Bootstrap(ssm=my_model, data=y)
    #models['guided'] = ssm.GuidedPF(ssm=my_model, data=y)
    results = particles.multiSMC(fk=models, N=100, nruns=10, nprocs=1, store_history=True)

    dummy = 0.0

    fk_model = ssm.Bootstrap(ssm=my_model, data=y)  # we use the Bootstrap filter
    pf = particles.SMC(fk=fk_model, N=100, resampling='stratified', store_history=True)  # the algorithm
    pf.run()  # actual computation


    """
    plt.style.use('ggplot')
    plt.plot(y)
    plt.xlabel('t')
    plt.ylabel('data');
    """
