import numpy as np
import matplotlib.pyplot as plt

from MechanicalModel import MechanicalModel
from DataReaderWriter import DataReaderWriter


def plot_observations(model, x, y_true, supress_zeros, export_name=None):
    y_nonlinear = model.state_to_observation(x)
    xp = np.zeros(x.shape)
    xp[1:, :] = x[:-1, :]
    y_linear = model.state_to_observation_linear(x, xp)
    if supress_zeros:
        y_nonlinear = y_nonlinear[:, (0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34)]
        y_linear = y_linear[:, (0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34)]
        y_true = y_true[:, (0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34)]
        obs_names = ['$\ddot x^0$', '$\ddot y^0$', '$\omega_z^0$',
                     '$\ddot x^1$', '$\ddot y^1$', '$\omega_z^1$',
                     '$\ddot x^2$', '$\ddot y^2$', '$\omega_z^2$',
                     '$\ddot x^3$', '$\ddot y^3$', '$\omega_z^3$',
                     '$\dot x^4$', '$\dot y^4$', '$\ddot x^4$', '$\ddot y^4$',
                     '$\dot x^5$', '$\dot y^5$', '$\ddot x^5$', '$\ddot y^5$']
    else:
        obs_names = ['$\ddot x^0$', '$\ddot y^0$', '$\ddot z^0$', '$\omega_x^0$', '$\omega_y^0$', '$\omega_z^0$',
                     '$\ddot x^1$', '$\ddot y^1$', '$\ddot z^1$', '$\omega_x^1$', '$\omega_y^1$', '$\omega_z^1$',
                     '$\ddot x^2$', '$\ddot y^2$', '$\ddot z^2$', '$\omega_x^2$', '$\omega_y^2$', '$\omega_z^2$',
                     '$\ddot x^3$', '$\ddot y^3$', '$\ddot z^3$', '$\omega_x^3$', '$\omega_y^3$', '$\omega_z^3$',
                     '$\dot x^4$', '$\dot y^4$', '$\dot z^4$', '$\ddot x^4$', '$\ddot y^4$', '$\ddot z^4$',
                     '$\dot x^5$', '$\dot y^5$', '$\dot z^5$', '$\ddot x^5$', '$\ddot y^5$', '$\ddot z^5$']

    nb_steps, nb_observations = y_true.shape
    t_vals = np.linspace(0.0, nb_steps * model.dt, nb_steps)
    nb_axes = 3
    nb_figures = int(np.ceil(nb_observations / nb_axes))
    for i in range(0, nb_figures):
        fig, axs = plt.subplots(ncols=1, nrows=nb_axes, figsize=(12, 8))
        for j in range(0, nb_axes):
            if i * nb_axes + j > nb_observations - 1:
                break
            axs[j].grid(axis='both')
            axs[j].plot(t_vals, y_nonlinear[:, 3 * i + j], label='Non-linear', linewidth=1.0)
            axs[j].plot(t_vals, y_linear[:, 3 * i + j], label='Linear', linewidth=1.0)
            axs[j].plot(t_vals, y_true[:, 3 * i + j], label='Truth', ls=':', alpha=0.8, linewidth=2.0)
            axs[j].legend()
            axs[j].set_title(obs_names[i * nb_axes + j])
        fig.tight_layout()
        if export_name:
            path = "State_Plots/{}_{}.pdf".format(export_name, i)
            plt.savefig(path)
    plt.show()


# ---------- data -----------------
path_truth = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Largetimestep/truth.dat'
path_obs = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Largetimestep/noised_observations.dat'
data_reader = DataReaderWriter()
max_timesteps = 80
data_reader.read_states_as_arr(path_truth, max_timesteps=max_timesteps)
data_reader.read_observations_as_arr(path_obs, max_timesteps=max_timesteps)
data_reader.prepare_lists()
x = data_reader.true_states
y = data_reader.observations

# ---------- model -----------------
dt = 0.1
leg_constants = np.array([0.5, 0.6, 0.5, 0.6])
imu_position = np.array([0.34, 0.29, 0.315, 0.33])
dim_state = 18
dim_observations = 36

my_model = MechanicalModel(dt=dt,
                           dim_states=dim_state,
                           dim_observations=dim_observations,
                           imu_position=imu_position,
                           leg_constants=leg_constants,
                           R=np.eye(dim_observations))

# ---------- plotting -----------------
supress_zeros = True
plot_observations(my_model, x, y, supress_zeros=supress_zeros, export_name=None)
