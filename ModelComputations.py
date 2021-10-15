import numpy as np
import matplotlib.pyplot as plt

from TwoLegSMCModel import TwoLegModel
from ReadData import DataReader


def state_to_observation_nonlinear(model, x):
    nb_steps, _ = x.shape
    y = np.empty(shape=(nb_steps, 36))
    # left femur
    y[:, 0] = model.cst[0] * x[:, 14] + model.g * np.sin(x[:, 2]) + np.sin(x[:, 2]) * x[:, 13] + np.cos(x[:, 2]) \
              * x[:, 12]
    y[:, 1] = model.cst[0] * x[:, 8] ** 2 + model.g * np.cos(x[:, 2]) - np.sin(x[:, 2]) * x[:, 12] \
              + np.cos(x[:, 2]) * x[:, 13]
    y[:, 2] = 0.0
    y[:, 3] = 0.0
    y[:, 4] = 0.0
    y[:, 5] = x[:, 8]

    # left fibula
    y[:, 6] = model.cst[1] * x[:, 14] + model.cst[1] * x[:, 15] + model.g * np.sin(x[:, 2] + x[:, 3]) + model.legs[0] \
              * np.sin(x[:, 3]) * x[:, 8] ** 2 + model.legs[0] * np.cos(x[:, 3]) * x[:, 14] \
              + np.sin(x[:, 2] + x[:, 3]) * x[:, 13] + np.cos(x[:, 2] + x[:, 3]) * x[:, 12]
    y[:, 7] = model.cst[1] * x[:, 8] ** 2 + 2 * model.cst[1] * x[:, 8] * x[:, 9] + model.cst[1] * x[:, 9] ** 2 \
              + model.g * np.cos(x[:, 2] + x[:, 3]) - model.legs[0] * np.sin(x[:, 3]) * x[:, 14] + model.legs[0] * \
              np.cos(x[:, 3]) * x[:, 8] ** 2 - np.sin(x[:, 2] + x[:, 3]) * x[:, 12] + np.cos(x[:, 2] + x[:, 3]) \
              * x[:, 13]
    y[:, 8] = 0.0
    y[:, 9] = 0.0
    y[:, 10] = 0.0
    y[:, 11] = x[:, 8] + x[:, 9]

    # right femur
    y[:, 12] = model.cst[2] * x[:, 16] + model.g * np.sin(x[:, 4]) + np.sin(x[:, 4]) * x[:, 13] + np.cos(x[:, 4]) \
               * x[:, 12]
    y[:, 13] = model.cst[2] * x[:, 10] ** 2 + model.g * np.cos(x[:, 4]) - np.sin(x[:, 4]) * x[:, 12] + np.cos(x[:, 4]) \
               * x[:, 13]
    y[:, 14] = 0.0
    y[:, 15] = 0.0
    y[:, 16] = 0.0
    y[:, 17] = x[:, 10]

    # right fibula
    y[:, 18] = model.cst[3] * x[:, 16] + model.cst[3] * x[:, 17] + model.g * np.sin(x[:, 4] + x[:, 5]) + model.legs[2] \
               * np.sin(x[:, 5]) * x[:, 10] ** 2 + model.legs[2] * np.cos(x[:, 5]) * x[:, 16] \
               + np.sin(x[:, 4] + x[:, 5]) * x[:, 13] + np.cos(x[:, 4] + x[:, 5]) * x[:, 12]
    y[:, 19] = model.cst[3] * x[:, 10] ** 2 + 2 * model.cst[3] * x[:, 10] * x[:, 11] + model.cst[3] * x[:, 11] ** 2 \
               + model.g * np.cos(x[:, 4] + x[:, 5]) - model.legs[2] * np.sin(x[:, 5]) * x[:, 16] + model.legs[2] \
               * np.cos(x[:, 5]) * x[:, 10] ** 2 - np.sin(x[:, 4] + x[:, 5]) * x[:, 12] + np.cos(x[:, 4] + x[:, 5]) \
               * x[:, 13]
    y[:, 20] = 0.0
    y[:, 21] = 0.0
    y[:, 22] = 0.0
    y[:, 23] = x[:, 10] + x[:, 11]

    # left heel
    y[:, 24] = model.legs[0] * np.cos(x[:, 2]) * x[:, 8] + model.legs[1] * (x[:, 8] + x[:, 9]) \
               * np.cos(x[:, 2] + x[:, 3]) + x[:, 6]
    y[:, 25] = model.legs[0] * np.sin(x[:, 2]) * x[:, 8] + model.legs[1] * (x[:, 8] + x[:, 9]) \
               * np.sin(x[:, 2] + x[:, 3]) + x[:, 7]
    y[:, 26] = 0.0
    y[:, 27] = -model.legs[0] * np.sin(x[:, 2]) * x[:, 8] ** 2 + model.legs[0] * np.cos(x[:, 2]) * x[:, 14] \
               - model.legs[1] * (x[:, 8] + x[:, 9]) ** 2 * np.sin(x[:, 2] + x[:, 3]) + model.legs[1] \
               * (x[:, 14] + x[:, 15]) * np.cos(x[:, 2] + x[:, 3]) + x[:, 12]
    y[:, 28] = model.legs[0] * np.sin(x[:, 2]) * x[:, 14] + model.legs[0] * np.cos(x[:, 2]) * x[:, 8] ** 2 \
               + model.legs[1] * (x[:, 8] + x[:, 9]) ** 2 * np.cos(x[:, 2] + x[:, 3]) + model.legs[1] \
               * (x[:, 14] + x[:, 15]) * np.sin(x[:, 2] + x[:, 3]) + x[:, 13]
    y[:, 29] = 0.0

    # right heel
    y[:, 30] = model.legs[2] * np.cos(x[:, 4]) * x[:, 10] + model.legs[3] * (x[:, 10] + x[:, 11]) \
               * np.cos(x[:, 4] + x[:, 5]) + x[:, 6]
    y[:, 31] = model.legs[2] * np.sin(x[:, 4]) * x[:, 10] + model.legs[3] * (x[:, 10] + x[:, 11]) \
               * np.sin(x[:, 4] + x[:, 5]) + x[:, 7]
    y[:, 32] = 0.0
    y[:, 33] = -model.legs[2] * np.sin(x[:, 4]) * x[:, 10] ** 2 + model.legs[2] * np.cos(x[:, 4]) * x[:, 16] - \
               model.legs[3] * (x[:, 10] + x[:, 11]) ** 2 * np.sin(x[:, 4] + x[:, 5]) + model.legs[3] \
               * (x[:, 16] + x[:, 17]) * np.cos(x[:, 4] + x[:, 5]) + x[:, 12]
    y[:, 34] = model.legs[2] * np.sin(x[:, 4]) * x[:, 16] + model.legs[2] * np.cos(x[:, 4]) * x[:, 10] ** 2 + \
               model.legs[3] * (x[:, 10] + x[:, 11]) ** 2 * np.cos(x[:, 4] + x[:, 5]) + model.legs[3] \
               * (x[:, 16] + x[:, 17]) * np.sin(x[:, 4] + x[:, 5]) + x[:, 13]
    y[:, 35] = 0.0
    if DIM_OBSERVATIONS == 20:
        y = y[:, (0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34), ]

    return y


def compute_observation_derivatives_1dim(model, x):
    df = np.zeros((36, DIM_STATES))

    # left femur
    df[0, 2] = -x[12] * np.sin(x[2]) + (x[13] + model.g) * np.cos(x[2])
    df[0, 12] = np.cos(x[2])
    df[0, 13] = np.sin(x[2])
    df[0, 14] = model.cst[0]
    df[1, 2] = -x[12] * np.cos(x[2]) - (x[13] + model.g) * np.sin(x[2])
    df[1, 8] = 2 * x[8] * model.cst[0]
    df[1, 12] = -np.sin(x[2])
    df[1, 13] = np.cos(x[2])
    df[5, 8] = 1

    # left fibula
    df[6, 2] = -x[12] * np.sin(x[2] + x[3]) + x[13] * np.cos(x[2] + x[3]) + model.g * np.cos(x[2] + x[3])
    df[6, 3] = -x[14] * model.legs[0] * np.sin(x[3]) - x[12] * np.sin(x[2] + x[3]) + x[13] * np.cos(
        x[2] + x[3]) + model.legs[0] * x[8] ** 2 * np.cos(x[3]) + model.g * np.cos(x[2] + x[3])
    df[6, 8] = 2 * model.legs[0] * x[8] * np.sin(x[3])
    df[6, 12] = np.cos(x[2] + x[3])
    df[6, 13] = np.sin(x[2] + x[3])
    df[6, 14] = model.cst[1] + model.legs[0] * np.cos(x[3])
    df[6, 15] = model.cst[1]
    df[7, 2] = -x[12] * np.cos(x[2] + x[3]) - x[13] * np.sin(x[2] + x[3]) - model.g * np.sin(x[2] + x[3])
    df[7, 3] = -x[14] * model.legs[0] * np.cos(x[3]) - x[12] * np.cos(x[2] + x[3]) - x[13] * np.sin(
        x[2] + x[3]) - model.legs[0] * x[8] ** 2 * np.sin(x[3]) - model.g * np.sin(x[2] + x[3])
    df[7, 8] = 2 * model.legs[0] * x[8] * np.cos(x[3]) + 2 * model.cst[1] * (x[8] + x[9])
    df[7, 9] = 2 * model.cst[1] * (x[8] + x[9])
    df[7, 12] = -np.sin(x[2] + x[3])
    df[7, 13] = np.cos(x[2] + x[3])
    df[7, 14] = -model.legs[0] * np.sin(x[3])
    df[11, 8] = 1
    df[11, 9] = 1

    # right femur
    df[12, 4] = -x[12] * np.sin(x[4]) + (x[13] + model.g) * np.cos(x[4])
    df[12, 12] = np.cos(x[4])
    df[12, 13] = np.sin(x[4])
    df[12, 16] = model.cst[2]
    df[13, 4] = -x[12] * np.cos(x[4]) - (x[13] + model.g) * np.sin(x[4])
    df[13, 10] = 2 * x[10] * model.cst[2]
    df[13, 12] = -np.sin(x[4])
    df[13, 13] = np.cos(x[4])
    df[17, 10] = 1

    # right fibula
    df[18, 4] = -x[12] * np.sin(x[4] + x[5]) + x[13] * np.cos(x[4] + x[5]) + model.g * np.cos(x[4] + x[5])
    df[18, 5] = -x[16] * model.legs[2] * np.sin(x[5]) - x[12] * np.sin(x[4] + x[5]) + x[13] * np.cos(
        x[4] + x[5]) + model.legs[2] * x[10] ** 2 * np.cos(x[5]) + model.g * np.cos(x[4] + x[5])
    df[18, 10] = 2 * model.legs[2] * x[10] * np.sin(x[5])
    df[18, 12] = np.cos(x[4] + x[5])
    df[18, 13] = np.sin(x[4] + x[5])
    df[18, 16] = model.cst[3] + model.legs[2] * np.cos(x[5])
    df[18, 17] = model.cst[3]
    df[19, 4] = -x[12] * np.cos(x[4] + x[5]) - x[13] * np.sin(x[4] + x[5]) - model.g * np.sin(x[4] + x[5])
    df[19, 5] = -x[16] * model.legs[2] * np.cos(x[5]) - x[12] * np.cos(x[4] + x[5]) - x[13] * np.sin(
        x[4] + x[5]) - model.legs[2] * x[10] ** 2 * np.sin(x[5]) - model.g * np.sin(x[4] + x[5])
    df[19, 10] = 2 * model.legs[2] * x[10] * np.cos(x[5]) + 2 * model.cst[3] * (x[10] + x[11])
    df[19, 11] = 2 * model.cst[3] * (x[10] + x[11])
    df[19, 12] = -np.sin(x[4] + x[5])
    df[19, 13] = np.cos(x[4] + x[5])
    df[19, 16] = -model.legs[2] * np.sin(x[5])
    df[23, 10] = 1
    df[23, 11] = 1

    # left heel
    df[24, 2] = -x[8] * model.legs[0] + np.sin(x[2]) - model.legs[1] * (x[8] + x[9]) * np.sin(x[2] + x[3])
    df[24, 3] = -model.legs[1] * (x[8] + x[9]) * np.sin(x[2] + x[3])
    df[24, 6] = 1
    df[24, 8] = model.legs[0] * np.cos(x[2]) + model.legs[1] * np.cos(x[2] + x[3])
    df[24, 9] = model.legs[1] * np.cos(x[2] + x[3])
    df[25, 2] = x[8] * model.legs[0] * np.cos(x[2]) + model.legs[1] * (x[8] + x[9]) * np.cos(x[2] + x[3])
    df[25, 3] = model.legs[1] * (x[8] + x[9]) * np.cos(x[2] + x[3])
    df[25, 7] = 1
    df[25, 8] = model.legs[0] * np.sin(x[2]) + model.legs[1] * np.sin(x[2] + x[3])
    df[25, 9] = model.legs[1] * np.sin(x[2] + x[3])
    df[27, 2] = -model.legs[0] * (x[14] * np.sin(x[2]) + x[8] ** 2 * np.cos(x[2])) - model.legs[1] * (
            x[14] + x[15]) * np.sin(x[2] + x[3]) - model.legs[1] * (x[8] + x[9]) ** 2 * np.cos(
        x[2] + x[3])
    df[27, 3] = -model.legs[1] * (x[14] + x[15]) * np.sin(x[2] + x[3]) - model.legs[1] * (
            x[8] + x[9]) ** 2 * np.cos(x[2] + x[3])
    df[27, 8] = -2 * model.legs[0] * x[8] * np.sin(x[2]) - 2 * model.legs[1] * (x[8] + x[9]) * np.sin(
        x[2] + x[3])
    df[27, 9] = -2 * model.legs[1] * (x[8] + x[9]) * np.sin(x[2] + x[3])
    df[27, 12] = 1
    df[27, 14] = model.legs[0] * np.cos(x[2]) + model.legs[1] * np.cos(x[2] + x[3])
    df[27, 15] = model.legs[1] * np.cos(x[2] + x[3])
    df[28, 2] = -model.legs[0] * (-x[14] * np.cos(x[2]) + x[8] ** 2 * np.sin(x[2])) + model.legs[1] * (
            x[14] + x[15]) * np.cos(x[2] + x[3]) - model.legs[1] * (x[8] + x[9]) ** 2 * np.sin(
        x[2] + x[3])
    df[28, 3] = model.legs[1] * (x[14] + x[15]) * np.cos(x[2] + x[3]) - model.legs[1] * (
            x[8] + x[9]) ** 2 * np.sin(x[2] + x[3])
    df[28, 8] = 2 * model.legs[0] * x[8] * np.cos(x[2]) + 2 * model.legs[1] * (x[8] + x[9]) * np.cos(
        x[2] + x[3])
    df[28, 9] = 2 * model.legs[1] * (x[8] + x[9]) * np.cos(x[2] + x[3])
    df[28, 13] = 1
    df[28, 14] = model.legs[0] * np.sin(x[2]) + model.legs[1] * np.sin(x[2] + x[3])
    df[28, 15] = model.legs[1] * np.sin(x[2] + x[3])

    # right heel
    df[30, 4] = -x[10] * model.legs[2] + np.sin(x[4]) - model.legs[3] * (x[10] + x[11]) * np.sin(x[4] + x[5])
    df[30, 5] = -model.legs[3] * (x[10] + x[11]) * np.sin(x[4] + x[5])
    df[30, 6] = 1
    df[30, 10] = model.legs[2] * np.cos(x[4]) + model.legs[3] * np.cos(x[4] + x[5])
    df[30, 11] = model.legs[3] * np.cos(x[4] + x[5])
    df[31, 4] = x[10] * model.legs[2] * np.cos(x[4]) + model.legs[3] * (x[10] + x[11]) * np.cos(x[4] + x[5])
    df[31, 5] = model.legs[3] * (x[10] + x[11]) * np.cos(x[4] + x[5])
    df[31, 7] = 1
    df[31, 10] = model.legs[2] * np.sin(x[4]) + model.legs[3] * np.sin(x[4] + x[5])
    df[31, 11] = model.legs[3] * np.sin(x[4] + x[5])
    df[33, 4] = -model.legs[2] * (x[16] * np.sin(x[4]) + x[10] ** 2 * np.cos(x[4])) - model.legs[3] * (
            x[16] + x[17]) * np.sin(x[4] + x[5]) - model.legs[3] * (x[10] + x[11]) ** 2 * np.cos(
        x[4] + x[5])
    df[33, 5] = -model.legs[3] * (x[16] + x[17]) * np.sin(x[4] + x[5]) - model.legs[3] * (
            x[10] + x[11]) ** 2 * np.cos(x[4] + x[5])
    df[33, 10] = -2 * model.legs[2] * x[10] * np.sin(x[4]) - 2 * model.legs[3] * (x[10] + x[11]) * np.sin(
        x[4] + x[5])
    df[33, 11] = -2 * model.legs[3] * (x[10] + x[11]) * np.sin(x[4] + x[5])
    df[33, 12] = 1
    df[33, 16] = model.legs[2] * np.cos(x[4]) + model.legs[3] * np.cos(x[4] + x[5])
    df[33, 17] = model.legs[3] * np.cos(x[4] + x[5])
    df[34, 4] = -model.legs[2] * (-x[16] * np.cos(x[4]) + x[10] ** 2 * np.sin(x[4])) + model.legs[3] * (
            x[16] + x[17]) * np.cos(x[4] + x[5]) - model.legs[3] * (x[10] + x[11]) ** 2 * np.sin(
        x[4] + x[5])
    df[34, 5] = model.legs[3] * (x[16] + x[17]) * np.cos(x[4] + x[5]) - model.legs[3] * (
            x[10] + x[11]) ** 2 * np.sin(x[4] + x[5])
    df[34, 10] = 2 * model.legs[2] * x[10] * np.cos(x[4]) + 2 * model.legs[3] * (x[10] + x[11]) * np.cos(
        x[4] + x[5])
    df[34, 11] = 2 * model.legs[3] * (x[10] + x[11]) * np.cos(x[4] + x[5])
    df[34, 13] = 1
    df[34, 16] = model.legs[2] * np.sin(x[4]) + model.legs[3] * np.sin(x[4] + x[5])
    df[34, 17] = model.legs[3] * np.sin(x[4] + x[5])

    if DIM_OBSERVATIONS == 20:
        df = df[(0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34), :]

    return df


def state_to_observation_linear(model, x):
    nb_steps, _ = x.shape
    y = np.empty(shape=(nb_steps, DIM_OBSERVATIONS))
    for i in range(0, nb_steps):
        df = compute_observation_derivatives_1dim(model, x[i])
        y[i, :] = np.matmul(df, x[i])
    if DIM_OBSERVATIONS == 36:
        y[:, (1, 7, 13, 19)] += model.g
    elif DIM_OBSERVATIONS == 20:
        y[:, (1, 4, 7, 10)] += model.g
    else:
        raise AssertionError('Observations must have dimension 20 or 36; got {}.'.format(DIM_OBSERVATIONS))
    return y


def plot_observations(model, x, y_true, export_name=None):
    y_nonlinear = state_to_observation_nonlinear(model, x)
    y_linear = state_to_observation_linear(model, x)
    if DIM_OBSERVATIONS == 20:
        y_true = y_true[:, (0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34)]
    nb_steps, nb_observations = y_true.shape

    # -------------------------------------------------

    if nb_observations == 36:
        obs_names = ['$\ddot x^0$', '$\ddot y^0$', '$\ddot z^0$', '$\omega_x^0$', '$\omega_y^0$', '$\omega_z^0$',
                     '$\ddot x^1$', '$\ddot y^1$', '$\ddot z^1$', '$\omega_x^1$', '$\omega_y^1$', '$\omega_z^1$',
                     '$\ddot x^2$', '$\ddot y^2$', '$\ddot z^2$', '$\omega_x^2$', '$\omega_y^2$', '$\omega_z^2$',
                     '$\ddot x^3$', '$\ddot y^3$', '$\ddot z^3$', '$\omega_x^3$', '$\omega_y^3$', '$\omega_z^3$',
                     '$\dot x^4$', '$\dot y^4$', '$\dot z^4$', '$\ddot x^4$', '$\ddot y^4$', '$\ddot z^4$',
                     '$\dot x^5$', '$\dot y^5$', '$\dot z^5$', '$\ddot x^5$', '$\ddot y^5$', '$\ddot z^5$']
    elif nb_observations == 20:
        obs_names = ['$\ddot x^0$', '$\ddot y^0$', '$\omega_z^0$',
                     '$\ddot x^1$', '$\ddot y^1$', '$\omega_z^1$',
                     '$\ddot x^2$', '$\ddot y^2$', '$\omega_z^2$',
                     '$\ddot x^3$', '$\ddot y^3$', '$\omega_z^3$',
                     '$\dot x^4$', '$\dot y^4$', '$\ddot x^4$', '$\ddot y^4$',
                     '$\dot x^5$', '$\dot y^5$', '$\ddot x^5$', '$\ddot y^5$']
    else:
        raise AssertionError('Observation dimension must be 20 or 36; got {}'.format(nb_observations))

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
            path = "Observation_Plots/{}_{}.pdf".format(export_name, i)
            plt.savefig(path)
    plt.show()


# -----------------------------------------------------------------------------------------------
DIM_STATES = 18
DIM_OBSERVATIONS = 20

path_truth = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Normal/truth_normal.dat'
path_obs = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Normal/noised_observations_normal.dat'
data_reader = DataReader()
max_timesteps = 1000
data_reader.read_states_as_arr(path_truth, max_timesteps=max_timesteps)
data_reader.read_observations_as_arr(path_obs, max_timesteps=max_timesteps)
data_reader.prepare_lists()
x = data_reader.true_states
y = data_reader.observations

my_model = TwoLegModel()
plot_observations(my_model, x, y)
