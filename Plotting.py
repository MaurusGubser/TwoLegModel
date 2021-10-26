import numpy as np
import matplotlib.pyplot as plt
import os


def set_export_path(folder_name, export_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    path = folder_name + '/' + export_name
    return path


class Plotter:
    def __init__(self, truth, delta_t):
        nb_steps, _, dim_states = truth.shape
        self.nb_steps = nb_steps
        self.dim_states = dim_states
        self.delta_t = delta_t
        self.truth = truth

    def plot_samples_detail(self, samples, export_name):
        nb_steps, nb_samples, dim_states = samples.shape
        if nb_steps != self.nb_steps or dim_states != self.dim_states:
            raise AssertionError(
                'Truth and states are not compatible: shape truth is {}; shape samples is {}'.format(self.truth.shape,
                                                                                                     samples.shape))
        nb_graphs = min(nb_samples, 5)
        nb_axes = 3
        nb_figures = int(np.ceil(self.dim_states / nb_axes))
        t_vals = np.linspace(0.0, self.nb_steps * self.delta_t, self.nb_steps)
        if self.dim_states == 18:
            state_names = ['$x_H$', '$y_H$', r'$\varphi_0$', r'$\varphi_1$', r'$\varphi_2$', r'$\varphi_3$',
                           r'$\dot x_H$', r'$\dot y_H$', r'$\dot \varphi_0$', r'$\dot \varphi_1$', r'$\dot \varphi_2$',
                           r'$\dot \varphi_3$',
                           r'$\ddot x_H$', r'$\ddot y_H$', r'$\ddot \varphi_0$', r'$\ddot \varphi_1$',
                           r'$\ddot \varphi_2$',
                           r'$\ddot \varphi_3$']
        elif self.dim_states == 12:
            state_names = ['$x_H$', '$y_H$', r'$\varphi_0$', r'$\varphi_1$',
                           r'$\dot x_H$', r'$\dot y_H$', r'$\dot \varphi_0$', r'$\dot \varphi_1$',
                           r'$\ddot x_H$', r'$\ddot y_H$', r'$\ddot \varphi_0$', r'$\ddot \varphi_1$']
            self.truth = self.truth[:, (0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15)]
        else:
            raise AssertionError('Dimension of state vector expected to be 12 or 18; got {}'.format(self.dim_states))
        export_path = set_export_path('State_Plots', export_name)

        fig_list = []
        axs_list = []
        for i in range(0, nb_figures):
            fig, axs = plt.subplots(ncols=1, nrows=nb_axes, figsize=(12, 8))
            for j in range(0, nb_axes):
                if i * nb_axes + j > self.dim_states - 1:
                    break
                axs[j].grid(axis='both')
                for k in range(0, nb_graphs):
                    axs[j % nb_axes].plot(t_vals, samples[:, k, nb_axes * i + j], label='Sample {}'.format(k),
                                          linewidth=1)
                axs[j % nb_axes].plot(t_vals, self.truth[:, :, nb_axes * i + j], label='truth', linewidth=1.5,
                                      color='green')
                axs[j % nb_axes].set_title(state_names[nb_axes * i + j])
                axs[j % nb_axes].legend()
            fig.tight_layout()
            fig_list.append(fig)
            axs_list.append(axs)
            if export_name is not None:
                plt.savefig(export_path + '_' + str(i) + '.pdf')
        plt.show()
        return None

    def plot_particle_moments(self, particles_mean, particles_var, export_name):
        nb_axes = 3
        nb_figures = int(np.ceil(self.dim_states / nb_axes))
        t_vals = np.linspace(0.0, self.nb_steps * self.delta_t, self.nb_steps)
        particles_sd = np.sqrt(particles_var)
        if self.dim_states == 18:
            state_names = ['$x_H$', '$y_H$', r'$\varphi_0$', r'$\varphi_1$', r'$\varphi_2$', r'$\varphi_3$',
                           r'$\dot x_H$', r'$\dot y_H$', r'$\dot \varphi_0$', r'$\dot \varphi_1$', r'$\dot \varphi_2$',
                           r'$\dot \varphi_3$',
                           r'$\ddot x_H$', r'$\ddot y_H$', r'$\ddot \varphi_0$', r'$\ddot \varphi_1$',
                           r'$\ddot \varphi_2$',
                           r'$\ddot \varphi_3$']
        elif self.dim_states == 12:
            state_names = ['$x_H$', '$y_H$', r'$\varphi_0$', r'$\varphi_1$',
                           r'$\dot x_H$', r'$\dot y_H$', r'$\dot \varphi_0$', r'$\dot \varphi_1$',
                           r'$\ddot x_H$', r'$\ddot y_H$', r'$\ddot \varphi_0$', r'$\ddot \varphi_1$']
            self.truth = self.truth[:, (0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15)]
        else:
            raise AssertionError('Dimension of state vector expected to be 12 or 18; got {}'.format(self.dim_states))
        export_path = set_export_path('Particles_Plots', export_name)
        fig_list = []
        axs_list = []
        for i in range(0, nb_figures):
            fig, axs = plt.subplots(ncols=1, nrows=nb_axes, figsize=(12, 8))
            for j in range(0, nb_axes):
                if i * nb_axes + j > self.dim_states - 1:
                    break
                axs[j].grid(axis='both')
                axs[j].plot(t_vals, particles_mean[:, nb_axes * i + j], label='Particle mean', color='blue')
                axs[j].fill_between(t_vals, particles_mean[:, nb_axes * i + j] - particles_sd[:, nb_axes * i + j],
                                    particles_mean[:, nb_axes * i + j] + particles_sd[:, nb_axes * i + j], alpha=0.2,
                                    color='blue')
                axs[j % nb_axes].plot(t_vals, self.truth[:, :, nb_axes * i + j], label='Truth', color='green',
                                      linewidth=1.5)
                axs[j % nb_axes].set_title(state_names[nb_axes * i + j])
                axs[j % nb_axes].legend()
            fig.tight_layout()
            fig_list.append(fig)
            axs_list.append(axs)
            if export_name is not None:
                plt.savefig(export_path + '_' + str(i) + '.pdf')
        plt.show()

        return None

    def plot_particles_trajectories(self, hist_X, export_name):
        nb_steps, nb_particles, dim_states = hist_X.shape
        if nb_steps != self.nb_steps or dim_states != self.dim_states:
            raise AssertionError(
                'Truth and particles are not compatible: shape truth is {}; shape particles is {}'.format(
                    self.truth.shape, hist_X.shape))

        nb_graphs = min(nb_particles, 8)
        nb_axes = 3
        nb_figures = int(np.ceil(self.dim_states / nb_axes))
        t_vals = np.linspace(0.0, self.nb_steps * self.delta_t, self.nb_steps)
        if self.dim_states == 18:
            state_names = ['$x_H$', '$y_H$', r'$\varphi_0$', r'$\varphi_1$', r'$\varphi_2$', r'$\varphi_3$',
                           r'$\dot x_H$', r'$\dot y_H$', r'$\dot \varphi_0$', r'$\dot \varphi_1$', r'$\dot \varphi_2$',
                           r'$\dot \varphi_3$',
                           r'$\ddot x_H$', r'$\ddot y_H$', r'$\ddot \varphi_0$', r'$\ddot \varphi_1$',
                           r'$\ddot \varphi_2$',
                           r'$\ddot \varphi_3$']
        elif self.dim_states == 12:
            state_names = ['$x_H$', '$y_H$', r'$\varphi_0$', r'$\varphi_1$',
                           r'$\dot x_H$', r'$\dot y_H$', r'$\dot \varphi_0$', r'$\dot \varphi_1$',
                           r'$\ddot x_H$', r'$\ddot y_H$', r'$\ddot \varphi_0$', r'$\ddot \varphi_1$']
            self.truth = self.truth[:, (0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15)]
        else:
            raise AssertionError('Dimension of state vector expected to be 12 or 18; got {}'.format(self.dim_states))
        export_path = set_export_path('Particle_Plots', export_name)

        fig_list = []
        axs_list = []
        for i in range(0, nb_figures):
            fig, axs = plt.subplots(ncols=1, nrows=nb_axes, figsize=(12, 8))
            for j in range(0, nb_axes):
                if i * nb_axes + j > self.dim_states - 1:
                    break
                axs[j].grid(axis='both')
                for k in range(0, nb_graphs):
                    axs[j % nb_axes].plot(t_vals, hist_X[:, k, nb_axes * i + j], label='Sample {}'.format(k),
                                          linewidth=1)
                axs[j % nb_axes].plot(t_vals, self.truth[:, :, nb_axes * i + j], label='truth', linewidth=1.5,
                                      color='green')
                axs[j % nb_axes].set_title(state_names[nb_axes * i + j])
                axs[j % nb_axes].legend()
            fig.tight_layout()
            fig_list.append(fig)
            axs_list.append(axs)
            if export_name is not None:
                plt.savefig(export_path + '_' + str(i) + '.pdf')
        plt.show()
        return None
