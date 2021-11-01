import numpy as np
import matplotlib.pyplot as plt
import os


def set_export_path(folder_name, export_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    path = folder_name + '/' + export_name
    return path


def compute_hist_stats(X_hist):
    hist_mean = np.mean(X_hist, axis=1)
    hist_var = np.var(X_hist, axis=1)
    return hist_mean, hist_var


class Plotter:
    def __init__(self, true_states, true_obs, delta_t):
        nb_steps_states, _, dim_states = true_states.shape
        nb_steps_obs, _, dim_observations = true_obs.shape
        if nb_steps_states != nb_steps_obs:
            raise AssertionError(
                'States and observation have different number of steps: {} and {}, respectively'.format(nb_steps_states,
                                                                                                        nb_steps_obs))
        self.nb_steps = nb_steps_states
        self.dim_states = dim_states
        self.dim_observations = dim_observations
        self.true_states = true_states
        self.true_obs = true_obs
        self.delta_t = delta_t

    def plot_samples_detail(self, samples, export_name=None):
        nb_steps, nb_samples, dim_states = samples.shape
        if nb_steps != self.nb_steps or dim_states != self.dim_states:
            raise AssertionError(
                'Truth and states are not compatible: shape truth is {}; shape samples is {}'.format(
                    self.true_states.shape,
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
            self.true_states = self.true_states[:, (0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15)]
        else:
            raise AssertionError('Dimension of state vector expected to be 12 or 18; got {}'.format(self.dim_states))

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
                axs[j % nb_axes].plot(t_vals, self.true_states[:, :, nb_axes * i + j], label='truth', linewidth=1.5,
                                      color='green')
                axs[j % nb_axes].set_title(state_names[nb_axes * i + j])
                axs[j % nb_axes].legend()
            fig.suptitle('Sampled states')
            fig.tight_layout()
            fig_list.append(fig)
            axs_list.append(axs)
            if export_name is not None:
                export_path = set_export_path('State_Plots', export_name)
                plt.savefig(export_path + '_' + str(i) + '.pdf')
        plt.show()
        return None

    def plot_particle_moments(self, particles_mean, particles_var, X_hist=None, export_name=None):
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
            self.true_states = self.true_states[:, (0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15)]
        else:
            raise AssertionError('Dimension of state vector expected to be 12 or 18; got {}'.format(self.dim_states))
        if X_hist is not None:
            hist_mean, hist_var = compute_hist_stats(X_hist)
            hist_sd = np.sqrt(hist_var)
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
                axs[j].plot(t_vals, self.true_states[:, :, nb_axes * i + j], label='Truth', color='green',
                            linewidth=1.5)
                if X_hist is not None:
                    axs[j].plot(t_vals, hist_mean[:, nb_axes * i + j], label='History mean', color='orange')
                    axs[j].fill_between(t_vals, hist_mean[:, nb_axes * i + j] - hist_sd[:, nb_axes * i + j],
                                        hist_mean[:, nb_axes * i + j] + hist_sd[:, nb_axes * i + j], alpha=0.6,
                                        color='orange')
                axs[j].set_title(state_names[nb_axes * i + j])
                axs[j].legend()

            fig.suptitle('Particles mean and variance')
            fig.tight_layout()
            fig_list.append(fig)
            axs_list.append(axs)
            if export_name is not None:
                export_path = set_export_path('ParticlesStats_Plots', export_name)
                plt.savefig(export_path + '_' + str(i) + '.pdf')
        plt.show()

        return None

    def plot_particles_trajectories(self, X_hist, export_name=None):
        nb_steps, nb_particles, dim_states = X_hist.shape
        if nb_steps != self.nb_steps or dim_states != self.dim_states:
            raise AssertionError(
                'Truth and particles are not compatible: shape truth is {}; shape particles is {}'.format(
                    self.true_states.shape, X_hist.shape))

        nb_graphs = min(nb_particles, 5)
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
            self.true_states = self.true_states[:, (0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15)]
        else:
            raise AssertionError('Dimension of state vector expected to be 12 or 18; got {}'.format(self.dim_states))

        fig_list = []
        axs_list = []
        for i in range(0, nb_figures):
            fig, axs = plt.subplots(ncols=1, nrows=nb_axes, figsize=(12, 8))
            for j in range(0, nb_axes):
                if i * nb_axes + j > self.dim_states - 1:
                    break
                axs[j].grid(axis='both')
                for k in range(0, nb_graphs):
                    axs[j].plot(t_vals, X_hist[:, k, nb_axes * i + j], label='Sample {}'.format(k), linewidth=1)
                axs[j].plot(t_vals, self.true_states[:, :, nb_axes * i + j], label='truth', linewidth=1.5,
                            color='green')
                axs[j].set_title(state_names[nb_axes * i + j])
                axs[j].legend()
            fig.suptitle('{} particle trajectories'.format(nb_graphs))
            fig.tight_layout()
            fig_list.append(fig)
            axs_list.append(axs)
            if export_name is not None:
                export_path = set_export_path('Particles_Plots', export_name)
                plt.savefig(export_path + '_' + str(i) + '.pdf')
        plt.show()
        return None

    def plot_observations(self, samples, model, export_name=None):
        nb_steps, nb_samples, dim_states = samples.shape
        if nb_steps != self.nb_steps or dim_states != self.dim_states:
            raise AssertionError(
                'Truth and states are not compatible: shape truth is {}; shape samples is {}'.format(
                    self.true_states.shape,
                    samples.shape))
        true_obs = self.true_obs
        obs = np.empty((nb_steps, nb_samples, self.dim_observations))
        for i in range(0, nb_samples):
            obs[:, i, :] = model.state_to_observation(samples[:, i, :])

        if self.dim_observations==20:
            obs_names = ['$\ddot x^0$', '$\ddot y^0$', '$\omega_z^0$',
                         '$\ddot x^1$', '$\ddot y^1$', '$\omega_z^1$',
                         '$\ddot x^2$', '$\ddot y^2$', '$\omega_z^2$',
                         '$\ddot x^3$', '$\ddot y^3$', '$\omega_z^3$',
                         '$\dot x^4$', '$\dot y^4$', '$\ddot x^4$', '$\ddot y^4$',
                         '$\dot x^5$', '$\dot y^5$', '$\ddot x^5$', '$\ddot y^5$']
        elif self.dim_observations==36:
            obs_names = ['$\ddot x^0$', '$\ddot y^0$', '$\ddot z^0$', '$\omega_x^0$', '$\omega_y^0$', '$\omega_z^0$',
                         '$\ddot x^1$', '$\ddot y^1$', '$\ddot z^1$', '$\omega_x^1$', '$\omega_y^1$', '$\omega_z^1$',
                         '$\ddot x^2$', '$\ddot y^2$', '$\ddot z^2$', '$\omega_x^2$', '$\omega_y^2$', '$\omega_z^2$',
                         '$\ddot x^3$', '$\ddot y^3$', '$\ddot z^3$', '$\omega_x^3$', '$\omega_y^3$', '$\omega_z^3$',
                         '$\dot x^4$', '$\dot y^4$', '$\dot z^4$', '$\ddot x^4$', '$\ddot y^4$', '$\ddot z^4$',
                         '$\dot x^5$', '$\dot y^5$', '$\dot z^5$', '$\ddot x^5$', '$\ddot y^5$', '$\ddot z^5$']
        else:
            raise ValueError('Observation dimension has to be 20 or 36; got {} instead.'.format(self.dim_observations))
        _, _, nb_observations = obs.shape
        t_vals = np.linspace(0.0, nb_steps * self.delta_t, nb_steps)
        nb_graphs = min(nb_samples, 5)
        nb_axes = 3
        nb_figures = int(np.ceil(nb_observations / nb_axes))
        for i in range(0, nb_figures):
            fig, axs = plt.subplots(ncols=1, nrows=nb_axes, figsize=(12, 8))
            for j in range(0, nb_axes):
                if i * nb_axes + j > nb_observations - 1:
                    break
                axs[j].grid(axis='both')
                for k in range(0, nb_graphs):
                    axs[j].plot(t_vals, obs[:, k, i * nb_axes + j], label='Sample {}'.format(k), linewidth=1)
                axs[j].plot(t_vals, true_obs[:, :, i * nb_axes + j], label='True observation', linewidth=1.5)
                axs[j].legend()
                axs[j].set_title(obs_names[i * nb_axes + j])
            fig.tight_layout()
            if export_name is not None:
                export_path = set_export_path('Observation_Plots', export_name)
                plt.savefig(export_path + '_' + str(i) + '.pdf')
        plt.show()
        return None
