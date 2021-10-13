import numpy as np
import matplotlib.pyplot as plt
import os


class Plotter:
    def __init__(self, samples, truth, export_name, delta_t):
        nb_steps_t, _, dim_states_t = truth.shape
        nb_steps_s, self.nb_samples, dim_states_s = samples.shape
        if nb_steps_s != nb_steps_t:
            raise AssertionError('Number of time steps of truth and sample does not coincide.')
        elif dim_states_s != dim_states_t:
            raise AssertionError('Dimension of true states and sample states does not coincide.')
        else:
            self.nb_steps = nb_steps_s
            self.dim_states = dim_states_s
        self.delta_t = delta_t
        self.samples = samples
        self.truth = truth
        self.export_name = export_name
        self.export_path = self.set_export_path()

    def set_export_path(self):
        if not os.path.exists('State_Plots'):
            os.mkdir('State_Plots')
        path = 'State_Plots/' + self.export_name
        return path

    def plot_samples_detail(self):
        nb_graphs = min(self.nb_samples, 5)
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
        fig_list = []
        axs_list = []
        for i in range(0, nb_figures):
            fig, axs = plt.subplots(ncols=1, nrows=nb_axes, figsize=(12, 8))
            for j in range(0, nb_axes):
                if i * nb_axes + j > self.dim_states - 1:
                    break
                axs[j].grid(axis='both')
                for k in range(0, nb_graphs):
                    axs[j % nb_axes].plot(t_vals, self.samples[:, k, nb_axes * i + j], label='Sample {}'.format(k),
                                          linewidth=1)
                axs[j % nb_axes].plot(t_vals, self.truth[:, :, nb_axes * i + j], label='truth', linewidth=1.5)
                axs[j % nb_axes].set_title(state_names[nb_axes * i + j])
                axs[j % nb_axes].legend()
            fig.tight_layout()
            fig_list.append(fig)
            axs_list.append(axs)
            if self.export_name is not None:
                plt.savefig(self.export_path + '_' + str(i) + '.pdf')
        plt.show()
        return None
