import itertools

import numpy as np
from particles import state_space_models as ssm
from particles import distributions as dists
from MyDists import MyMvNormal, MvStudent
from scipy.linalg import block_diag

from MechanicalModel import state_to_obs, compute_jacobian_obs, state_to_obs_linear

CONST_GRAVITATION = 9.81


class TwoLegModel(ssm.StateSpaceModel):
    """
    Two leg model...
    """

    def __init__(self,
                 dt=0.01,
                 dim_states=18,
                 dim_observations=36,
                 leg_constants=np.array([0.5, 0.6, 0.5, 0.6]),
                 imu_position=np.array([0.34, 0.29, 0.315, 0.33]),
                 a=np.array([0.01, 1.06, -0.13, -0.25, 0.37, -0.19,
                             0.57, 0.10, 2.54, -3.8, -0.08, -0.82,
                             -0.00, 0.01, -1.78, 3.32, -0.30, 0.54]),
                 P=0.01 * np.eye(18),
                 cov_step=0.01,
                 scale_x=1.0,
                 scale_y=1.0,
                 scale_phi=1.0,
                 factor_Q=1.0,
                 diag_Q=False,
                 sigma_imu_acc=0.1,
                 sigma_imu_gyro=0.01,
                 sigma_press_velo=0.1,
                 sigma_press_acc=1000.0,
                 factor_H=1.0
                 ):
        self.dt = dt
        self.dim_states = dim_states
        self.dim_observations = dim_observations
        self.A = np.zeros((dim_states, dim_states))
        self.set_process_transition_matrix()
        self.g = CONST_GRAVITATION
        self.cst = imu_position
        self.legs = leg_constants
        self.a = a
        self.P = P
        self.cov_step = cov_step
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_phi = scale_phi
        self.factor_Q = factor_Q
        self.diag_Q = diag_Q
        self.sigma_imu_acc = sigma_imu_acc
        self.sigma_imu_gyro = sigma_imu_gyro
        self.sigma_press_velo = sigma_press_velo
        self.sigma_press_acc = sigma_press_acc
        self.factor_H = factor_H
        self.set_process_covariance()
        self.set_observation_covariance()

    def set_process_transition_matrix(self):
        self.A = np.eye(self.dim_states)
        for row in range(0, self.dim_states):
            for col in range(0, self.dim_states):
                if row + 6 == col:
                    self.A[row, col] = self.dt
                if row + 12 == col:
                    self.A[row, col] = self.dt ** 2 / 2.0
        return None

    def set_process_covariance(self):
        self.Q = np.zeros((self.dim_states, self.dim_states))
        block_size = self.dim_states // 3
        for row in range(0, self.dim_states):
            for col in range(0, self.dim_states):
                if row < block_size:
                    if row == col:
                        self.Q[row, col] = self.cov_step ** 5 / 20.0
                    elif row + 6 == col and not self.diag_Q:
                        self.Q[row, col] = self.cov_step ** 4 / 8.0
                        self.Q[col, row] = self.cov_step ** 4 / 8.0
                    elif row + 12 == col and not self.diag_Q:
                        self.Q[row, col] = self.cov_step ** 3 / 6.0
                        self.Q[col, row] = self.cov_step ** 3 / 6.0
                elif block_size <= row < 2 * block_size:
                    if row == col:
                        self.Q[row, col] = self.cov_step ** 3 / 3.0
                    elif row + 6 == col and not self.diag_Q:
                        self.Q[row, col] = self.cov_step ** 2 / 2.0
                        self.Q[col, row] = self.cov_step ** 2 / 2.0
                elif 2 * block_size <= row:
                    if row == col:
                        self.Q[row, col] = self.cov_step
        idx_groups = [[0, 6, 12], [1, 7, 13], [2, 8, 14], [3, 9, 15], [4, 10, 16], [5, 11, 17]]
        scale_factors = [self.scale_x, self.scale_y, self.scale_phi, self.scale_phi, self.scale_phi, self.scale_phi]
        for factor, idxs in zip(scale_factors, idx_groups):
            for row, col in itertools.product(idxs, idxs):
                self.Q[row, col] *= factor
        self.Q *= self.factor_Q
        print('Determinant process cov det(Q)={}'.format(np.linalg.det(self.Q)))
        return None

    def set_observation_covariance(self):
        if self.dim_observations == 20:
            self.H = np.diag([self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_gyro,
                              self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_gyro,
                              self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_gyro,
                              self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_gyro,
                              self.sigma_press_velo, self.sigma_press_velo, self.sigma_press_acc, self.sigma_press_acc,
                              self.sigma_press_velo, self.sigma_press_velo, self.sigma_press_acc, self.sigma_press_acc])
        elif self.dim_observations == 36:
            self.H = np.diag([self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_gyro,
                              self.sigma_imu_gyro, self.sigma_imu_gyro,
                              self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_gyro,
                              self.sigma_imu_gyro, self.sigma_imu_gyro,
                              self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_gyro,
                              self.sigma_imu_gyro, self.sigma_imu_gyro,
                              self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_gyro,
                              self.sigma_imu_gyro, self.sigma_imu_gyro,
                              self.sigma_press_velo, self.sigma_press_velo, self.sigma_press_velo, self.sigma_press_acc,
                              self.sigma_press_acc, self.sigma_press_acc,
                              self.sigma_press_velo, self.sigma_press_velo, self.sigma_press_velo, self.sigma_press_acc,
                              self.sigma_press_acc, self.sigma_press_acc])
        else:
            raise AssertionError(
                'Observation dimension must be 20 or 36; got {} instead.'.format(self.dim_observations))
        self.H *= self.factor_H
        print('Determinant observation cov det(H)={}'.format(np.linalg.det(self.H)))
        return None

    def state_transition(self, xp):
        return np.matmul(self.A, xp.T).T

    def state_to_observation(self, x):
        return state_to_obs(x, self.dim_observations, self.g, self.legs, self.cst)

    def state_to_observation_linear(self, x, xp):
        return state_to_obs_linear(x, xp, self.dim_states, self.dim_observations, self.g, self.legs, self.cst)

    def PX0(self):
        return dists.MvNormal(loc=self.a, cov=self.P)

    def PX(self, t, xp):
        return dists.MvNormal(loc=self.state_transition(xp), cov=self.Q)

    def PY(self, t, xp, x):
        ############### remove after debug ################
        """
        y_nonlin = self.state_to_observation(x)
        y_lin = self.state_to_observation_linear(x, xp)
        y_res = np.abs(y_nonlin - y_lin)
        y_res_mean = np.mean(y_res, axis=0)
        """
        ###################################################
        return dists.MvNormal(loc=self.state_to_observation(x), cov=self.H)
        # return dists.MvNormal(loc=self.state_to_observation_linear(x, xp), cov=self.H)


class TwoLegModelGuided(TwoLegModel):
    def __init__(self,
                 dt=0.01,
                 dim_states=18,
                 dim_observations=36,
                 leg_constants=np.array([0.5, 0.6, 0.5, 0.6]),
                 imu_position=np.array([0.34, 0.29, 0.315, 0.33]),
                 a=np.array([0.01, 1.06, -0.13, -0.25, 0.37, -0.19,
                             0.57, 0.10, 2.54, -3.8, -0.08, -0.82,
                             -0.00, 0.01, -1.78, 3.32, -0.30, 0.54]),
                 P=0.01 * np.eye(18),
                 cov_step=0.01,
                 scale_x=1.0,
                 scale_y=1.0,
                 scale_phi=1.0,
                 factor_Q=1.0,
                 diag_Q=False,
                 sigma_imu_acc=0.1,
                 sigma_imu_gyro=0.01,
                 sigma_press_velo=0.1,
                 sigma_press_acc=1000.0,
                 factor_H=1.0,
                 factor_kalman=1.0):
        super().__init__(dt=dt, dim_states=dim_states, dim_observations=dim_observations, leg_constants=leg_constants,
                         imu_position=imu_position, a=a, P=P, cov_step=cov_step, scale_x=scale_x, scale_y=scale_y,
                         scale_phi=scale_phi, factor_Q=factor_Q, diag_Q=diag_Q, sigma_imu_acc=sigma_imu_acc,
                         sigma_imu_gyro=sigma_imu_gyro, sigma_press_velo=sigma_press_velo,
                         sigma_press_acc=sigma_press_acc, factor_H=factor_H)
        self.H_inv = np.linalg.inv(self.H)
        self.Q_inv = np.linalg.inv(self.Q)
        self.kalman_covs = np.empty((1, self.dim_states, self.dim_states))
        self.factor_kalman = factor_kalman

    def compute_observation_derivatives(self, x):
        return compute_jacobian_obs(x, self.dim_states, self.dim_observations, self.g, self.legs, self.cst)

    def init_kalman_covs(self, nb_particles):
        self.kalman_covs = np.array([self.P for _ in range(0, nb_particles)])
        return None

    def compute_ekf_proposal(self, xp, data_t, sigma):
        x_hat = np.reshape(self.state_transition(xp), (1, self.dim_states))
        sigma = np.matmul(self.A, np.matmul(sigma, self.A.T)) + self.Q
        df = self.compute_observation_derivatives(x_hat)

        innovation_inv = np.linalg.inv(np.matmul(df, np.matmul(sigma, df.T)) + self.H)
        kalman_gain = np.matmul(sigma, np.matmul(df.T, innovation_inv))

        mu = x_hat + np.matmul(kalman_gain, (data_t - self.state_to_observation(x_hat)).T).T
        sigma = np.matmul(np.eye(self.dim_states) - np.matmul(kalman_gain, df), sigma)
        return mu, sigma

    def compute_tom_proposal(self, xp, data_t):
        x_hat = np.reshape(self.state_transition(xp), (1, self.dim_states))
        df = self.compute_observation_derivatives(x_hat)

        sigma = np.linalg.inv(np.matmul(df.T, np.matmul(self.H_inv, df)) + self.Q_inv)
        x_hat = np.matmul(np.matmul(df.T, self.H_inv), (data_t - self.state_to_observation(x_hat)).T).T + np.matmul(
            self.Q_inv, x_hat.T).T
        mu = np.matmul(sigma, x_hat.T).T

        return mu, sigma

    def compute_cappe_proposal(self, xp, data_t):
        df = self.compute_observation_derivatives(xp)
        innovation_inv = np.linalg.inv(np.matmul(df, np.matmul(self.Q, df.T)) + self.H)
        kalman_gain = np.matmul(self.Q, np.matmul(df.T, innovation_inv))

        x_hat = np.reshape(self.state_transition(xp), (1, self.dim_states))
        mean = x_hat + np.matmul(kalman_gain, (data_t - self.state_to_observation(x_hat)).T).T
        cov = np.matmul(np.eye(self.dim_states) - np.matmul(kalman_gain, df), self.Q)
        return mean, cov

    def proposal0(self, data):
        return self.PX0()

    def proposal(self, t, xp, data):
        ############### remove after debug ################
        """
        xpp = self.state_transition(xp)
        y_nonlin = self.state_to_observation(xpp)
        y_lin = self.state_to_observation_linear(xpp, xp)
        y_res = np.mean(np.abs(y_nonlin - y_lin), axis=0)
        y_res_nonlin = np.mean(np.abs(data[t] - y_nonlin), axis=0)
        err_rel_nonlin = y_res_nonlin/np.abs(data[t])
        y_res_lin = np.mean(np.abs(data[t] - y_lin), axis=0)
        err_rel_lin = y_res_lin/np.abs(data[t])
        """
        ###################################################
        nb_particles, dim_state = xp.shape
        if t == 1:
            self.init_kalman_covs(nb_particles)
        x_hats = np.empty((nb_particles, dim_state))
        kalman_covs = np.empty((nb_particles, dim_state, dim_state))
        for i in range(0, nb_particles):
            # sigma = self.kalman_covs[i]
            sigma = np.zeros((dim_state, dim_state))
            # sigma = 1.0 * np.eye(dim_state)
            x_hat, sigma = self.compute_ekf_proposal(xp[i], data[t], sigma)  # ekf version
            # x_hat, sigma = self.compute_cappe_proposal(xp[i], data[t])  # cappe version
            # x_hat, sigma = self.compute_tom_proposal(xp[i], data[t])
            x_hats[i, :] = x_hat
            kalman_covs[i, :, :] = sigma
        self.kalman_covs = kalman_covs
        mean = x_hats
        covar = self.factor_kalman * np.mean(kalman_covs, axis=0)  # covar = self.factor_kalman * kalman_covs
        # return MyMvNormal(loc=mean, cov=kalman_covs)
        return dists.MvNormal(loc=mean, cov=covar)
        # return MvStudent(loc=mean, shape=covar)
