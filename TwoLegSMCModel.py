import itertools

import numpy as np
from particles import state_space_models as ssm
from particles import distributions as dists
from MyDists import MvNormalMultiDimCov, MvStudent, MvNormalMissingObservations
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
                 dim_observations=20,
                 len_legs=np.array([0.5, 0.6, 0.5, 0.6]),
                 pos_imus=np.array([0.34, 0.29, 0.315, 0.33]),
                 a=np.array([0.01, 1.06, -0.13, -0.25, 0.37, -0.19,
                             0.57, 0.10, 2.54, -3.8, -0.08, -0.82,
                             -0.00, 0.01, -1.78, 3.32, -0.30, 0.54]),
                 P=0.01 * np.eye(18),
                 cov_step=0.01,
                 scale_x=100.0,
                 scale_y=100.0,
                 scale_phi=250.0,
                 factor_Q=1000.0,
                 diag_Q=False,
                 sigma_imu_acc=0.1,
                 sigma_imu_gyro=0.01,
                 sigma_press_velo=0.1,
                 sigma_press_acc=1000.0,
                 factor_H=0.01,
                 factor_proposal=1.1):
        super().__init__()
        self.dt = dt
        self.dim_states = dim_states
        self.dim_observations = dim_observations
        self.A = np.zeros((dim_states, dim_states))
        self.set_process_transition_matrix()
        self.g = CONST_GRAVITATION
        self.pos_imus = pos_imus
        self.len_legs = len_legs
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
        self.Q = np.zeros((self.dim_states, self.dim_states))
        self.set_process_covariance()
        self.H = np.zeros((self.dim_observations, self.dim_observations))
        self.set_observation_covariance()
        self.factor_proposal = factor_proposal
        self.kalman_covs = np.empty((1, self.dim_states, self.dim_states))

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
        return None

    def state_transition(self, xp):
        return np.matmul(self.A, xp.T).T

    def state_to_observation(self, x):
        return state_to_obs(x, self.dim_observations, self.g, self.len_legs, self.pos_imus)

    def state_to_observation_linear(self, x, xp):
        return state_to_obs_linear(x, xp, self.dim_states, self.dim_observations, self.g, self.len_legs, self.pos_imus)

    def PX0(self):
        return dists.MvNormal(loc=self.a, cov=self.P)

    def PX(self, t, xp):
        return dists.MvNormal(loc=self.state_transition(xp), cov=self.Q)

    def PY(self, t, xp, x):
        return MvNormalMissingObservations(loc=self.state_to_observation(x), cov=self.H)
        # return dists.MvNormal(loc=self.state_to_observation(x), cov=self.H)

    def compute_observation_derivatives(self, x):
        return compute_jacobian_obs(x, self.dim_states, self.dim_observations, self.g, self.len_legs, self.pos_imus)

    def compute_ekf_proposal(self, xp, data_t):
        x_hat = self.state_transition(xp)
        df = self.compute_observation_derivatives(x_hat)

        innovation_inv = np.linalg.inv(np.matmul(df, np.matmul(self.Q, np.transpose(df, (0, 2, 1)))) + self.H)
        kalman_gain = np.matmul(self.Q, np.matmul(np.transpose(df, (0, 2, 1)), innovation_inv))
        prediction_err = np.nan_to_num(data_t - self.state_to_observation(x_hat), nan=0.0)

        mu = x_hat + np.einsum('ijk, ik -> ij', kalman_gain, prediction_err)
        sigma = np.matmul(np.eye(self.dim_states) - np.matmul(kalman_gain, df), self.Q)
        return mu, sigma

    def proposal0(self, data):
        return self.PX0()

    def proposal(self, t, xp, data):
        mean, kalman_covs = self.compute_ekf_proposal(xp, data[t])
        covar = self.factor_proposal * np.mean(kalman_covs, axis=0)
        return dists.MvNormal(loc=mean, cov=covar)
