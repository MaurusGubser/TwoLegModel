import itertools
import numpy as np
import scipy
from particles import state_space_models as ssm
from particles import distributions as dists
from CustomDistributions import MvNormalMultiDimCov, MvStudent, MvNormalMissingObservations
from scipy.linalg import block_diag

from MechanicalModel import state_to_obs, compute_jacobian_obs, state_to_obs_linear, create_rotation_matrix_z

CONST_GRAVITATION = 9.81


class TwoLegModel(ssm.StateSpaceModel):
    """
    Two leg model...
    """

    def __init__(self,
                 dt=0.01,
                 dim_states=18,
                 dim_observations=20,
                 femur_left=0.5,
                 fibula_left=0.6,
                 femur_right=0.5,
                 fibula_right=0.6,
                 pos_imu0=0.34,
                 pos_imu1=0.29,
                 pos_imu2=0.315,
                 pos_imu3=0.33,
                 b0=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                 alpha_0=0.0,
                 alpha_1=0.0,
                 alpha_2=0.0,
                 alpha_3=0.0,
                 factor_init=0.01,  # 0.01
                 cov_step=0.01,
                 lambda_x=10000.0,  # 10000.0
                 lambda_y=1000.0,  # 1000.0
                 lambda_phi=10000000.0,  # 10000000.0
                 factor_Q=1.0,  # 1.0
                 diag_Q=False,
                 sigma_imu_acc=0.1,  # 0.1
                 sigma_imu_gyro=0.1,  # 0.1
                 sigma_press_velo=0.1,  # 0.1
                 sigma_press_acc=1.0,  # 1.0
                 factor_S=1.0,  # 1.0
                 factor_proposal=1.2):  # 1.2
        ssm.StateSpaceModel().__init__()
        self.dt = dt
        self.dim_states = dim_states
        self.dim_observations = dim_observations
        self.A = np.zeros((dim_states, dim_states))
        self.set_process_transition_matrix()
        self.g = CONST_GRAVITATION
        self.femur_left = femur_left
        self.fibula_left = fibula_left
        self.femur_right = femur_right
        self.fibula_right = fibula_right
        self.len_legs = np.array([self.femur_left, self.fibula_left, self.femur_right, self.fibula_right])
        self.pos_imu0 = pos_imu0
        self.pos_imu1 = pos_imu1
        self.pos_imu2 = pos_imu2
        self.pos_imu3 = pos_imu3
        self.pos_imus = np.array([self.pos_imu0, self.pos_imu1, self.pos_imu2, self.pos_imu3])
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.R = np.eye(self.dim_observations)
        self.set_imu_rotation_matrices()
        self.a = b0
        self.factor_init = factor_init
        self.P = np.eye(self.dim_states)
        self.set_init_covariance()
        self.cov_step = cov_step
        self.scale_x = lambda_x
        self.scale_y = lambda_y
        self.scale_phi = lambda_phi
        self.factor_Q = factor_Q
        self.diag_Q = diag_Q
        self.sigma_imu_acc = sigma_imu_acc
        self.sigma_imu_gyro = sigma_imu_gyro
        self.sigma_press_velo = sigma_press_velo
        self.sigma_press_acc = sigma_press_acc
        self.factor_H = factor_S
        self.Q = np.zeros((self.dim_states, self.dim_states))
        self.set_process_covariance()
        self.H = np.zeros((self.dim_observations, self.dim_observations))
        self.set_observation_covariance()
        self.factor_proposal = factor_proposal
        self.kalman_covs = np.empty((1, self.dim_states, self.dim_states))

    def set_init_covariance(self):
        self.P[0, 0] = 0.1
        self.P[1, 1] = 0.1
        self.P[2, 2] = 0.2
        self.P[3, 3] = 0.2
        self.P[4, 4] = 0.2
        self.P[5, 5] = 0.2
        self.P[6, 6] = 0.1
        self.P[7, 7] = 0.1
        self.P[8, 8] = 1.0
        self.P[9, 9] = 1.0
        self.P[10, 10] = 1.0
        self.P[11, 11] = 1.0
        self.P[12, 12] = 1.0
        self.P[13, 13] = 1.0
        self.P[14, 14] = 5.0
        self.P[15, 15] = 5.0
        self.P[16, 16] = 5.0
        self.P[17, 17] = 5.0
        self.P = self.factor_init * self.P
        return None

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
        self.Q = self.factor_Q * self.Q
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
        self.H = self.factor_H * self.H
        return None

    def set_imu_rotation_matrices(self):
        R0 = create_rotation_matrix_z(self.alpha_0)
        R1 = create_rotation_matrix_z(self.alpha_1)
        R2 = create_rotation_matrix_z(self.alpha_2)
        R3 = create_rotation_matrix_z(self.alpha_3)
        R = block_diag(R0, R0, R1, R1, R2, R2, R3, R3, np.eye(12))
        self.R = R

    def state_transition(self, xp):
        return np.matmul(self.A, xp.T).T

    def state_to_observation(self, x):
        return state_to_obs(x, self.dim_observations, self.g, self.len_legs, self.pos_imus, self.R)

    # is not needed -> remove?
    def state_to_observation_linear(self, x, xp):
        return state_to_obs_linear(x, xp, self.dim_states, self.dim_observations, self.g, self.len_legs, self.pos_imus)

    def PX0(self):
        return dists.MvNormal(loc=self.a, cov=self.P)

    def PX(self, t, xp):
        return dists.MvNormal(loc=self.state_transition(xp), cov=self.Q)

    def PY(self, t, xp, x):
        return MvNormalMissingObservations(loc=self.state_to_observation(x), cov=self.H)

    def compute_observation_derivatives(self, x):
        return compute_jacobian_obs(x, self.dim_states, self.dim_observations, self.g, self.len_legs, self.pos_imus,
                                    self.R)

    def compute_ekf_proposal(self, xp, data_t):
        nb_particles = xp.shape[0]
        mask_not_nan = np.invert(np.isnan(data_t))
        mask_2d = np.outer(mask_not_nan, mask_not_nan)
        nb_non_nan = np.sum(mask_not_nan)

        # covariance masked
        x_hat = self.state_transition(xp)
        dh = self.compute_observation_derivatives(x_hat)
        dh = dh[:, mask_not_nan.flatten(), :]
        H_masked = np.reshape(self.H[mask_2d], (nb_non_nan, nb_non_nan))

        dh_Q = np.matmul(dh, self.Q)
        S_inv = np.linalg.inv(np.matmul(dh_Q, np.transpose(dh, (0, 2, 1))) + H_masked)
        kalman_gain = np.matmul(np.transpose(dh_Q, (0, 2, 1)), S_inv)
        prediction_err = data_t[mask_not_nan] - self.state_to_observation(x_hat)[:, mask_not_nan.flatten()]

        mu = x_hat + np.einsum('ijk, ik -> ij', kalman_gain, prediction_err)
        sigma = self.Q - np.matmul(kalman_gain, dh_Q)

        return mu, sigma

    def proposal0(self, data):
        return self.PX0()

    def proposal(self, t, xp, data):
        mean, kalman_covs = self.compute_ekf_proposal(xp, data[t])
        covar = self.factor_proposal * np.mean(kalman_covs, axis=0)
        return dists.MvNormal(loc=mean, cov=covar)

    def upper_bound_log_pt(self, t):
        return 1.0 / np.sqrt((2 * np.pi)**self.dim_states * np.linalg.det(self.Q))
