import itertools
import numpy as np
from particles import state_space_models as ssm
from particles import distributions as dists
from CustomDistributions import MvNormalMissingObservations
from scipy.linalg import block_diag

from MechanicalModel import state_to_obs, compute_jacobian_obs, create_rotation_matrix_z

CONST_GRAVITATION = 9.81


class TwoLegModel(ssm.StateSpaceModel):

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
                 alpha0=0.0,
                 alpha1=0.0,
                 alpha2=0.0,
                 alpha3=0.0,
                 b0=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                 factor_Q0=0.1,  # 0.1
                 lambda_x=10000.0,  # 10000.0
                 lambda_y=1000.0,  # 1000.0
                 lambda_phi=10000000.0,  # 10000000.0
                 sigma_imu_acc=0.1,  # 0.1
                 sigma_imu_gyro=0.1,  # 0.1
                 sigma_press_velo=0.1,  # 0.1
                 sigma_press_acc=1.0,  # 1.0
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
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.R = np.eye(self.dim_observations)
        self.set_imu_rotation_matrices()
        self.b0 = b0
        self.factor_Q0 = factor_Q0
        self.Q0 = self.factor_Q0 * np.eye(self.dim_states)
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.lambda_phi = lambda_phi
        self.sigma_imu_acc = sigma_imu_acc
        self.sigma_imu_gyro = sigma_imu_gyro
        self.sigma_press_velo = sigma_press_velo
        self.sigma_press_acc = sigma_press_acc
        self.Q = np.zeros((self.dim_states, self.dim_states))
        self.set_process_covariance()
        self.V = np.zeros((self.dim_observations, self.dim_observations))
        self.set_observation_covariance()
        self.factor_proposal = factor_proposal

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
                        self.Q[row, col] = self.dt ** 5 / 20.0
                    elif row + 6 == col:
                        self.Q[row, col] = self.dt ** 4 / 8.0
                        self.Q[col, row] = self.dt ** 4 / 8.0
                    elif row + 12 == col:
                        self.Q[row, col] = self.dt ** 3 / 6.0
                        self.Q[col, row] = self.dt ** 3 / 6.0
                elif block_size <= row < 2 * block_size:
                    if row == col:
                        self.Q[row, col] = self.dt ** 3 / 3.0
                    elif row + 6 == col:
                        self.Q[row, col] = self.dt ** 2 / 2.0
                        self.Q[col, row] = self.dt ** 2 / 2.0
                elif 2 * block_size <= row:
                    if row == col:
                        self.Q[row, col] = self.dt
        idx_groups = [[0, 6, 12], [1, 7, 13], [2, 8, 14], [3, 9, 15], [4, 10, 16], [5, 11, 17]]
        scale_factors = [self.lambda_x, self.lambda_y, self.lambda_phi, self.lambda_phi, self.lambda_phi,
                         self.lambda_phi]
        for factor, idxs in zip(scale_factors, idx_groups):
            for row, col in itertools.product(idxs, idxs):
                self.Q[row, col] *= factor
        return None

    def set_observation_covariance(self):
        if self.dim_observations == 20:
            self.V = np.diag([self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_gyro,
                              self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_gyro,
                              self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_gyro,
                              self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_gyro,
                              self.sigma_press_velo, self.sigma_press_velo, self.sigma_press_acc, self.sigma_press_acc,
                              self.sigma_press_velo, self.sigma_press_velo, self.sigma_press_acc, self.sigma_press_acc])
        elif self.dim_observations == 36:
            self.V = np.diag([self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_acc, self.sigma_imu_gyro,
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
        return None

    def set_imu_rotation_matrices(self):
        R0 = create_rotation_matrix_z(self.alpha0)
        R1 = create_rotation_matrix_z(self.alpha1)
        R2 = create_rotation_matrix_z(self.alpha2)
        R3 = create_rotation_matrix_z(self.alpha3)
        R = block_diag(R0, R0, R1, R1, R2, R2, R3, R3, np.eye(12))
        self.R = R

    def state_transition(self, xp):
        return np.matmul(self.A, xp.T).T

    def state_to_observation(self, x):
        return state_to_obs(x, self.dim_observations, self.g, self.len_legs, self.pos_imus, self.R)

    def PX0(self):
        return dists.MvNormal(loc=self.b0, cov=self.Q0)

    def PX(self, t, xp):
        return dists.MvNormal(loc=self.state_transition(xp), cov=self.Q)

    def PY(self, t, xp, x):
        return MvNormalMissingObservations(loc=self.state_to_observation(x), cov=self.V)

    def compute_observation_derivatives(self, x):
        return compute_jacobian_obs(x, self.dim_states, self.dim_observations, self.g, self.len_legs, self.pos_imus,
                                    self.R)

    def compute_ekf_proposal(self, xp, data_t):
        mask_not_nan = np.invert(np.isnan(data_t))
        mask_2d = np.outer(mask_not_nan, mask_not_nan)
        nb_non_nan = np.sum(mask_not_nan)

        # compute masked covariance and Jacobian
        x_hat = self.state_transition(xp)
        dh = self.compute_observation_derivatives(x_hat)
        dh = dh[:, mask_not_nan.flatten(), :]
        V_masked = np.reshape(self.V[mask_2d], (nb_non_nan, nb_non_nan))

        # EKF
        dh_Q = np.matmul(dh, self.Q)
        S_inv = np.linalg.inv(np.matmul(dh_Q, np.transpose(dh, (0, 2, 1))) + V_masked)
        kalman_gain = np.matmul(np.transpose(dh_Q, (0, 2, 1)), S_inv)
        prediction_err = data_t[mask_not_nan] - self.state_to_observation(x_hat)[:, mask_not_nan.flatten()]

        mu = x_hat + np.einsum('ijk, ik -> ij', kalman_gain, prediction_err)
        Sigma = self.Q - np.matmul(kalman_gain, dh_Q)

        return mu, Sigma

    def proposal0(self, data):
        return self.PX0()

    def proposal(self, t, xp, data):
        mu, Sigma = self.compute_ekf_proposal(xp, data[t])
        Sigma = self.factor_proposal * np.mean(Sigma, axis=0)
        return dists.MvNormal(loc=mu, cov=Sigma)

    def upper_bound_log_pt(self, t):
        return 1.0 / np.sqrt((2 * np.pi) ** self.dim_states * np.linalg.det(self.Q))
