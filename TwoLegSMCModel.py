import itertools

import numpy as np
from particles import state_space_models as ssm
from particles import distributions as dists

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
                 sigma_imu_acc=0.1,
                 sigma_imu_gyro=0.01,
                 sigma_press_velo=0.1,
                 sigma_press_acc=10.0,
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
        self.sigma_imu_acc = sigma_imu_acc
        self.sigma_imu_gyro = sigma_imu_gyro
        self.sigma_press_velo = sigma_press_velo
        self.sigma_press_acc = sigma_press_acc
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
                    elif row + 6 == col:
                        self.Q[row, col] = self.cov_step ** 4 / 8.0
                        self.Q[col, row] = self.cov_step ** 4 / 8.0
                    elif row + 12 == col:
                        self.Q[row, col] = self.cov_step ** 3 / 6.0
                        self.Q[col, row] = self.cov_step ** 3 / 6.0
                elif block_size <= row < 2 * block_size:
                    if row == col:
                        self.Q[row, col] = self.cov_step ** 3 / 3.0
                    elif row + 6 == col:
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
        return None

    def state_transition(self, xp):
        return np.matmul(self.A, xp.T).T

    def state_to_observation(self, x):
        nb_particles, _ = x.shape
        y = np.empty(shape=(nb_particles, self.dim_observations))
        for i in range(0, nb_particles):
            y[i] = self.state_to_observation_1dim(x[i])
        return y

    def state_to_observation_1dim(self, x):
        y = np.empty(shape=36)
        # left femur
        y[0] = self.cst[0] * x[14] + self.g * np.sin(x[2]) + np.sin(x[2]) * x[13] + np.cos(x[2]) * x[12]
        y[1] = self.cst[0] * x[8] ** 2 + self.g * np.cos(x[2]) - np.sin(x[2]) * x[12] + np.cos(x[2]) * x[13]
        y[2] = 0.0
        y[3] = 0.0
        y[4] = 0.0
        y[5] = x[8]

        # left fibula
        y[6] = self.cst[1] * x[14] + self.cst[1] * x[15] + self.g * np.sin(x[2] + x[3]) + self.legs[0] \
               * np.sin(x[3]) * x[8] ** 2 + self.legs[0] * np.cos(x[3]) * x[14] \
               + np.sin(x[2] + x[3]) * x[13] + np.cos(x[2] + x[3]) * x[12]
        y[7] = self.cst[1] * x[8] ** 2 + 2 * self.cst[1] * x[8] * x[9] + self.cst[1] * x[9] ** 2 \
               + self.g * np.cos(x[2] + x[3]) - self.legs[0] * np.sin(x[3]) * x[14] + self.legs[0] \
               * np.cos(x[3]) * x[8] ** 2 - np.sin(x[2] + x[3]) * x[12] + np.cos(x[2] + x[3]) * x[13]
        y[8] = 0.0
        y[9] = 0.0
        y[10] = 0.0
        y[11] = x[8] + x[9]

        # right femur
        y[12] = self.cst[2] * x[16] + self.g * np.sin(x[4]) + np.sin(x[4]) * x[13] + np.cos(x[4]) * x[12]
        y[13] = self.cst[2] * x[10] ** 2 + self.g * np.cos(x[4]) - np.sin(x[4]) * x[12] + np.cos(x[4]) * x[13]
        y[14] = 0.0
        y[15] = 0.0
        y[16] = 0.0
        y[17] = x[10]

        # right fibula
        y[18] = self.cst[3] * x[16] + self.cst[3] * x[17] + self.g * np.sin(x[4] + x[5]) + self.legs[2] \
                * np.sin(x[5]) * x[10] ** 2 + self.legs[2] * np.cos(x[5]) * x[16] \
                + np.sin(x[4] + x[5]) * x[13] + np.cos(x[4] + x[5]) * x[12]
        y[19] = self.cst[3] * x[10] ** 2 + 2 * self.cst[3] * x[10] * x[11] + self.cst[3] * x[11] ** 2 \
                + self.g * np.cos(x[4] + x[5]) - self.legs[2] * np.sin(x[5]) * x[16] + self.legs[2] \
                * np.cos(x[5]) * x[10] ** 2 - np.sin(x[4] + x[5]) * x[12] + np.cos(x[4] + x[5]) * x[13]
        y[20] = 0.0
        y[21] = 0.0
        y[22] = 0.0
        y[23] = x[10] + x[11]

        # left heel
        y[24] = self.legs[0] * np.cos(x[2]) * x[8] + self.legs[1] * (x[8] + x[9]) * np.cos(x[2] + x[3]) + x[6]
        y[25] = self.legs[0] * np.sin(x[2]) * x[8] + self.legs[1] * (x[8] + x[9]) * np.sin(x[2] + x[3]) + x[7]
        y[26] = 0.0
        y[27] = -self.legs[0] * np.sin(x[2]) * x[8] ** 2 + self.legs[0] * np.cos(x[2]) * x[14] \
            - self.legs[1] * (x[8] + x[9]) ** 2 * np.sin(x[2] + x[3]) + self.legs[1] \
            * (x[14] + x[15]) * np.cos(x[2] + x[3]) + x[12]
        y[28] = self.legs[0] * np.sin(x[2]) * x[14] + self.legs[0] * np.cos(x[2]) * x[8] ** 2 \
            + self.legs[1] * (x[8] + x[9]) ** 2 * np.cos(x[2] + x[3]) + self.legs[1] \
            * (x[14] + x[15]) * np.sin(x[2] + x[3]) + x[13]
        y[29] = 0.0

        # right heel
        y[30] = self.legs[2] * np.cos(x[4]) * x[10] + self.legs[3] * (x[10] + x[11]) * np.cos(x[4] + x[5]) + x[6]
        y[31] = self.legs[2] * np.sin(x[4]) * x[10] + self.legs[3] * (x[10] + x[11]) * np.sin(x[4] + x[5]) + x[7]
        y[32] = 0.0
        y[33] = -self.legs[2] * np.sin(x[4]) * x[10] ** 2 + self.legs[2] * np.cos(x[4]) * x[16] \
            - self.legs[3] * (x[10] + x[11]) ** 2 * np.sin(x[4] + x[5]) + self.legs[3] \
            * (x[16] + x[17]) * np.cos(x[4] + x[5]) + x[12]
        y[34] = self.legs[2] * np.sin(x[4]) * x[16] + self.legs[2] * np.cos(x[4]) * x[10] ** 2 \
            + self.legs[3] * (x[10] + x[11]) ** 2 * np.cos(x[4] + x[5]) + self.legs[3] \
            * (x[16] + x[17]) * np.sin(x[4] + x[5]) + x[13]
        y[35] = 0.0

        if self.dim_observations == 20:
            y = y[(0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34), ]

        return y

    def PX0(self):
        return dists.MvNormal(loc=self.a, cov=self.P)

    def PX(self, t, xp):
        # return dists.MvNormal(loc=self.state_transition(xp), cov=np.eye(self.dim_states))
        return dists.MvNormal(loc=self.state_transition(xp), cov=1.0*self.Q)

    def PY(self, t, xp, x):
        nb_particles, _ = x.shape
        mu = np.zeros(shape=(nb_particles, self.dim_observations))
        # return dists.MvNormal(loc=mu, cov=self.H)
        return dists.MvNormal(loc=self.state_to_observation(x), cov=1.0*self.H)


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
                 sigma_imu_acc=0.1,
                 sigma_imu_gyro=0.01,
                 sigma_press_velo=0.1,
                 sigma_press_acc=10.0):
        super().__init__(dt, dim_states, dim_observations, leg_constants, imu_position, a, P, cov_step, scale_x,
                         scale_y, scale_phi, sigma_imu_acc, sigma_imu_gyro, sigma_press_velo, sigma_press_acc)
        self.kalman_covs = np.empty((1, self.dim_states, self.dim_states))

    def compute_observation_derivatives(self, x):
        df = np.zeros((36, self.dim_states))

        # left femur
        df[0, 2] = -x[12] * np.sin(x[2]) + (x[13] + self.g) * np.cos(x[2])
        df[0, 12] = np.cos(x[2])
        df[0, 13] = np.sin(x[2])
        df[0, 14] = self.cst[0]
        df[1, 2] = -x[12] * np.cos(x[2]) - (x[13] + self.g) * np.sin(x[2])
        df[1, 8] = 2 * x[8] * self.cst[0]
        df[1, 12] = -np.sin(x[2])
        df[1, 13] = np.cos(x[2])
        df[5, 8] = 1

        # left fibula
        df[6, 2] = -x[12] * np.sin(x[2] + x[3]) + x[13] * np.cos(x[2] + x[3]) + self.g * np.cos(x[2] + x[3])
        df[6, 3] = -x[14] * self.legs[0] * np.sin(x[3]) - x[12] * np.sin(x[2] + x[3]) + x[13] * np.cos(
            x[2] + x[3]) + self.legs[0] * x[8] ** 2 * np.cos(x[3]) + self.g * np.cos(x[2] + x[3])
        df[6, 8] = 2 * self.legs[0] * x[8] * np.sin(x[3])
        df[6, 12] = np.cos(x[2] + x[3])
        df[6, 13] = np.sin(x[2] + x[3])
        df[6, 14] = self.cst[1] + self.legs[0] * np.cos(x[3])
        df[6, 15] = self.cst[1]
        df[7, 2] = -x[12] * np.cos(x[2] + x[3]) - x[13] * np.sin(x[2] + x[3]) - self.g * np.sin(x[2] + x[3])
        df[7, 3] = -x[14] * self.legs[0] * np.cos(x[3]) - x[12] * np.cos(x[2] + x[3]) - x[13] * np.sin(
            x[2] + x[3]) - self.legs[0] * x[8] ** 2 * np.sin(x[3]) - self.g * np.sin(x[2] + x[3])
        df[7, 8] = 2 * self.legs[0] * x[8] * np.cos(x[3]) + 2 * self.cst[1] * (x[8] + x[9])
        df[7, 9] = 2 * self.cst[1] * (x[8] + x[9])
        df[7, 12] = -np.sin(x[2] + x[3])
        df[7, 13] = np.cos(x[2] + x[3])
        df[7, 14] = -self.legs[0] * np.sin(x[3])
        df[11, 8] = 1
        df[11, 9] = 1

        # right femur
        df[12, 4] = -x[12] * np.sin(x[4]) + (x[13] + self.g) * np.cos(x[4])
        df[12, 12] = np.cos(x[4])
        df[12, 13] = np.sin(x[4])
        df[12, 16] = self.cst[2]
        df[13, 4] = -x[12] * np.cos(x[4]) - (x[13] + self.g) * np.sin(x[4])
        df[13, 10] = 2 * x[10] * self.cst[2]
        df[13, 12] = -np.sin(x[4])
        df[13, 13] = np.cos(x[4])
        df[17, 10] = 1

        # right fibula
        df[18, 4] = -x[12] * np.sin(x[4] + x[5]) + x[13] * np.cos(x[4] + x[5]) + self.g * np.cos(x[4] + x[5])
        df[18, 5] = -x[16] * self.legs[2] * np.sin(x[5]) - x[12] * np.sin(x[4] + x[5]) + x[13] * np.cos(
            x[4] + x[5]) + self.legs[2] * x[10] ** 2 * np.cos(x[5]) + self.g * np.cos(x[4] + x[5])
        df[18, 10] = 2 * self.legs[2] * x[10] * np.sin(x[5])
        df[18, 12] = np.cos(x[4] + x[5])
        df[18, 13] = np.sin(x[4] + x[5])
        df[18, 16] = self.cst[3] + self.legs[2] * np.cos(x[5])
        df[18, 17] = self.cst[3]
        df[19, 4] = -x[12] * np.cos(x[4] + x[5]) - x[13] * np.sin(x[4] + x[5]) - self.g * np.sin(x[4] + x[5])
        df[19, 5] = -x[16] * self.legs[2] * np.cos(x[5]) - x[12] * np.cos(x[4] + x[5]) - x[13] * np.sin(
            x[4] + x[5]) - self.legs[2] * x[10] ** 2 * np.sin(x[5]) - self.g * np.sin(x[4] + x[5])
        df[19, 10] = 2 * self.legs[2] * x[10] * np.cos(x[5]) + 2 * self.cst[3] * (x[10] + x[11])
        df[19, 11] = 2 * self.cst[3] * (x[10] + x[11])
        df[19, 12] = -np.sin(x[4] + x[5])
        df[19, 13] = np.cos(x[4] + x[5])
        df[19, 16] = -self.legs[2] * np.sin(x[5])
        df[23, 10] = 1
        df[23, 11] = 1

        # left heel
        df[24, 2] = -x[8] * self.legs[0] + np.sin(x[2]) - self.legs[1] * (x[8] + x[9]) * np.sin(x[2] + x[3])
        df[24, 3] = -self.legs[1] * (x[8] + x[9]) * np.sin(x[2] + x[3])
        df[24, 6] = 1
        df[24, 8] = self.legs[0] * np.cos(x[2]) + self.legs[1] * np.cos(x[2] + x[3])
        df[24, 9] = self.legs[1] * np.cos(x[2] + x[3])
        df[25, 2] = x[8] * self.legs[0] * np.cos(x[2]) + self.legs[1] * (x[8] + x[9]) * np.cos(x[2] + x[3])
        df[25, 3] = self.legs[1] * (x[8] + x[9]) * np.cos(x[2] + x[3])
        df[25, 7] = 1
        df[25, 8] = self.legs[0] * np.sin(x[2]) + self.legs[1] * np.sin(x[2] + x[3])
        df[25, 9] = self.legs[1] * np.sin(x[2] + x[3])
        df[27, 2] = -self.legs[0] * (x[14] * np.sin(x[2]) + x[8] ** 2 * np.cos(x[2])) - self.legs[1] * (
                x[14] + x[15]) * np.sin(x[2] + x[3]) - self.legs[1] * (x[8] + x[9]) ** 2 * np.cos(
            x[2] + x[3])
        df[27, 3] = -self.legs[1] * (x[14] + x[15]) * np.sin(x[2] + x[3]) - self.legs[1] * (
                x[8] + x[9]) ** 2 * np.cos(x[2] + x[3])
        df[27, 8] = -2 * self.legs[0] * x[8] * np.sin(x[2]) - 2 * self.legs[1] * (x[8] + x[9]) * np.sin(
            x[2] + x[3])
        df[27, 9] = -2 * self.legs[1] * (x[8] + x[9]) * np.sin(x[2] + x[3])
        df[27, 12] = 1
        df[27, 14] = self.legs[0] * np.cos(x[2]) + self.legs[1] * np.cos(x[2] + x[3])
        df[27, 15] = self.legs[1] * np.cos(x[2] + x[3])
        df[28, 2] = -self.legs[0] * (-x[14] * np.cos(x[2]) + x[8] ** 2 * np.sin(x[2])) + self.legs[1] * (
                x[14] + x[15]) * np.cos(x[2] + x[3]) - self.legs[1] * (x[8] + x[9]) ** 2 * np.sin(
            x[2] + x[3])
        df[28, 3] = self.legs[1] * (x[14] + x[15]) * np.cos(x[2] + x[3]) - self.legs[1] * (
                x[8] + x[9]) ** 2 * np.sin(x[2] + x[3])
        df[28, 8] = 2 * self.legs[0] * x[8] * np.cos(x[2]) + 2 * self.legs[1] * (x[8] + x[9]) * np.cos(
            x[2] + x[3])
        df[28, 9] = 2 * self.legs[1] * (x[8] + x[9]) * np.cos(x[2] + x[3])
        df[28, 13] = 1
        df[28, 14] = self.legs[0] * np.sin(x[2]) + self.legs[1] * np.sin(x[2] + x[3])
        df[28, 15] = self.legs[1] * np.sin(x[2] + x[3])

        # right heel
        df[30, 4] = -x[10] * self.legs[2] + np.sin(x[4]) - self.legs[3] * (x[10] + x[11]) * np.sin(x[4] + x[5])
        df[30, 5] = -self.legs[3] * (x[10] + x[11]) * np.sin(x[4] + x[5])
        df[30, 6] = 1
        df[30, 10] = self.legs[2] * np.cos(x[4]) + self.legs[3] * np.cos(x[4] + x[5])
        df[30, 11] = self.legs[3] * np.cos(x[4] + x[5])
        df[31, 4] = x[10] * self.legs[2] * np.cos(x[4]) + self.legs[3] * (x[10] + x[11]) * np.cos(x[4] + x[5])
        df[31, 5] = self.legs[3] * (x[10] + x[11]) * np.cos(x[4] + x[5])
        df[31, 7] = 1
        df[31, 10] = self.legs[2] * np.sin(x[4]) + self.legs[3] * np.sin(x[4] + x[5])
        df[31, 11] = self.legs[3] * np.sin(x[4] + x[5])
        df[33, 4] = -self.legs[2] * (x[16] * np.sin(x[4]) + x[10] ** 2 * np.cos(x[4])) - self.legs[3] * (
                x[16] + x[17]) * np.sin(x[4] + x[5]) - self.legs[3] * (x[10] + x[11]) ** 2 * np.cos(
            x[4] + x[5])
        df[33, 5] = -self.legs[3] * (x[16] + x[17]) * np.sin(x[4] + x[5]) - self.legs[3] * (
                x[10] + x[11]) ** 2 * np.cos(x[4] + x[5])
        df[33, 10] = -2 * self.legs[2] * x[10] * np.sin(x[4]) - 2 * self.legs[3] * (x[10] + x[11]) * np.sin(
            x[4] + x[5])
        df[33, 11] = -2 * self.legs[3] * (x[10] + x[11]) * np.sin(x[4] + x[5])
        df[33, 12] = 1
        df[33, 16] = self.legs[2] * np.cos(x[4]) + self.legs[3] * np.cos(x[4] + x[5])
        df[33, 17] = self.legs[3] * np.cos(x[4] + x[5])
        df[34, 4] = -self.legs[2] * (-x[16] * np.cos(x[4]) + x[10] ** 2 * np.sin(x[4])) + self.legs[3] * (
                x[16] + x[17]) * np.cos(x[4] + x[5]) - self.legs[3] * (x[10] + x[11]) ** 2 * np.sin(x[4] + x[5])
        df[34, 5] = self.legs[3] * (x[16] + x[17]) * np.cos(x[4] + x[5]) - self.legs[3] * (
                x[10] + x[11]) ** 2 * np.sin(x[4] + x[5])
        df[34, 10] = 2 * self.legs[2] * x[10] * np.cos(x[4]) + 2 * self.legs[3] * (x[10] + x[11]) * np.cos(
            x[4] + x[5])
        df[34, 11] = 2 * self.legs[3] * (x[10] + x[11]) * np.cos(x[4] + x[5])
        df[34, 13] = 1
        df[34, 16] = self.legs[2] * np.sin(x[4]) + self.legs[3] * np.sin(x[4] + x[5])
        df[34, 17] = self.legs[3] * np.sin(x[4] + x[5])

        if self.dim_observations == 20:
            df = df[(0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34), :]

        return df

    def init_kalman_covs(self, nb_particles):
        self.kalman_covs = np.array([np.eye(self.dim_states) for _ in range(0, nb_particles)])
        return None

    def compute_ekf_proposal(self, xp, data_t, sigma):
        x_hat = np.reshape(self.state_transition(xp), (1, self.dim_states))
        sigma = np.matmul(self.A, np.matmul(sigma, self.A.T)) + self.Q

        df = self.compute_observation_derivatives(x_hat.flatten())  # self.compute_observation_derivatives_1dim(xp)
        innovation = np.matmul(df, np.matmul(sigma, df.T)) + self.H
        kalman_gain = np.matmul(sigma, np.matmul(df.T, np.linalg.inv(innovation)))

        x_hat = x_hat + np.matmul(kalman_gain, (data_t - self.state_to_observation(x_hat)).T).T
        sigma = np.matmul(np.eye(self.dim_states) - np.matmul(kalman_gain, df), sigma)
        return x_hat, sigma

    def proposal0(self, data):
        return self.PX0()
        # return self.PX0().posterior(data[0], Sigma=np.eye(self.dim_states))

    def proposal(self, t, xp, data):
        nb_particles, dim_state = xp.shape
        if t == 1:
            self.init_kalman_covs(nb_particles)
        x_hats = np.empty((nb_particles, dim_state))
        kalman_covs = np.empty((nb_particles, dim_state, dim_state))
        # x_hats = []
        # kalman_covs = []
        for i in range(0, nb_particles):
            sigma = self.kalman_covs[i]
            x_hat, sigma = self.compute_ekf_proposal(xp[i], data[t], sigma)
            x_hats[i, :] = x_hat
            kalman_covs[i, :, :] = sigma
            #x_hats.append(x_hat.flatten())
            #kalman_covs.append(sigma)
        self.kalman_covs = kalman_covs
        mean = x_hats
        covar = 1.0*np.mean(kalman_covs, axis=0)
        # mean = np.reshape(x_hats, (-1, ))
        # covar = block_diag(*kalman_covs)

        return dists.MvNormal(loc=mean, cov=covar)
        # return dists.IndepProd(*[dists.MvNormal(loc=x_hats[k], cov=kalman_covs[k]) for k in range(0, self.dim_states)])
