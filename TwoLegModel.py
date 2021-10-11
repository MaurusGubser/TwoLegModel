import itertools

import numpy as np

from particles import state_space_models as ssm
from particles import distributions as dists

DIM_STATES = 18
DIM_OBSERVATIONS = 36
CONST_GRAVITATION = 9.81


class TwoLegModel(ssm.StateSpaceModel):
    """
    Two leg model...
    """

    def __init__(self,
                 dt=0.01,
                 leg_constants=np.array([0.5, 0.6, 0.5, 0.6]),
                 imu_position=np.array([0.34, 0.29, 0.315, 0.33]),
                 a=np.array([0.01, 1.06, -0.13, -0.25, 0.37, -0.19,
                             0.57, 0.10, 2.54, -3.8, -0.08, -0.82,
                             -0.00, 0.01, -1.78, 3.32, -0.30, 0.54]),
                 P=0.1 * np.eye(DIM_STATES),
                 cov_step=0.1,
                 scale_x=1.0,
                 scale_y=1.0,
                 scale_phi=1.0,
                 sigma_x=1.0,
                 sigma_y=1.0,
                 sigma_phi=1.0,
                 sf_H=1.0,
                 H=np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01,
                            0.1, 0.1, 0.1, 0.01, 0.01, 0.01,
                            0.1, 0.1, 0.1, 0.01, 0.01, 0.01,
                            0.1, 0.1, 0.1, 0.01, 0.01, 0.01,
                            0.1, 0.1, 0.1, 1.0, 1.0, 1.0,
                            0.1, 0.1, 0.1, 1.0, 1.0, 1.0])):
        self.dt = dt
        self.A = np.zeros((DIM_STATES, DIM_STATES))
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
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_phi = sigma_phi
        self.Q = np.zeros((DIM_STATES, DIM_STATES))
        self.set_process_cov_theory()
        self.sf_H = sf_H
        self.H = H
        self.scale_H()

    """
        self.ax = a[0]
        self.Px = P[0]
        self.ay = a[1]
        self.Py = P[1]
        self.a0 = a[2]
        self.P0 = P[2]
        self.a1 = a[3]
        self.P1 = P[3]
        self.a2 = a[4]
        self.P2 = P[4]
        self.a3 = a[5]
        self.P3 = P[5]
        self.Qx = sigma_trans[0]
        self.Qy = sigma_trans[1]
        self.Qphi = sigma_trans[2]
        
        self.sigma_imu_acc = sigma_obs[0]
        self.sigma_imu_gyro = sigma_obs[1]
        self.sigma_press_velo = sigma_obs[2]
        self.sigma_press_acc = sigma_obs[3]
    """

    def set_process_transition_matrix(self):
        self.A = np.eye(DIM_STATES)
        for row in range(0, DIM_STATES):
            for col in range(0, DIM_STATES):
                if row + 6 == col:
                    self.A[row, col] = self.dt
                if row + 12 == col:
                    self.A[row, col] = self.dt ** 2 / 2.0
        return None

    def set_process_cov_theory(self):
        block_size = DIM_STATES // 3
        for row in range(0, DIM_STATES):
            for col in range(0, DIM_STATES):
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

    def set_process_cov_state_groups(self):
        sigmas = np.diag([self.sigma_x, self.sigma_y, self.sigma_phi, self.sigma_phi, self.sigma_phi, self.sigma_phi])
        self.Q = np.block([[sigmas/20.0, sigmas/8.0, sigmas/6.0],
                           [sigmas/8.0, sigmas/3.0, sigmas/2.0],
                           [sigmas/6.0, sigmas/2.0, sigmas]])
        return None

    def scale_H(self):
        self.H = self.sf_H * self.H
        return None

    def state_transition(self, xp):
        return np.matmul(self.A, xp.T).T

    def state_to_observation(self, x):
        """
        Transformation from state x to observation y
        """
        nb_parallel, _ = x.shape
        y = np.empty(shape=(nb_parallel, DIM_OBSERVATIONS))
        for i in range(0, nb_parallel):
            self.state_to_observation_1dim(x[i], y[i])
        return y

    def state_to_observation_1dim(self, x, y):
        y[0] = self.cst[0] * x[14] + self.g * np.sin(x[2]) + np.sin(x[2]) * x[13] + np.cos(x[2]) * x[12]
        y[1] = self.cst[0] * x[8] ** 2 + self.g * np.cos(x[2]) - np.sin(x[2]) * x[12] + np.cos(x[2]) * x[13]
        y[2] = 0.0
        y[3] = 0.0
        y[4] = 0.0
        y[5] = x[8]

        y[6] = self.cst[1] * x[14] + self.cst[1] * x[15] + self.g * np.sin(x[2] + x[3]) + self.legs[0] * np.sin(x[3]) * \
               x[8] ** 2 + self.legs[0] * np.cos(x[3]) * x[14] + np.sin(x[2] + x[3]) * x[13] + np.cos(x[2] + x[3]) * x[
                   12]
        y[7] = self.cst[1] * x[8] ** 2 + 2 * self.cst[1] * x[8] * x[9] + self.cst[1] * x[9] ** 2 + self.g * np.cos(
            x[2] + x[3]) - self.legs[0] * np.sin(x[3]) * x[14] + self.legs[0] * np.cos(x[3]) * x[8] ** 2 - np.sin(
            x[2] + x[3]) * x[12] + np.cos(x[2] + x[3]) * x[13]
        y[8] = 0.0
        y[9] = 0.0
        y[10] = 0.0
        y[11] = x[8] + x[9]

        y[12] = self.cst[2] * x[16] + self.g * np.sin(x[4]) + np.sin(x[4]) * x[13] + np.cos(x[4]) * x[12]
        y[13] = self.cst[2] * x[10] ** 2 + self.g * np.cos(x[4]) - np.sin(x[4]) * x[12] + np.cos(x[4]) * x[13]
        y[14] = 0.0
        y[15] = 0.0
        y[16] = 0.0
        y[17] = x[10]

        y[18] = self.cst[3] * x[16] + self.cst[3] * x[17] + self.g * np.sin(x[4] + x[5]) + self.legs[2] * np.sin(x[5]) * \
                x[10] ** 2 + self.legs[2] * np.cos(x[5]) * x[16] + np.sin(x[4] + x[5]) * x[13] + np.cos(x[4] + x[5]) * \
                x[12]
        y[19] = self.cst[3] * x[10] ** 2 + 2 * self.cst[3] * x[10] * x[11] + self.cst[3] * x[11] ** 2 + self.g * np.cos(
            x[4] + x[5]) - self.legs[2] * np.sin(x[5]) * x[16] + self.legs[2] * np.cos(x[5]) * x[10] ** 2 - np.sin(
            x[4] + x[5]) * x[12] + np.cos(x[4] + x[5]) * x[13]
        y[20] = 0.0
        y[21] = 0.0
        y[22] = 0.0
        y[23] = x[10] + x[11]

        y[24] = self.legs[0] * np.cos(x[2]) * x[8] + self.legs[1] * (x[8] + x[9]) * np.cos(x[2] + x[3]) + x[6]
        y[25] = self.legs[0] * np.sin(x[2]) * x[8] + self.legs[1] * (x[8] + x[9]) * np.sin(x[2] + x[3]) + x[7]
        y[26] = 0.0
        y[27] = -self.legs[0] * np.sin(x[2]) * x[8] ** 2 + self.legs[0] * np.cos(x[2]) * x[14] - self.legs[1] * (
                x[8] + x[9]) ** 2 * np.sin(x[2] + x[3]) + self.legs[1] * (x[14] + x[15]) * np.cos(x[2] + x[3]) + x[
                    12]
        y[28] = self.legs[0] * np.sin(x[2]) * x[14] + self.legs[0] * np.cos(x[2]) * x[8] ** 2 + self.legs[1] * (
                x[8] + x[9]) ** 2 * np.cos(x[2] + x[3]) + self.legs[1] * (x[14] + x[15]) * np.sin(x[2] + x[3]) + x[
                    13]
        y[29] = 0.0

        y[30] = self.legs[2] * np.cos(x[4]) * x[10] + self.legs[3] * (x[10] + x[11]) * np.cos(x[4] + x[5]) + x[6]
        y[31] = self.legs[2] * np.sin(x[4]) * x[10] + self.legs[3] * (x[10] + x[11]) * np.sin(x[4] + x[5]) + x[7]
        y[32] = 0.0
        y[33] = -self.legs[2] * np.sin(x[4]) * x[10] ** 2 + self.legs[2] * np.cos(x[4]) * x[16] - self.legs[3] * (
                x[10] + x[11]) ** 2 * np.sin(x[4] + x[5]) + self.legs[3] * (x[16] + x[17]) * np.cos(x[4] + x[5]) + \
                x[12]
        y[34] = self.legs[2] * np.sin(x[4]) * x[16] + self.legs[2] * np.cos(x[4]) * x[10] ** 2 + self.legs[3] * (
                x[10] + x[11]) ** 2 * np.cos(x[4] + x[5]) + self.legs[3] * (x[16] + x[17]) * np.sin(x[4] + x[5]) + \
                x[13]
        y[35] = 0.0

        return y

    def PX0(self):
        """
        # case for 6x3 states
        return dists.IndepProd(*[dists.Normal(loc=self.a, scale=self.P) for a, P in zip(self.a_init, self.P_init)])
        """
        return dists.MvNormal(loc=self.a, cov=self.P)

    def PX(self, t, xp):
        return dists.MvNormal(loc=self.state_transition(xp), cov=self.Q)

    def PY(self, t, xp, x):
        return dists.MvNormal(loc=self.state_to_observation(x), cov=self.H)


class TwoLegModelGuided(TwoLegModel):
    def proposal0(self, data):
        return self.PX0()

    def proposal(t, xp, data):  # a silly proposal

        return dists.Normal(loc=rho * xp + data[t], scale=self.sigma)