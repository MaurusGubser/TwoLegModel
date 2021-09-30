import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import particles

from particles import state_space_models as ssm
from particles import distributions as dists

DIM_STATES = 18
DIM_OBSERVATIONS = 36


class TwoLegModel(ssm):
    """
    Two leg model...
    """

    def __init__(self, dt, leg_constants, a, P, Q, H):
        self.A = np.ones(DIM_STATES)
        for row in range(0, DIM_STATES):
            for col in range(0, DIM_STATES):
                if row + 6 == col:
                    self.A[row, col] = dt
                if row + 12 == col:
                    self.A[row, col] = dt ** 2 / 2.0
        self.g = 9.81
        self.cst = leg_constants
        self.a = a
        self.P = P
        self.Q = Q
        self.H = H
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

    def state_to_observation(self, x):
        """
        Transformation from state x to observation y
        """
        y = np.empty(shape=(36,))
        y[0] = self.cst[0] * x[14] + self.g * np.sin(x[2]) + np.sin(x[2]) * x[13] + np.cos(x[2]) * x[12]
        y[1] = self.cst[0] * x[8] ** 2 + self.g * np.cos(x[2]) - np.sin(x[2]) * x[12] + np.cos(x[2]) * x[13]
        y[2] = 0.0
        y[3] = 0.0
        y[4] = 0.0
        y[5] = x[8]

        y[6] = self.cst[1] * x[14] + self.cst[1] * x[15] + self.g * np.sin(x[2] + x[3]) + l_0 * np.sin(x[3]) * \
                 x[8] ** 2 + l_0 * np.cos(x[3]) * x[14] + np.sin(x[2] + x[3]) * x[13] + np.cos(x[2] + x[3]) * \
                 x[12]
        y[7] = self.cst[1] * x[8] ** 2 + 2 * self.cst[1] * x[8] * x[9] + self.cst[1] * x[
            9] ** 2 + self.g * np.cos(x[2] + x[3]) - l_0 * np.sin(x[3]) * x[14] + l_0 * np.cos(x[3]) * x[
                     8] ** 2 - np.sin(x[2] + x[3]) * x[12] + np.cos(x[2] + x[3]) * x[13]
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

        y[18] = self.cst[3] * x[16] + self.cst[3] * x[17] + self.g * np.sin(x[4] + x[5]) + l_2 * np.sin(x[5]) * \
                  x[10] + l_2 * np.cos(x[5]) * x[16] + np.sin(x[4] + x[5]) * x[13] + np.cos(x[4] + x[5]) * x[
                      12]
        y[19] = self.cst[3] * x[10] ** 2 + 2 * self.cst[3] * x[10] * x[11] + self.cst[3] * x[
            11] ** 2 + self.g * np.cos(x[4] + x[5]) - l_2 * np.sin(x[5]) * x[16] + l_2 * np.cos(x[5]) * x[
                      10] ** 2 - sin(x[4] + x[5]) * x[12] + np.cos(x[4] + x[5]) * x[13]
        y[20] = 0.0
        y[21] = 0.0
        y[22] = 0.0
        y[23] = x[10] + x[11]

        y[24] = l_0 * np.cos(x[2]) * x[8] + l_1 * (x[8] + x[9]) * np.cos(x[2] + x[3]) + x[6]
        y[25] = l_0 * np.sin(x[2]) * x[8] + l_1 * (x[8] + x[9]) * np.sin(x[2] + x[3]) + x[7]
        y[26] = 0.0
        y[27] = -l_0 * np.sin(x[2]) * x[8] ** 2 + l_0 * np.cos(x[2]) * x[14] - l_1 * (
                x[8] + x[9]) ** 2 * np.sin(x[2] + x[3]) + l_1 * (x[14] + x[15]) * np.cos(x[2] + x[3]) + x[
                      12]
        y[28] = l_0 * np.sin(x[2]) * x[14] + l_0 * np.cos(x[2]) * x[8] ** 2 + l_1 * (x[8] + x[9]) ** 2 * np.cos(
            x[2] + x[3]) + l_1 * (x[14] + x[15]) * np.sin(x[2] + x[3]) + x[13]
        y[29] = 0.0

        y[30] = l_2 * np.cos(x[4]) * x[10] + l_3 * (x[10] + x[11]) * np.cos(x[4] + x[5]) + x[6]
        y[31] = l_2 * np.sin(x[4]) * x[10] + l_3 * (x[10] + x[11]) * np.sin(x[4] + x[5]) + x[7]
        y[32] = 0.0
        y[33] = -l_2 * np.sin(x[4]) * x[10] ** 2 + l_2 * np.cos(x[4]) * x[16] - l_3 * (
                x[10] + x[11]) ** 2 * np.sin(x[4] + x[5]) + l_3 * (x[16] + x[17]) * np.cos(x[4] + x[5]) + \
                  x[12]
        y[34] = l_2 * np.sin(x[4]) * x[16] + l_2 * np.cos(x[4]) * x[10] ** 2 + l_3 * (
                x[10] + x[11]) ** 2 * np.cos(x[4] + x[5]) + l_3 * (x[16] + x[17]) * np.sin(x[4] + x[5]) + \
                  x[13]
        y[35] = 0.0

        return y

    def state_transition(self, xp):
        return self.A * xp

    def PX0(self):  # Distribution of X_0
        """
        # case for 6x3 states
        return dists.IndepProd(*[dists.Normal(loc=self.a, scale=self.P) for a, P in zip(self.a_init, self.P_init)])
        """
        return dists.MvNormal(loc=self.a, scale=self.P)

    def PX(self, xp):
        return dists.MvNormal(loc=self.state_transition(xp), scale=self.Q)

    def PY(self, x):
        return dists.MvNormal(loc=self.state_to_observation(x), scale=self.H)
