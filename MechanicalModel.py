import numpy as np


class MechanicalModel:
    def __init__(self, dt, dim_state, dim_observations, imu_position, leg_constants):
        self.dt = dt
        self.dim_states = dim_state
        self.dim_observations = dim_observations
        self.A = np.zeros((self.dim_states, self.dim_states))
        self.set_process_transition_matrix()
        self.g = 9.81
        self.cst = imu_position
        self.legs = leg_constants

    def set_process_transition_matrix(self):
        self.A = np.eye(self.dim_states)
        for row in range(0, self.dim_states):
            for col in range(0, self.dim_states):
                if row + 6 == col:
                    self.A[row, col] = self.dt
                if row + 12 == col:
                    self.A[row, col] = self.dt ** 2 / 2.0
        return None

    def state_transition(self, x):
        return np.matmul(self.A, x.T).T

    def state_to_observation(self, x):
        nb_steps, _ = x.shape
        y = np.empty(shape=(nb_steps, 36))
        # left femur
        y[:, 0] = self.cst[0] * x[:, 14] + self.g * np.sin(x[:, 2]) + np.sin(x[:, 2]) * x[:, 13] + np.cos(x[:, 2]) \
            * x[:, 12]
        y[:, 1] = self.cst[0] * x[:, 8] ** 2 + self.g * np.cos(x[:, 2]) - np.sin(x[:, 2]) * x[:, 12] \
            + np.cos(x[:, 2]) * x[:, 13]
        y[:, 2] = 0.0
        y[:, 3] = 0.0
        y[:, 4] = 0.0
        y[:, 5] = x[:, 8]

        # left fibula
        y[:, 6] = self.cst[1] * x[:, 14] + self.cst[1] * x[:, 15] + self.g * np.sin(x[:, 2] + x[:, 3]) + self.legs[0] \
            * np.sin(x[:, 3]) * x[:, 8] ** 2 + self.legs[0] * np.cos(x[:, 3]) * x[:, 14] \
            + np.sin(x[:, 2] + x[:, 3]) * x[:, 13] + np.cos(x[:, 2] + x[:, 3]) * x[:, 12]
        y[:, 7] = self.cst[1] * x[:, 8] ** 2 + 2 * self.cst[1] * x[:, 8] * x[:, 9] + self.cst[1] * x[:, 9] ** 2 \
            + self.g * np.cos(x[:, 2] + x[:, 3]) - self.legs[0] * np.sin(x[:, 3]) * x[:, 14] + self.legs[0] \
            * np.cos(x[:, 3]) * x[:, 8] ** 2 - np.sin(x[:, 2] + x[:, 3]) * x[:, 12] + np.cos(x[:, 2] + x[:, 3]) \
            * x[:, 13]
        y[:, 8] = 0.0
        y[:, 9] = 0.0
        y[:, 10] = 0.0
        y[:, 11] = x[:, 8] + x[:, 9]

        # right femur
        y[:, 12] = self.cst[2] * x[:, 16] + self.g * np.sin(x[:, 4]) + np.sin(x[:, 4]) * x[:, 13] + np.cos(x[:, 4]) \
            * x[:, 12]
        y[:, 13] = self.cst[2] * x[:, 10] ** 2 + self.g * np.cos(x[:, 4]) - np.sin(x[:, 4]) * x[:, 12] \
            + np.cos(x[:, 4]) * x[:, 13]
        y[:, 14] = 0.0
        y[:, 15] = 0.0
        y[:, 16] = 0.0
        y[:, 17] = x[:, 10]

        # right fibula
        y[:, 18] = self.cst[3] * x[:, 16] + self.cst[3] * x[:, 17] + self.g * np.sin(x[:, 4] + x[:, 5]) + self.legs[2] \
            * np.sin(x[:, 5]) * x[:, 10] ** 2 + self.legs[2] * np.cos(x[:, 5]) * x[:, 16] \
            + np.sin(x[:, 4] + x[:, 5]) * x[:, 13] + np.cos(x[:, 4] + x[:, 5]) * x[:, 12]
        y[:, 19] = self.cst[3] * x[:, 10] ** 2 + 2 * self.cst[3] * x[:, 10] * x[:, 11] + self.cst[3] * x[:, 11] ** 2 \
            + self.g * np.cos(x[:, 4] + x[:, 5]) - self.legs[2] * np.sin(x[:, 5]) * x[:, 16] + self.legs[2] \
            * np.cos(x[:, 5]) * x[:, 10] ** 2 - np.sin(x[:, 4] + x[:, 5]) * x[:, 12] + np.cos(x[:, 4] + x[:, 5]) \
            * x[:, 13]
        y[:, 20] = 0.0
        y[:, 21] = 0.0
        y[:, 22] = 0.0
        y[:, 23] = x[:, 10] + x[:, 11]

        # left heel
        y[:, 24] = self.legs[0] * np.cos(x[:, 2]) * x[:, 8] + self.legs[1] * (x[:, 8] + x[:, 9]) \
            * np.cos(x[:, 2] + x[:, 3]) + x[:, 6]
        y[:, 25] = self.legs[0] * np.sin(x[:, 2]) * x[:, 8] + self.legs[1] * (x[:, 8] + x[:, 9]) \
            * np.sin(x[:, 2] + x[:, 3]) + x[:, 7]
        y[:, 26] = 0.0
        y[:, 27] = -self.legs[0] * np.sin(x[:, 2]) * x[:, 8] ** 2 + self.legs[0] * np.cos(x[:, 2]) * x[:, 14] \
            - self.legs[1] * (x[:, 8] + x[:, 9]) ** 2 * np.sin(x[:, 2] + x[:, 3]) + self.legs[1] \
            * (x[:, 14] + x[:, 15]) * np.cos(x[:, 2] + x[:, 3]) + x[:, 12]
        y[:, 28] = self.legs[0] * np.sin(x[:, 2]) * x[:, 14] + self.legs[0] * np.cos(x[:, 2]) * x[:, 8] ** 2 \
            + self.legs[1] * (x[:, 8] + x[:, 9]) ** 2 * np.cos(x[:, 2] + x[:, 3]) + self.legs[1] \
            * (x[:, 14] + x[:, 15]) * np.sin(x[:, 2] + x[:, 3]) + x[:, 13]
        y[:, 29] = 0.0

        # right heel
        y[:, 30] = self.legs[2] * np.cos(x[:, 4]) * x[:, 10] + self.legs[3] * (x[:, 10] + x[:, 11]) \
            * np.cos(x[:, 4] + x[:, 5]) + x[:, 6]
        y[:, 31] = self.legs[2] * np.sin(x[:, 4]) * x[:, 10] + self.legs[3] * (x[:, 10] + x[:, 11]) \
            * np.sin(x[:, 4] + x[:, 5]) + x[:, 7]
        y[:, 32] = 0.0
        y[:, 33] = -self.legs[2] * np.sin(x[:, 4]) * x[:, 10] ** 2 + self.legs[2] * np.cos(x[:, 4]) * x[:, 16] \
            - self.legs[3] * (x[:, 10] + x[:, 11]) ** 2 * np.sin(x[:, 4] + x[:, 5]) + self.legs[3] \
            * (x[:, 16] + x[:, 17]) * np.cos(x[:, 4] + x[:, 5]) + x[:, 12]
        y[:, 34] = self.legs[2] * np.sin(x[:, 4]) * x[:, 16] + self.legs[2] * np.cos(x[:, 4]) * x[:, 10] ** 2 \
            + self.legs[3] * (x[:, 10] + x[:, 11]) ** 2 * np.cos(x[:, 4] + x[:, 5]) + self.legs[3] \
            * (x[:, 16] + x[:, 17]) * np.sin(x[:, 4] + x[:, 5]) + x[:, 13]
        y[:, 35] = 0.0
        if self.dim_observations == 20:
            y = y[:, (0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34), ]

        return y

    def compute_jacobian_observation(self, x):
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
                x[16] + x[17]) * np.cos(x[4] + x[5]) - self.legs[3] * (x[10] + x[11]) ** 2 * np.sin(
            x[4] + x[5])
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

    def state_to_observation_linear(self, x):
        nb_steps, _ = x.shape
        y = np.empty(shape=(nb_steps, self.dim_observations))
        for i in range(0, nb_steps):
            df = self.compute_jacobian_observation(x[i])
            y[i, :] = np.matmul(df, x[i])
        if self.dim_observations == 36:
            y[:, (1, 7, 13, 19)] += self.g
        elif self.dim_observations == 20:
            y[:, (1, 4, 7, 10)] += self.g
        else:
            raise AssertionError('Observations must have dimension 20 or 36; got {}.'.format(self.dim_observations))
        return y