import numpy as np
import matplotlib.pyplot as plt

from ReadData import DataReader
from Plotting import Plotter


def jacobian_numeric(f, x, rows=None, dx=0.001):
    """Numerical Jacobian of f at x, allows for non square output"""
    if rows is None:
        rows = len(x)
    J = np.empty((rows, len(x)))
    xpdx = np.empty_like(x)
    xmdx = np.empty_like(x)
    for j in range(len(x)):
        xpdx[:] = xmdx[:] = x
        xpdx[j] += dx
        xmdx[j] -= dx
        f_j = f(xpdx) - f(xmdx)
        J[:, j] = f_j / dx
    return J


def jacobian_analytic(x):
    g = CONST_GRAVITATION
    legs = np.array([0.5, 0.6, 0.5, 0.6])
    cst = np.array([0.34, 0.29, 0.315, 0.33])

    df = np.zeros((36, DIM_STATES))
    # left femur
    df[0, 2] = -x[12] * np.sin(x[2]) + (x[13] + g) * np.cos(x[2])
    df[0, 12] = np.cos(x[2])
    df[0, 13] = np.sin(x[2])
    df[0, 14] = cst[0]
    df[1, 2] = -x[12] * np.cos(x[2]) - (x[13] + g) * np.sin(x[2])
    df[1, 8] = 2 * x[8] * cst[0]
    df[1, 12] = -np.sin(x[2])
    df[1, 13] = np.cos(x[2])
    df[5, 8] = 1

    # left fibula
    df[6, 2] = -x[12] * np.sin(x[2] + x[3]) + x[13] * np.cos(x[2] + x[3]) + g * np.cos(x[2] + x[3])
    df[6, 3] = -x[14] * legs[0] * np.sin(x[3]) - x[12] * np.sin(x[2] + x[3]) + x[13] * np.cos(
        x[2] + x[3]) + legs[0] * x[8] ** 2 * np.cos(x[3]) + g * np.cos(x[2] + x[3])
    df[6, 8] = 2 * legs[0] * x[8] * np.sin(x[3])
    df[6, 12] = np.cos(x[2] + x[3])
    df[6, 13] = np.sin(x[2] + x[3])
    df[6, 14] = cst[1] + legs[0] * np.cos(x[3])
    df[6, 15] = cst[1]
    df[7, 2] = -x[12] * np.cos(x[2] + x[3]) - x[13] * np.sin(x[2] + x[3]) - g * np.sin(x[2] + x[3])
    df[7, 3] = -x[14] * legs[0] * np.cos(x[3]) - x[12] * np.cos(x[2] + x[3]) - x[13] * np.sin(
        x[2] + x[3]) - legs[0] * x[8] ** 2 * np.sin(x[3]) - g * np.sin(x[2] + x[3])
    df[7, 8] = 2 * legs[0] * x[8] * np.cos(x[3]) + 2 * cst[1] * (x[8] + x[9])
    df[7, 9] = 2 * cst[1] * (x[8] + x[9])
    df[7, 12] = -np.sin(x[2] + x[3])
    df[7, 13] = np.cos(x[2] + x[3])
    df[7, 14] = -legs[0] * np.sin(x[3])
    df[11, 8] = 1
    df[11, 9] = 1

    # right femur
    df[12, 2] = -x[12] * np.sin(x[4]) + (x[13] + g) * np.cos(x[4])
    df[12, 12] = np.cos(x[4])
    df[12, 13] = np.sin(x[4])
    df[12, 14] = cst[2]
    df[13, 2] = -x[12] * np.cos(x[4]) - (x[13] + g) * np.sin(x[4])
    df[13, 8] = 2 * x[10] * cst[2]
    df[13, 12] = -np.sin(x[4])
    df[13, 13] = np.cos(x[4])
    df[17, 10] = 1

    # right fibula
    df[18, 4] = -x[12] * np.sin(x[4] + x[5]) + x[13] * np.cos(x[4] + x[5]) + g * np.cos(x[4] + x[5])
    df[18, 5] = -x[16] * legs[2] * np.sin(x[5]) - x[12] * np.sin(x[4] + x[5]) + x[13] * np.cos(
        x[4] + x[5]) + legs[2] * x[10] ** 2 * np.cos(x[5]) + g * np.cos(x[4] + x[5])
    df[18, 10] = 2 * legs[2] * x[10] * np.sin(x[5])
    df[18, 12] = np.cos(x[4] + x[5])
    df[18, 13] = np.sin(x[4] + x[5])
    df[18, 16] = cst[3] + legs[2] * np.cos(x[5])
    df[18, 17] = cst[3]
    df[19, 4] = -x[12] * np.cos(x[4] + x[5]) - x[13] * np.sin(x[4] + x[5]) - g * np.sin(x[4] + x[5])
    df[19, 5] = -x[16] * legs[2] * np.cos(x[5]) - x[12] * np.cos(x[4] + x[5]) - x[13] * np.sin(
        x[4] + x[5]) - legs[2] * x[10] ** 2 * np.sin(x[5]) - g * np.sin(x[4] + x[5])
    df[19, 10] = 2 * legs[2] * x[10] * np.cos(x[5]) + 2 * cst[3] * (x[10] + x[11])
    df[19, 12] = 2 * cst[3] * (x[10] + x[11])
    df[19, 13] = -np.sin(x[4] + x[5])
    df[19, 16] = np.cos(x[4] + x[5])
    df[19, 17] = -legs[2] * np.sin(x[5])
    df[23, 10] = 1
    df[23, 11] = 1

    # left heel
    df[24, 2] = -x[8] * legs[0] + np.sin(x[2]) - legs[1] * (x[8] + x[9]) * np.sin(x[2] + x[3])
    df[24, 3] = -legs[1] * (x[8] + x[9]) * np.sin(x[2] + x[3])
    df[24, 6] = 1
    df[24, 8] = legs[0] * np.cos(x[2]) + legs[1] * np.cos(x[2] + x[3])
    df[24, 9] = legs[1] * np.cos(x[2] + x[3])
    df[25, 3] = x[8] * legs[0] * np.cos(x[2]) + legs[1] * (x[8] + x[9]) * np.cos(x[2] + x[3])
    df[25, 4] = legs[1] * (x[8] + x[9]) * np.cos(x[2] + x[3])
    df[25, 7] = 1
    df[25, 8] = legs[0] * np.sin(x[2]) + legs[1] * np.sin(x[2] + x[3])
    df[25, 9] = legs[1] * np.sin(x[2] + x[3])
    df[27, 2] = -legs[0] * (x[14] * np.sin(x[2]) + x[8] ** 2 * np.cos(x[2])) - legs[1] * (
            x[14] + x[15]) * np.sin(x[2] + x[3]) - legs[1] * (x[8] + x[9]) ** 2 * np.cos(
        x[2] + x[3])
    df[27, 3] = -legs[1] * (x[14] + x[15]) * np.sin(x[2] + x[3]) - legs[1] * (
            x[8] + x[9]) ** 2 * np.cos(x[2] + x[3])
    df[27, 8] = -2 * legs[0] * x[8] * np.sin(x[2]) - 2 * legs[1] * (x[8] + x[9]) * np.sin(
        x[2] + x[3])
    df[27, 9] = -2 * legs[1] * (x[8] + x[9]) * np.sin(x[2] + x[3])
    df[27, 12] = 1
    df[27, 14] = legs[0] * np.cos(x[2]) + legs[1] * np.cos(x[2] + x[3])
    df[27, 15] = legs[1] * np.cos(x[2] + x[3])
    df[28, 2] = -legs[0] * (x[14] * np.cos(x[2]) - x[8] ** 2 * np.sin(x[2])) + legs[1] * (
            x[14] + x[15]) * np.cos(x[2] + x[3]) - legs[1] * (x[8] + x[9]) ** 2 * np.sin(
        x[2] + x[3])
    df[28, 3] = legs[1] * (x[14] + x[15]) * np.cos(x[2] + x[3]) - legs[1] * (
            x[8] + x[9]) ** 2 * np.sin(x[2] + x[3])
    df[28, 8] = 2 * legs[0] * x[8] * np.cos(x[2]) + 2 * legs[1] * (x[8] + x[9]) * np.cos(
        x[2] + x[3])
    df[28, 9] = 2 * legs[1] * (x[8] + x[9]) * np.cos(x[2] + x[3])
    df[28, 13] = 1
    df[28, 14] = legs[0] * np.sin(x[2]) + legs[1] * np.sin(x[2] + x[3])
    df[28, 15] = legs[1] * np.sin(x[2] + x[3])

    # right heel
    df[30, 4] = -x[10] * legs[2] + np.sin(x[4]) - legs[3] * (x[10] + x[11]) * np.sin(x[4] + x[5])
    df[30, 5] = -legs[3] * (x[10] + x[11]) * np.sin(x[4] + x[5])
    df[30, 6] = 1
    df[30, 10] = legs[2] * np.cos(x[4]) + legs[3] * np.cos(x[4] + x[5])
    df[30, 11] = legs[3] * np.cos(x[4] + x[5])
    df[31, 3] = x[10] * legs[2] * np.cos(x[4]) + legs[3] * (x[10] + x[11]) * np.cos(x[4] + x[5])
    df[31, 4] = legs[3] * (x[10] + x[11]) * np.cos(x[4] + x[5])
    df[31, 7] = 1
    df[31, 10] = legs[2] * np.sin(x[4]) + legs[3] * np.sin(x[4] + x[5])
    df[31, 11] = legs[3] * np.sin(x[4] + x[5])
    df[33, 4] = -legs[2] * (x[16] * np.sin(x[4]) + x[10] ** 2 * np.cos(x[4])) - legs[3] * (
            x[16] + x[17]) * np.sin(x[4] + x[5]) - legs[3] * (x[10] + x[11]) ** 2 * np.cos(
        x[4] + x[5])
    df[33, 5] = -legs[3] * (x[16] + x[17]) * np.sin(x[4] + x[5]) - legs[3] * (
            x[10] + x[11]) ** 2 * np.cos(x[4] + x[5])
    df[33, 10] = -2 * legs[2] * x[10] * np.sin(x[4]) - 2 * legs[3] * (x[10] + x[11]) * np.sin(
        x[4] + x[5])
    df[33, 11] = -2 * legs[3] * (x[10] + x[11]) * np.sin(x[4] + x[5])
    df[33, 12] = 1
    df[33, 16] = legs[2] * np.cos(x[4]) + legs[3] * np.cos(x[4] + x[5])
    df[33, 17] = legs[3] * np.cos(x[4] + x[5])
    df[34, 4] = -legs[2] * (x[16] * np.cos(x[4]) - x[10] ** 2 * np.sin(x[4])) + legs[3] * (
            x[16] + x[17]) * np.cos(x[4] + x[5]) - legs[3] * (x[10] + x[11]) ** 2 * np.sin(
        x[4] + x[5])
    df[34, 5] = legs[3] * (x[16] + x[17]) * np.cos(x[4] + x[5]) - legs[3] * (
            x[10] + x[11]) ** 2 * np.sin(x[4] + x[5])
    df[34, 10] = 2 * legs[2] * x[10] * np.cos(x[4]) + 2 * legs[3] * (x[10] + x[11]) * np.cos(
        x[4] + x[5])
    df[34, 11] = 2 * legs[3] * (x[10] + x[11]) * np.cos(x[4] + x[5])
    df[34, 13] = 1
    df[34, 16] = legs[2] * np.sin(x[4]) + legs[3] * np.sin(x[4] + x[5])
    df[34, 17] = legs[3] * np.sin(x[4] + x[5])

    if DIM_OBSERVATIONS == 20:
        df = df[(0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34), :]

    return df


class EKF:
    def __init__(self, f, h, x0, Q, H):
        """EKF(f, h, x0, Q, H)
        x' = f(x, u)    state transition function
        z' = h(x)       observation function
        x0              initial state estimate"""
        self.f, self.h = f, h
        self.x = np.array(x0)
        self.I, self.P = np.eye(len(x0)), np.eye(len(x0))
        # modify these if the noise is not independent
        self.Q, self.H = Q, H

    def predict(self):
        # F = jacobian_numeric(lambda x: self.f(x), self.x)
        dt = CONST_TIMESTEP
        A = np.eye(DIM_STATES)
        for row in range(0, DIM_STATES):
            for col in range(0, DIM_STATES):
                if row + 6 == col:
                    A[row, col] = dt
                if row + 12 == col:
                    A[row, col] = dt ** 2 / 2.0
        F = A
        self.x = self.f(self.x)
        self.P = (F @ self.P @ F.T) + self.Q

    def update(self, z):
        x, P, H, I, h = self.x, self.P, self.H, self.I, self.h
        # df = jacobian_numeric(h, x, len(z))
        df = jacobian_analytic(x)
        y = z - h(x)
        S = (df @ P @ df.T) + H
        K = P @ df.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (I - K @ df) @ P


def f(xp):
    dt = CONST_TIMESTEP
    A = np.eye(DIM_STATES)
    for row in range(0, DIM_STATES):
        for col in range(0, DIM_STATES):
            if row + 6 == col:
                A[row, col] = dt
            if row + 12 == col:
                A[row, col] = dt ** 2 / 2.0
    return np.matmul(A, xp)


def h(x):
    y = np.empty(DIM_OBSERVATIONS)
    g = CONST_GRAVITATION
    legs = np.array([0.5, 0.6, 0.5, 0.6])
    cst = np.array([0.34, 0.29, 0.315, 0.33])

    y[0] = cst[0] * x[14] + g * np.sin(x[2]) + np.sin(x[2]) * x[13] + np.cos(x[2]) * x[12]
    y[1] = cst[0] * x[8] ** 2 + g * np.cos(x[2]) - np.sin(x[2]) * x[12] + np.cos(x[2]) * x[13]
    y[2] = 0.0
    y[3] = 0.0
    y[4] = 0.0
    y[5] = x[8]

    # left fibula
    y[6] = cst[1] * x[14] + cst[1] * x[15] + g * np.sin(x[2] + x[3]) + legs[0] * np.sin(x[3]) * \
           x[8] ** 2 + legs[0] * np.cos(x[3]) * x[14] + np.sin(x[2] + x[3]) * x[13] + np.cos(x[2] + x[3]) * x[
               12]
    y[7] = cst[1] * x[8] ** 2 + 2 * cst[1] * x[8] * x[9] + cst[1] * x[9] ** 2 + g * np.cos(
        x[2] + x[3]) - legs[0] * np.sin(x[3]) * x[14] + legs[0] * np.cos(x[3]) * x[8] ** 2 - np.sin(
        x[2] + x[3]) * x[12] + np.cos(x[2] + x[3]) * x[13]
    y[8] = 0.0
    y[9] = 0.0
    y[10] = 0.0
    y[11] = x[8] + x[9]

    # right femur
    y[12] = cst[2] * x[16] + g * np.sin(x[4]) + np.sin(x[4]) * x[13] + np.cos(x[4]) * x[12]
    y[13] = cst[2] * x[10] ** 2 + g * np.cos(x[4]) - np.sin(x[4]) * x[12] + np.cos(x[4]) * x[13]
    y[14] = 0.0
    y[15] = 0.0
    y[16] = 0.0
    y[17] = x[10]

    # right fibula
    y[18] = cst[3] * x[16] + cst[3] * x[17] + g * np.sin(x[4] + x[5]) + legs[2] * np.sin(x[5]) * \
            x[10] ** 2 + legs[2] * np.cos(x[5]) * x[16] + np.sin(x[4] + x[5]) * x[13] + np.cos(x[4] + x[5]) * \
            x[12]
    y[19] = cst[3] * x[10] ** 2 + 2 * cst[3] * x[10] * x[11] + cst[3] * x[11] ** 2 + g * np.cos(
        x[4] + x[5]) - legs[2] * np.sin(x[5]) * x[16] + legs[2] * np.cos(x[5]) * x[10] ** 2 - np.sin(
        x[4] + x[5]) * x[12] + np.cos(x[4] + x[5]) * x[13]
    y[20] = 0.0
    y[21] = 0.0
    y[22] = 0.0
    y[23] = x[10] + x[11]

    # left heel
    y[24] = legs[0] * np.cos(x[2]) * x[8] + legs[1] * (x[8] + x[9]) * np.cos(x[2] + x[3]) + x[6]
    y[25] = legs[0] * np.sin(x[2]) * x[8] + legs[1] * (x[8] + x[9]) * np.sin(x[2] + x[3]) + x[7]
    y[26] = 0.0
    y[27] = -legs[0] * np.sin(x[2]) * x[8] ** 2 + legs[0] * np.cos(x[2]) * x[14] - legs[1] * (
            x[8] + x[9]) ** 2 * np.sin(x[2] + x[3]) + legs[1] * (x[14] + x[15]) * np.cos(x[2] + x[3]) + x[
                12]
    y[28] = legs[0] * np.sin(x[2]) * x[14] + legs[0] * np.cos(x[2]) * x[8] ** 2 + legs[1] * (
            x[8] + x[9]) ** 2 * np.cos(x[2] + x[3]) + legs[1] * (x[14] + x[15]) * np.sin(x[2] + x[3]) + x[
                13]
    y[29] = 0.0

    # right heel
    y[30] = legs[2] * np.cos(x[4]) * x[10] + legs[3] * (x[10] + x[11]) * np.cos(x[4] + x[5]) + x[6]
    y[31] = legs[2] * np.sin(x[4]) * x[10] + legs[3] * (x[10] + x[11]) * np.sin(x[4] + x[5]) + x[7]
    y[32] = 0.0
    y[33] = -legs[2] * np.sin(x[4]) * x[10] ** 2 + legs[2] * np.cos(x[4]) * x[16] - legs[3] * (
            x[10] + x[11]) ** 2 * np.sin(x[4] + x[5]) + legs[3] * (x[16] + x[17]) * np.cos(x[4] + x[5]) + \
            x[12]
    y[34] = legs[2] * np.sin(x[4]) * x[16] + legs[2] * np.cos(x[4]) * x[10] ** 2 + legs[3] * (
            x[10] + x[11]) ** 2 * np.cos(x[4] + x[5]) + legs[3] * (x[16] + x[17]) * np.sin(x[4] + x[5]) + \
            x[13]
    y[35] = 0.0

    return y


def compute_matrix_theory(dt, sx, sy, sphi):
    Q = np.array([[sx * dt ** 5 / 20, 0.0, 0.0, 0.0, 0.0, 0.0, sx * dt ** 4 / 8, 0.0, 0.0, 0.0, 0.0, 0.0,
                   sx * dt ** 3 / 6, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, sy * dt ** 5 / 20, 0.0, 0.0, 0.0, 0.0, 0.0, sy * dt ** 4 / 8, 0.0, 0.0, 0.0, 0.0,
                   0.0, sy * dt ** 3 / 6, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, sphi * dt ** 5 / 20, 0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 4 / 8, 0.0, 0.0, 0.0,
                   0.0, 0.0, sphi * dt ** 3 / 6, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, sphi * dt ** 5 / 20, 0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 4 / 8, 0.0, 0.0,
                   0.0, 0.0, 0.0, sphi * dt ** 3 / 6, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, sphi * dt ** 5 / 20, 0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 4 / 8, 0.0,
                   0.0, 0.0, 0.0, 0.0, sphi * dt ** 3 / 6, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 5 / 20, 0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 4 / 8,
                   0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 3 / 6],
                  [sx * dt ** 4 / 8, 0.0, 0.0, 0.0, 0.0, 0.0, sx * dt ** 3 / 3, 0.0, 0.0, 0.0, 0.0, 0.0,
                   sx * dt ** 2 / 2, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, sy * dt ** 4 / 8, 0.0, 0.0, 0.0, 0.0, 0.0, sy * dt ** 3 / 3, 0.0, 0.0, 0.0, 0.0,
                   0.0, sy * dt ** 2 / 2, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, sphi * dt ** 4 / 8, 0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 3 / 3, 0.0, 0.0, 0.0,
                   0.0, 0.0, sphi * dt ** 2 / 2, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, sphi * dt ** 4 / 8, 0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 3 / 3, 0.0, 0.0,
                   0.0, 0.0, 0.0, sphi * dt ** 2 / 2, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, sphi * dt ** 4 / 8, 0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 3 / 3, 0.0,
                   0.0, 0.0, 0.0, 0.0, sphi * dt ** 2 / 2, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 4 / 8, 0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 3 / 3,
                   0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 2 / 2],
                  [sx * dt ** 3 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, sx * dt ** 2 / 2, 0.0, 0.0, 0.0, 0.0, 0.0,
                   sx * dt, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, sy * dt ** 3 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, sy * dt ** 2 / 2, 0.0, 0.0, 0.0, 0.0,
                   0.0, sy * dt, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, sphi * dt ** 3 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 2 / 2, 0.0, 0.0, 0.0,
                   0.0, 0.0, sphi * dt, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, sphi * dt ** 3 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 2 / 2, 0.0, 0.0,
                   0.0, 0.0, 0.0, sphi * dt, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, sphi * dt ** 3 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 2 / 2, 0.0,
                   0.0, 0.0, 0.0, 0.0, sphi * dt, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 3 / 6, 0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt ** 2 / 2,
                   0.0, 0.0, 0.0, 0.0, 0.0, sphi * dt]])
    try:
        np.linalg.cholesky(Q)
        print('A is positive definite.')
        return Q
    except np.linalg.LinAlgError:
        print('A is NOT positive definite.')
        return None


DIM_STATES = 18
DIM_OBSERVATIONS = 36
CONST_GRAVITATION = 9.81
CONST_TIMESTEP = 0.01

# -------- Data -----------
path_truth = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Normal/truth_normal.dat'
path_obs = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Normal/noised_observations_normal.dat'
data_reader = DataReader()
max_timesteps = 1000
data_reader.read_states_as_arr(path_truth, max_timesteps=max_timesteps)
data_reader.read_observations_as_arr(path_obs, max_timesteps=max_timesteps)
data_reader.prepare_lists()
truth = data_reader.true_states
obs = data_reader.observations

# -------- EKF -----------
a = np.array([0.01, 1.06, -0.13, -0.25, 0.37, -0.19,
              0.57, 0.10, 2.54, -3.8, -0.08, -0.82,
              -0.00, 0.01, -1.78, 3.32, -0.30, 0.54])
sigma_x = 0.1
sigma_y = 1.0
sigma_phi = 10.0
Q = compute_matrix_theory(dt=0.01, sx=sigma_x, sy=sigma_y, sphi=sigma_phi)
sigma_imu_acc = 0.1
sigma_imu_gyro = 0.01
sigma_press_velo = 0.1
sigma_press_acc = 10.0
H = 1.0 * np.diag([sigma_imu_acc, sigma_imu_acc, sigma_imu_acc, sigma_imu_gyro, sigma_imu_gyro, sigma_imu_gyro,
                   sigma_imu_acc, sigma_imu_acc, sigma_imu_acc, sigma_imu_gyro, sigma_imu_gyro, sigma_imu_gyro,
                   sigma_imu_acc, sigma_imu_acc, sigma_imu_acc, sigma_imu_gyro, sigma_imu_gyro, sigma_imu_gyro,
                   sigma_imu_acc, sigma_imu_acc, sigma_imu_acc, sigma_imu_gyro, sigma_imu_gyro, sigma_imu_gyro,
                   sigma_press_velo, sigma_press_velo, sigma_press_velo, sigma_press_acc, sigma_press_acc, sigma_imu_acc,
                   sigma_press_velo, sigma_press_velo, sigma_press_velo, sigma_press_acc, sigma_press_acc, sigma_imu_acc])

my_ekf = EKF(f=f, h=h, x0=a, Q=Q, H=H)

# -------- Simulation -----------
x_vals = []
y_vals = []
for t in range(0, max_timesteps):
    my_ekf.predict()
    x_vals.append(my_ekf.x)
    my_ekf.update(obs[t])

# -------- Plotting -----------
truth = np.reshape(truth, (max_timesteps, 1, DIM_STATES))
x_vals = np.reshape(x_vals, (max_timesteps, 1, DIM_STATES))
my_plotter = Plotter(samples=x_vals, truth=truth, export_name='scratch_ekf', delta_t=CONST_TIMESTEP)
my_plotter.plot_samples_detail()
