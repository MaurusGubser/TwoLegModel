import numpy as np


def create_rotation_matrix_z(alpha):
    """
    Create rotation matrix for rotation of angle alpha around z-axis
    :param alpha: float
        rotation angle in radian
    :return: np.ndarray
        rotation matrix R of shape (3, 3)
    """
    R = np.array(
        [[np.cos(alpha), np.sin(alpha), 0.0], [-np.sin(alpha), np.cos(alpha), 0.0],
         [0.0, 0.0, 1.0]])
    return R


def state_to_obs(x, dim_observations, g, legs, imus, R):
    """
    Compute state to observation transition for fixed two-leg-model
    :param x: np.ndarray
        N x 18-dimensional state vector where N is the number of particles and the 18
        dimension composed of $ (x_H, y_H, phi_0, phi_1, phi_2, phi_3) $ and the
        corresponding first and second (temporal) derivatives
    :param dim_observations: int
        Dimension of observation vector, should be either 20 or 36
    :param g: float
        gravitational constant $ g = 9.81m/s^{2} $
    :param legs: np.ndarray
        4-dimensional array, containing the length of left femur, left fibula, right
        femur, right fibula
    :param imus: np.ndarray
        4-dimensional array, containing the position of the four imus, measured from the
        hip or the knees, respectively
    :param R: np.ndarray
        Rotation matrix of shape (36, 36)
    :return: np.ndarray
        N x dim_observation-dimensional observation vector
    """
    nb_steps, _ = x.shape
    y = np.empty((nb_steps, 36))
    # left femur
    y[:, 0] = imus[0] * x[:, 14] + g * np.sin(x[:, 2]) + np.sin(x[:, 2]) * x[:,
                                                                           13] + np.cos(
        x[:, 2]) * x[:, 12]
    y[:, 1] = imus[0] * x[:, 8] ** 2 + g * np.cos(x[:, 2]) - np.sin(x[:, 2]) * x[:,
                                                                               12] + np.cos(
        x[:, 2]) * x[:, 13]
    y[:, 2] = 0.0
    y[:, 3] = 0.0
    y[:, 4] = 0.0
    y[:, 5] = x[:, 8]

    # left fibula
    y[:, 6] = imus[1] * x[:, 14] + imus[1] * x[:, 15] + g * np.sin(x[:, 2] + x[:, 3]) + \
              legs[0] * np.sin(
        x[:, 3]) * x[:, 8] ** 2 + legs[0] * np.cos(x[:, 3]) * x[:, 14] + np.sin(
        x[:, 2] + x[:, 3]) * x[:, 13] + np.cos(
        x[:, 2] + x[:, 3]) * x[:, 12]
    y[:, 7] = imus[1] * x[:, 8] ** 2 + 2 * imus[1] * x[:, 8] * x[:, 9] + imus[1] * x[:,
                                                                                   9] ** 2 + g * np.cos(
        x[:, 2] + x[:, 3]) - legs[0] * np.sin(x[:, 3]) * x[:, 14] + legs[0] * np.cos(
        x[:, 3]) * x[:, 8] ** 2 - np.sin(x[:, 2] + x[:, 3]) * x[:, 12] + np.cos(
        x[:, 2] + x[:, 3]) * x[:, 13]
    y[:, 8] = 0.0
    y[:, 9] = 0.0
    y[:, 10] = 0.0
    y[:, 11] = x[:, 8] + x[:, 9]

    # right femur
    y[:, 12] = imus[2] * x[:, 16] + g * np.sin(x[:, 4]) + np.sin(x[:, 4]) * x[:,
                                                                            13] + np.cos(
        x[:, 4]) * x[:, 12]
    y[:, 13] = imus[2] * x[:, 10] ** 2 + g * np.cos(x[:, 4]) - np.sin(x[:, 4]) * x[:,
                                                                                 12] + np.cos(
        x[:, 4]) * x[:, 13]
    y[:, 14] = 0.0
    y[:, 15] = 0.0
    y[:, 16] = 0.0
    y[:, 17] = x[:, 10]

    # right fibula
    y[:, 18] = imus[3] * x[:, 16] + imus[3] * x[:, 17] + g * np.sin(x[:, 4] + x[:, 5]) + \
               legs[2] * np.sin(
        x[:, 5]) * x[:, 10] ** 2 + legs[2] * np.cos(x[:, 5]) * x[:, 16] + np.sin(
        x[:, 4] + x[:, 5]) * x[:, 13] + np.cos(
        x[:, 4] + x[:, 5]) * x[:, 12]
    y[:, 19] = imus[3] * x[:, 10] ** 2 + 2 * imus[3] * x[:, 10] * x[:, 11] + imus[
        3] * x[:, 11] ** 2 + g * np.cos(
        x[:, 4] + x[:, 5]) - legs[2] * np.sin(x[:, 5]) * x[:, 16] + legs[2] * np.cos(
        x[:, 5]) * x[:, 10] ** 2 - np.sin(
        x[:, 4] + x[:, 5]) * x[:, 12] + np.cos(x[:, 4] + x[:, 5]) * x[:, 13]
    y[:, 20] = 0.0
    y[:, 21] = 0.0
    y[:, 22] = 0.0
    y[:, 23] = x[:, 10] + x[:, 11]

    # left heel
    y[:, 24] = legs[0] * np.cos(x[:, 2]) * x[:, 8] + legs[1] * (
            x[:, 8] + x[:, 9]) * np.cos(x[:, 2] + x[:, 3]) + x[:, 6]
    y[:, 25] = legs[0] * np.sin(x[:, 2]) * x[:, 8] + legs[1] * (
            x[:, 8] + x[:, 9]) * np.sin(x[:, 2] + x[:, 3]) + x[:, 7]
    y[:, 26] = 0.0
    y[:, 27] = -legs[0] * np.sin(x[:, 2]) * x[:, 8] ** 2 + legs[0] * np.cos(
        x[:, 2]) * x[:, 14] - legs[1] * (
                       x[:, 8] + x[:, 9]) ** 2 * np.sin(x[:, 2] + x[:, 3]) + legs[1] * (
                       x[:, 14] + x[:, 15]) * np.cos(
        x[:, 2] + x[:, 3]) + x[:, 12]
    y[:, 28] = legs[0] * np.sin(x[:, 2]) * x[:, 14] + legs[0] * np.cos(x[:, 2]) * x[:,
                                                                                  8] ** 2 + \
               legs[1] * (
                       x[:, 8] + x[:, 9]) ** 2 * np.cos(x[:, 2] + x[:, 3]) + legs[1] * (
                       x[:, 14] + x[:, 15]) * np.sin(
        x[:, 2] + x[:, 3]) + x[:, 13]
    y[:, 29] = 0.0

    # right heel
    y[:, 30] = legs[2] * np.cos(x[:, 4]) * x[:, 10] + legs[3] * (
            x[:, 10] + x[:, 11]) * np.cos(x[:, 4] + x[:, 5]) + x[:,
                                                               6]
    y[:, 31] = legs[2] * np.sin(x[:, 4]) * x[:, 10] + legs[3] * (
            x[:, 10] + x[:, 11]) * np.sin(x[:, 4] + x[:, 5]) + x[:,
                                                               7]
    y[:, 32] = 0.0
    y[:, 33] = -legs[2] * np.sin(x[:, 4]) * x[:, 10] ** 2 + legs[2] * np.cos(
        x[:, 4]) * x[:, 16] - legs[3] * (
                       x[:, 10] + x[:, 11]) ** 2 * np.sin(x[:, 4] + x[:, 5]) + legs[
                   3] * (x[:, 16] + x[:, 17]) * np.cos(
        x[:, 4] + x[:, 5]) + x[:, 12]
    y[:, 34] = legs[2] * np.sin(x[:, 4]) * x[:, 16] + legs[2] * np.cos(x[:, 4]) * x[:,
                                                                                  10] ** 2 + \
               legs[3] * (
                       x[:, 10] + x[:, 11]) ** 2 * np.cos(x[:, 4] + x[:, 5]) + legs[
                   3] * (x[:, 16] + x[:, 17]) * np.sin(
        x[:, 4] + x[:, 5]) + x[:, 13]
    y[:, 35] = 0.0

    y = np.matmul(R, y.T).T

    if dim_observations == 20:
        y = y[:,
            (0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34)]
    return y


def compute_jacobian_obs(x, dim_states, dim_observations, g, legs, imus, R):
    """
    Compute the Jacobian of the state-observation-transition at a given point
    :param x: np.ndarray
        N x 18-dimensional state vector where N is the number of particles and the 18
        dimension composed of $ (x_H, y_H, phi_0, phi_1, phi_2, phi_3) $ and the
        corresponding first and second (temporal) derivatives
    :param dim_states: int
        Dimension of state vector, should be 18
    :param dim_observations: int
        Dimension of observation vector, should be either 20 or 36
    :param g: float
            gravitational constant $ g = 9.81m/s^{2} $
    :param legs: np.ndarray
            4-dimensional array, containing the length of left femur, left fibula,
            right femur, right fibula
    :param imus: np.ndarray
            4-dimensional array, containing the position of the four imus, measured from
            the hip or the knees, respectively
    :param R: np.ndarray
            Rotation matrix of shape (36, 36)
    :return: np.ndarray
        N x dim_states x dim_observation-dimensional Jacobian
    """
    nb_particles, _ = x.shape
    df = np.zeros((nb_particles, 36, dim_states))
    # left femur
    df[:, 0, 2] = -x[:, 12] * np.sin(x[:, 2]) + (x[:, 13] + g) * np.cos(x[:, 2])
    df[:, 0, 12] = np.cos(x[:, 2])
    df[:, 0, 13] = np.sin(x[:, 2])
    df[:, 0, 14] = imus[0]
    df[:, 1, 2] = -x[:, 12] * np.cos(x[:, 2]) - (x[:, 13] + g) * np.sin(x[:, 2])
    df[:, 1, 8] = 2 * x[:, 8] * imus[0]
    df[:, 1, 12] = -np.sin(x[:, 2])
    df[:, 1, 13] = np.cos(x[:, 2])
    df[:, 5, 8] = 1

    # left fibula
    df[:, 6, 2] = -x[:, 12] * np.sin(x[:, 2] + x[:, 3]) + x[:, 13] * np.cos(
        x[:, 2] + x[:, 3]) + g * np.cos(
        x[:, 2] + x[:, 3])
    df[:, 6, 3] = -x[:, 14] * legs[0] * np.sin(x[:, 3]) - x[:, 12] * np.sin(
        x[:, 2] + x[:, 3]) + x[:, 13] * np.cos(
        x[:, 2] + x[:, 3]) + legs[0] * x[:, 8] ** 2 * np.cos(x[:, 3]) + g * np.cos(
        x[:, 2] + x[:, 3])
    df[:, 6, 8] = 2 * legs[0] * x[:, 8] * np.sin(x[:, 3])
    df[:, 6, 12] = np.cos(x[:, 2] + x[:, 3])
    df[:, 6, 13] = np.sin(x[:, 2] + x[:, 3])
    df[:, 6, 14] = imus[1] + legs[0] * np.cos(x[:, 3])
    df[:, 6, 15] = imus[1]
    df[:, 7, 2] = -x[:, 12] * np.cos(x[:, 2] + x[:, 3]) - x[:, 13] * np.sin(
        x[:, 2] + x[:, 3]) - g * np.sin(
        x[:, 2] + x[:, 3])
    df[:, 7, 3] = -x[:, 14] * legs[0] * np.cos(x[:, 3]) - x[:, 12] * np.cos(
        x[:, 2] + x[:, 3]) - x[:, 13] * np.sin(
        x[:, 2] + x[:, 3]) - legs[0] * x[:, 8] ** 2 * np.sin(x[:, 3]) - g * np.sin(
        x[:, 2] + x[:, 3])
    df[:, 7, 8] = 2 * legs[0] * x[:, 8] * np.cos(x[:, 3]) + 2 * imus[1] * (
            x[:, 8] + x[:, 9])
    df[:, 7, 9] = 2 * imus[1] * (x[:, 8] + x[:, 9])
    df[:, 7, 12] = -np.sin(x[:, 2] + x[:, 3])
    df[:, 7, 13] = np.cos(x[:, 2] + x[:, 3])
    df[:, 7, 14] = -legs[0] * np.sin(x[:, 3])
    df[:, 11, 8] = 1
    df[:, 11, 9] = 1

    # right femur
    df[:, 12, 4] = -x[:, 12] * np.sin(x[:, 4]) + (x[:, 13] + g) * np.cos(x[:, 4])
    df[:, 12, 12] = np.cos(x[:, 4])
    df[:, 12, 13] = np.sin(x[:, 4])
    df[:, 12, 16] = imus[2]
    df[:, 13, 4] = -x[:, 12] * np.cos(x[:, 4]) - (x[:, 13] + g) * np.sin(x[:, 4])
    df[:, 13, 10] = 2 * x[:, 10] * imus[2]
    df[:, 13, 12] = -np.sin(x[:, 4])
    df[:, 13, 13] = np.cos(x[:, 4])
    df[:, 17, 10] = 1

    # right fibula
    df[:, 18, 4] = -x[:, 12] * np.sin(x[:, 4] + x[:, 5]) + x[:, 13] * np.cos(
        x[:, 4] + x[:, 5]) + g * np.cos(
        x[:, 4] + x[:, 5])
    df[:, 18, 5] = -x[:, 16] * legs[2] * np.sin(x[:, 5]) - x[:, 12] * np.sin(
        x[:, 4] + x[:, 5]) + x[:, 13] * np.cos(
        x[:, 4] + x[:, 5]) + legs[2] * x[:, 10] ** 2 * np.cos(x[:, 5]) + g * np.cos(
        x[:, 4] + x[:, 5])
    df[:, 18, 10] = 2 * legs[2] * x[:, 10] * np.sin(x[:, 5])
    df[:, 18, 12] = np.cos(x[:, 4] + x[:, 5])
    df[:, 18, 13] = np.sin(x[:, 4] + x[:, 5])
    df[:, 18, 16] = imus[3] + legs[2] * np.cos(x[:, 5])
    df[:, 18, 17] = imus[3]
    df[:, 19, 4] = -x[:, 12] * np.cos(x[:, 4] + x[:, 5]) - x[:, 13] * np.sin(
        x[:, 4] + x[:, 5]) - g * np.sin(
        x[:, 4] + x[:, 5])
    df[:, 19, 5] = -x[:, 16] * legs[2] * np.cos(x[:, 5]) - x[:, 12] * np.cos(
        x[:, 4] + x[:, 5]) - x[:, 13] * np.sin(
        x[:, 4] + x[:, 5]) - legs[2] * x[:, 10] ** 2 * np.sin(x[:, 5]) - g * np.sin(
        x[:, 4] + x[:, 5])
    df[:, 19, 10] = 2 * legs[2] * x[:, 10] * np.cos(x[:, 5]) + 2 * imus[3] * (
            x[:, 10] + x[:, 11])
    df[:, 19, 11] = 2 * imus[3] * (x[:, 10] + x[:, 11])
    df[:, 19, 12] = -np.sin(x[:, 4] + x[:, 5])
    df[:, 19, 13] = np.cos(x[:, 4] + x[:, 5])
    df[:, 19, 16] = -legs[2] * np.sin(x[:, 5])
    df[:, 23, 10] = 1
    df[:, 23, 11] = 1

    # left heel
    df[:, 24, 2] = -x[:, 8] * legs[0] + np.sin(x[:, 2]) - legs[1] * (
            x[:, 8] + x[:, 9]) * np.sin(x[:, 2] + x[:, 3])
    df[:, 24, 3] = -legs[1] * (x[:, 8] + x[:, 9]) * np.sin(x[:, 2] + x[:, 3])
    df[:, 24, 6] = 1
    df[:, 24, 8] = legs[0] * np.cos(x[:, 2]) + legs[1] * np.cos(x[:, 2] + x[:, 3])
    df[:, 24, 9] = legs[1] * np.cos(x[:, 2] + x[:, 3])
    df[:, 25, 2] = x[:, 8] * legs[0] * np.cos(x[:, 2]) + legs[1] * (
            x[:, 8] + x[:, 9]) * np.cos(x[:, 2] + x[:, 3])
    df[:, 25, 3] = legs[1] * (x[:, 8] + x[:, 9]) * np.cos(x[:, 2] + x[:, 3])
    df[:, 25, 7] = 1
    df[:, 25, 8] = legs[0] * np.sin(x[:, 2]) + legs[1] * np.sin(x[:, 2] + x[:, 3])
    df[:, 25, 9] = legs[1] * np.sin(x[:, 2] + x[:, 3])
    df[:, 27, 2] = -legs[0] * (
            x[:, 14] * np.sin(x[:, 2]) + x[:, 8] ** 2 * np.cos(x[:, 2])) - legs[
                       1] * (
                           x[:, 14] + x[:, 15]) * np.sin(
        x[:, 2] + x[:, 3]) - legs[1] * (x[:, 8] + x[:, 9]) ** 2 * np.cos(
        x[:, 2] + x[:, 3])
    df[:, 27, 3] = -legs[1] * (x[:, 14] + x[:, 15]) * np.sin(x[:, 2] + x[:, 3]) - legs[
        1] * (
                           x[:, 8] + x[:, 9]) ** 2 * np.cos(x[:, 2] + x[:, 3])
    df[:, 27, 8] = -2 * legs[0] * x[:, 8] * np.sin(x[:, 2]) - 2 * legs[1] * (
            x[:, 8] + x[:, 9]) * np.sin(
        x[:, 2] + x[:, 3])
    df[:, 27, 9] = -2 * legs[1] * (x[:, 8] + x[:, 9]) * np.sin(x[:, 2] + x[:, 3])
    df[:, 27, 12] = 1
    df[:, 27, 14] = legs[0] * np.cos(x[:, 2]) + legs[1] * np.cos(x[:, 2] + x[:, 3])
    df[:, 27, 15] = legs[1] * np.cos(x[:, 2] + x[:, 3])
    df[:, 28, 2] = -legs[0] * (
            -x[:, 14] * np.cos(x[:, 2]) + x[:, 8] ** 2 * np.sin(x[:, 2])) + legs[
                       1] * (
                           x[:, 14] + x[:, 15]) * np.cos(
        x[:, 2] + x[:, 3]) - legs[1] * (x[:, 8] + x[:, 9]) ** 2 * np.sin(
        x[:, 2] + x[:, 3])
    df[:, 28, 3] = legs[1] * (x[:, 14] + x[:, 15]) * np.cos(x[:, 2] + x[:, 3]) - legs[
        1] * (
                           x[:, 8] + x[:, 9]) ** 2 * np.sin(x[:, 2] + x[:, 3])
    df[:, 28, 8] = 2 * legs[0] * x[:, 8] * np.cos(x[:, 2]) + 2 * legs[1] * (
            x[:, 8] + x[:, 9]) * np.cos(
        x[:, 2] + x[:, 3])
    df[:, 28, 9] = 2 * legs[1] * (x[:, 8] + x[:, 9]) * np.cos(x[:, 2] + x[:, 3])
    df[:, 28, 13] = 1
    df[:, 28, 14] = legs[0] * np.sin(x[:, 2]) + legs[1] * np.sin(x[:, 2] + x[:, 3])
    df[:, 28, 15] = legs[1] * np.sin(x[:, 2] + x[:, 3])

    # right heel
    df[:, 30, 4] = -x[:, 10] * legs[2] + np.sin(x[:, 4]) - legs[3] * (
            x[:, 10] + x[:, 11]) * np.sin(x[:, 4] + x[:, 5])
    df[:, 30, 5] = -legs[3] * (x[:, 10] + x[:, 11]) * np.sin(x[:, 4] + x[:, 5])
    df[:, 30, 6] = 1
    df[:, 30, 10] = legs[2] * np.cos(x[:, 4]) + legs[3] * np.cos(x[:, 4] + x[:, 5])
    df[:, 30, 11] = legs[3] * np.cos(x[:, 4] + x[:, 5])
    df[:, 31, 4] = x[:, 10] * legs[2] * np.cos(x[:, 4]) + legs[3] * (
            x[:, 10] + x[:, 11]) * np.cos(x[:, 4] + x[:, 5])
    df[:, 31, 5] = legs[3] * (x[:, 10] + x[:, 11]) * np.cos(x[:, 4] + x[:, 5])
    df[:, 31, 7] = 1
    df[:, 31, 10] = legs[2] * np.sin(x[:, 4]) + legs[3] * np.sin(x[:, 4] + x[:, 5])
    df[:, 31, 11] = legs[3] * np.sin(x[:, 4] + x[:, 5])
    df[:, 33, 4] = -legs[2] * (
            x[:, 16] * np.sin(x[:, 4]) + x[:, 10] ** 2 * np.cos(x[:, 4])) - legs[
                       3] * (
                           x[:, 16] + x[:, 17]) * np.sin(
        x[:, 4] + x[:, 5]) - legs[3] * (x[:, 10] + x[:, 11]) ** 2 * np.cos(
        x[:, 4] + x[:, 5])
    df[:, 33, 5] = -legs[3] * (x[:, 16] + x[:, 17]) * np.sin(x[:, 4] + x[:, 5]) - legs[
        3] * (
                           x[:, 10] + x[:, 11]) ** 2 * np.cos(x[:, 4] + x[:, 5])
    df[:, 33, 10] = -2 * legs[2] * x[:, 10] * np.sin(x[:, 4]) - 2 * legs[3] * (
            x[:, 10] + x[:, 11]) * np.sin(
        x[:, 4] + x[:, 5])
    df[:, 33, 11] = -2 * legs[3] * (x[:, 10] + x[:, 11]) * np.sin(x[:, 4] + x[:, 5])
    df[:, 33, 12] = 1
    df[:, 33, 16] = legs[2] * np.cos(x[:, 4]) + legs[3] * np.cos(x[:, 4] + x[:, 5])
    df[:, 33, 17] = legs[3] * np.cos(x[:, 4] + x[:, 5])
    df[:, 34, 4] = -legs[2] * (
            -x[:, 16] * np.cos(x[:, 4]) + x[:, 10] ** 2 * np.sin(x[:, 4])) + legs[
                       3] * (
                           x[:, 16] + x[:, 17]) * np.cos(
        x[:, 4] + x[:, 5]) - legs[3] * (x[:, 10] + x[:, 11]) ** 2 * np.sin(
        x[:, 4] + x[:, 5])
    df[:, 34, 5] = legs[3] * (x[:, 16] + x[:, 17]) * np.cos(x[:, 4] + x[:, 5]) - legs[
        3] * (
                           x[:, 10] + x[:, 11]) ** 2 * np.sin(x[:, 4] + x[:, 5])
    df[:, 34, 10] = 2 * legs[2] * x[:, 10] * np.cos(x[:, 4]) + 2 * legs[3] * (
            x[:, 10] + x[:, 11]) * np.cos(
        x[:, 4] + x[:, 5])
    df[:, 34, 11] = 2 * legs[3] * (x[:, 10] + x[:, 11]) * np.cos(x[:, 4] + x[:, 5])
    df[:, 34, 13] = 1
    df[:, 34, 16] = legs[2] * np.sin(x[:, 4]) + legs[3] * np.sin(x[:, 4] + x[:, 5])
    df[:, 34, 17] = legs[3] * np.sin(x[:, 4] + x[:, 5])

    df = np.matmul(R, df)

    if dim_observations == 20:
        df = df[:, (0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31,
                    33, 34), :]
    return df


def state_to_obs_linear(x, xp, dim_states, dim_observations, g, legs, imus, R):
    """
    Compute linearised state-to-observation transition. For a given state x and is
    predecessor xp, the observation for state x is approximated by
    $ h(x) \approx h(xp) + Dh(xp) \times (x-xp) $.
    :param x: np.ndarray
        N x 18-dimensional state vector where N is the number of particles and the 18
        dimension composed of $ (x_H, y_H, phi_0, phi_1, phi_2, phi_3) $ and the
        corresponding first and second (temporal) derivatives
    :param xp: np.ndarray
        N x 18-dimensional state vector where N is the number of particles and the 18
        dimension composed of $ (x_H, y_H, phi_0, phi_1, phi_2, phi_3) $ and the
        corresponding first and second (temporal) derivatives
    :param dim_states: int
        Dimension of state vector, should be 18
    :param dim_observations: int
        Dimension of observation vector, should be either 20 or 36
    :param g: float
            gravitational constant $ g = 9.81m/s^{2} $
    :param legs: np.ndarray
            4-dimensional array, containing the length of left femur, left fibula,
            right femur, right fibula
    :param imus: np.ndarray
            4-dimensional array, containing the position of the four imus, measured from
            the hip or the knees, respectively
    :param R: np.ndarray
            Rotation matrix of shape (36, 36)
    :return: np.ndarray
        N x dim_observation-dimensional observation vector
    """
    if xp is None:
        xp = np.zeros(x.shape)
    nb_steps, _ = x.shape
    df = compute_jacobian_obs(xp, dim_states, dim_observations, g, legs, imus, R)
    y = state_to_obs(xp, dim_observations, g, legs, imus, R) + np.einsum(
        'ijk, ik -> ij', df, x - xp)
    return y


class MechanicalModel:
    def __init__(self, dim_states, dim_observations, imu_positions, leg_constants, R):
        self.dim_states = dim_states
        self.dim_observations = dim_observations
        self.g = 9.81
        self.imu_pos = imu_positions
        self.leg_len = leg_constants
        self.R = R

    def state_to_observation(self, x):
        return state_to_obs(x, self.dim_observations, self.g, self.leg_len,
                            self.imu_pos, self.R)

    def compute_jacobian_observation(self, x):
        return compute_jacobian_obs(x, self.dim_states, self.dim_observations, self.g,
                                    self.leg_len, self.imu_pos, self.R)

    def compute_jacobian_observation_numeric(self, x, delta_x=0.01):
        df = np.empty((self.dim_observations, self.dim_states))
        x_plus = np.empty_like(x)
        x_minus = np.empty_like(x)
        for j in range(self.dim_states):
            x_plus[:, :] = x_minus[:, :] = x
            x_plus[:, j] += delta_x
            x_minus[:, j] -= delta_x
            delta_f = self.state_to_observation(x_plus) - self.state_to_observation(
                x_minus)
            df[:, j] = delta_f / (2 * delta_x)
        return df

    def state_to_observation_linear(self, x, xp):
        return state_to_obs_linear(x, xp, self.dim_states, self.dim_observations,
                                   self.g, self.leg_len, self.imu_pos, self.R)
