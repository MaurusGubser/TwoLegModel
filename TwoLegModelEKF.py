import itertools

import numpy as np

from MechanicalModel import MechanicalModel
from DataReaderWriter import DataReaderWriter
from Plotter import Plotter


class TwoLegEKF:
    def __init__(self, model, x0, Q, S, numeric_jacobian):
        self.a, self.h = model.state_transition, model.state_to_observation
        self.A = model.A
        self.x = np.reshape(x0, (1, -1))
        self.I, self.P = np.eye(self.x.shape[1]), np.eye(self.x.shape[1])
        # modify these if the noise is not independent
        self.Q, self.H = Q, S
        self.dim_states = Q.shape[0]
        self.dim_observations = S.shape[0]
        if numeric_jacobian:
            self.compute_jacobian = model.compute_jacobian_observation_numeric
        else:
            self.compute_jacobian = model.compute_jacobian_observation

    def predict(self):
        F = self.A
        self.x = self.a(self.x)
        self.P = (F @ self.P @ F.T) + self.Q

    def update(self, data_t):
        x, P, H, I, h = self.x, self.P, self.H, self.I, self.h
        df = self.compute_jacobian(x)
        df = np.reshape(df, (self.dim_observations, self.dim_states))
        y = data_t - h(x)
        U = (df @ P @ df.T) + H
        K = P @ df.T @ np.linalg.inv(U)
        self.x += (K @ y.T).T
        self.P = (I - K @ df) @ P


def set_process_covariance(dim_states, dt, lambda_x, lambda_y, lambda_phi):
    Q = np.zeros((dim_states, dim_states))
    block_size = dim_states // 3
    for row in range(0, dim_states):
        for col in range(0, dim_states):
            if row < block_size:
                if row == col:
                    Q[row, col] = dt ** 5 / 20.0
                elif row + 6 == col:
                    Q[row, col] = dt ** 4 / 8.0
                    Q[col, row] = dt ** 4 / 8.0
                elif row + 12 == col:
                    Q[row, col] = dt ** 3 / 6.0
                    Q[col, row] = dt ** 3 / 6.0
            elif block_size <= row < 2 * block_size:
                if row == col:
                    Q[row, col] = dt ** 3 / 3.0
                elif row + 6 == col:
                    Q[row, col] = dt ** 2 / 2.0
                    Q[col, row] = dt ** 2 / 2.0
            elif 2 * block_size <= row:
                if row == col:
                    Q[row, col] = dt
    idx_groups = [[0, 6, 12], [1, 7, 13], [2, 8, 14], [3, 9, 15], [4, 10, 16], [5, 11, 17]]
    scale_factors = [lambda_x, lambda_y, lambda_phi, lambda_phi, lambda_phi, lambda_phi]
    for factor, idxs in zip(scale_factors, idx_groups):
        for row, col in itertools.product(idxs, idxs):
            Q[row, col] *= factor
    try:
        np.linalg.cholesky(Q)
        print('Q is positive definite.')
        return Q
    except np.linalg.LinAlgError:
        print('Q is NOT positive definite.')
        return None


def set_observation_covar(dim_observations, s_imu_acc, s_imu_gyro, s_press_velo, s_press_acc):
    if dim_observations == 20:
        h = [s_imu_acc, s_imu_acc, s_imu_gyro,
             s_imu_acc, s_imu_acc, s_imu_gyro,
             s_imu_acc, s_imu_acc, s_imu_gyro,
             s_imu_acc, s_imu_acc, s_imu_gyro,
             s_press_velo, s_press_velo, s_press_acc, s_press_acc,
             s_press_velo, s_press_velo, s_press_acc, s_press_acc]
    else:
        h = [s_imu_acc, s_imu_acc, s_imu_acc, s_imu_gyro, s_imu_gyro, s_imu_gyro,
             s_imu_acc, s_imu_acc, s_imu_acc, s_imu_gyro, s_imu_gyro, s_imu_gyro,
             s_imu_acc, s_imu_acc, s_imu_acc, s_imu_gyro, s_imu_gyro, s_imu_gyro,
             s_imu_acc, s_imu_acc, s_imu_acc, s_imu_gyro, s_imu_gyro, s_imu_gyro,
             s_press_velo, s_press_velo, s_press_velo, s_press_acc, s_press_acc, s_press_acc,
             s_press_velo, s_press_velo, s_press_velo, s_press_acc, s_press_acc, s_press_acc]
    return np.diag(h)


# -------- Model -----------
dt = 0.01
leg_constants = np.array([0.5, 0.6, 0.5, 0.6])
imu_positions = np.array([0.34, 0.29, 0.315, 0.33])
dim_states = 18
dim_observations = 20
R = np.eye(36)

model = MechanicalModel(dt=dt,
                        dim_states=dim_states,
                        dim_observations=dim_observations,
                        imu_positions=imu_positions,
                        leg_constants=leg_constants,
                        R=R
                        )

# -------- Data -----------
### EKF cannot handle missing data so far! ###
generation_type = 'Normal'
nb_timesteps = 1000
dim_obs = 20
data_reader = DataReaderWriter()
true_states, obs = data_reader.get_data_as_lists(generation_type, nb_timesteps, dim_obs)

# -------- EKF -----------
b0 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# dt already defined for model
lambda_x = 10000.0  # 100.0
lambda_y = 1000.0  # 100.0
lambda_phi = 10000000.0  # 250.0
Q = set_process_covariance(dim_states=dim_states, dt=dt, lambda_x=lambda_x, lambda_y=lambda_y, lambda_phi=lambda_phi)
sigma_imu_acc = 0.1  # 0.1
sigma_imu_gyro = 0.1  # 0.01
sigma_press_velo = 0.1  # 0.1
sigma_press_acc = 1.0  # 1000.0
S = set_observation_covar(dim_observations=dim_observations, s_imu_acc=sigma_imu_acc, s_imu_gyro=sigma_imu_gyro,
                          s_press_velo=sigma_press_velo,
                          s_press_acc=sigma_press_acc)
ekf = TwoLegEKF(model=model, x0=b0, Q=Q, S=S, numeric_jacobian=False)

# -------- run EKF -----------
x_vals = []
y_vals = []
for t in range(0, nb_timesteps):
    ekf.predict()
    x_vals.append(ekf.x)
    ekf.update(obs[t])

# -------- plotting -----------
x_vals = np.reshape(x_vals, (nb_timesteps, 1, dim_states))
plotter = Plotter(true_states=np.array(true_states), true_obs=np.array(obs), delta_t=dt, export_name='ekf_analytic',
                  show_fig=True)
plotter.plot_observations(samples=x_vals, observation_map=model.state_to_observation)
plotter.plot_smoothed_trajectories(samples=x_vals)
