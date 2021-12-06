import numpy as np

from MechanicalModel import MechanicalModel
from ReadData import DataReaderWriter
from Plotting import Plotter


class TwoLegEKF:
    def __init__(self, model, x0, Q, H, numeric_jacobian):
        """EKF(model, x0, Q, H)
        model has state transition function f and state to observation function h
        x' = f(x, u)    state transition function
        z' = h(x)       observation function
        x0              initial state estimate"""
        self.f, self.h = model.state_transition, model.state_to_observation
        self.A = model.A
        self.x = np.reshape(x0, (1, -1))
        self.I, self.P = np.eye(len(x0)), np.eye(len(x0))
        # modify these if the noise is not independent
        self.Q, self.H = Q, H
        self.dim_states = Q.shape[0]
        self.dim_observations = H.shape[0]
        if numeric_jacobian:
            self.compute_jacobian = model.compute_jacobian_observation_numeric
        else:
            self.compute_jacobian = model.compute_jacobian_observation

    def predict(self):
        F = self.A
        self.x = self.f(self.x)
        self.P = (F @ self.P @ F.T) + self.Q

    def update(self, z):
        x, P, H, I, h = self.x, self.P, self.H, self.I, self.h
        df = self.compute_jacobian(x)
        df = np.reshape(df, (self.dim_observations, self.dim_states))
        y = z - h(x)
        S = (df @ P @ df.T) + H
        K = P @ df.T @ np.linalg.inv(S)
        self.x += (K @ y.T).T
        self.P = (I - K @ df) @ P


def generate_process_covar(dt, sx, sy, sphi):
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
        print('Q is positive definite.')
        return Q
    except np.linalg.LinAlgError:
        print('Q is NOT positive definite.')
        return None


def generate_observation_covar(s_imu_acc, s_imu_gyro, s_press_velo, s_press_acc, dim_observations):
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
imu_position = np.array([0.34, 0.29, 0.315, 0.33])
DIM_STATES = 18
DIM_OBSERVATIONS = 20

my_model = MechanicalModel(dt=dt,
                           dim_states=DIM_STATES,
                           dim_observations=DIM_OBSERVATIONS,
                           imu_position=imu_position,
                           leg_constants=leg_constants,
                           R=np.eye(36)
                           )

# -------- Data -----------
path_truth = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Normal/truth_normal.dat'
path_obs = '/home/maurus/Pycharm_Projects/TwoLegModelSMC/GeneratedData/Normal/noised_observations_normal.dat'
data_reader = DataReaderWriter()
max_timesteps = 1000
data_reader.read_states_as_arr(path_truth, max_timesteps=max_timesteps)
data_reader.read_observations_as_arr(path_obs, max_timesteps=max_timesteps)
data_reader.prepare_lists()
true_states = data_reader.true_states
obs = data_reader.observations
true_states = np.reshape(true_states, (max_timesteps, 1, DIM_STATES))
if DIM_OBSERVATIONS == 20:
    obs = obs[:, (0, 1, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 27, 28, 30, 31, 33, 34)]
obs = np.reshape(obs, (max_timesteps, 1, DIM_OBSERVATIONS))

# -------- EKF -----------
a = np.array([0.01, 1.06, -0.13, -0.25, 0.37, -0.19,
              0.57, 0.10, 2.54, -3.8, -0.08, -0.82,
              -0.00, 0.01, -1.78, 3.32, -0.30, 0.54])
# dt already defined for model
scale_x = 100.01
scale_y = 100.0
scale_phi = 250.0
Q = generate_process_covar(dt=dt, sx=scale_x, sy=scale_y, sphi=scale_phi)
sigma_imu_acc = 0.1
sigma_imu_gyro = 0.01
sigma_press_velo = 0.1
sigma_press_acc = 1000.0
H = generate_observation_covar(s_imu_acc=sigma_imu_acc, s_imu_gyro=sigma_imu_gyro, s_press_velo=sigma_press_velo,
                               s_press_acc=sigma_press_acc, dim_observations=DIM_OBSERVATIONS)

my_ekf = TwoLegEKF(model=my_model, x0=a, Q=Q, H=H, numeric_jacobian=False)

# -------- Simulation -----------
x_vals = []
y_vals = []
for t in range(0, max_timesteps):
    my_ekf.predict()
    x_vals.append(my_ekf.x)
    my_ekf.update(obs[t])

# -------- Plotting -----------
x_vals = np.reshape(x_vals, (max_timesteps, 1, DIM_STATES))
my_plotter = Plotter(true_states=true_states, true_obs=obs, delta_t=dt)
my_plotter.plot_smoothed_trajectories(samples=x_vals, export_name='ekf_analytic')
