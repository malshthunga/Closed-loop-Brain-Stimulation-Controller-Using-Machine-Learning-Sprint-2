import gymnasium
from gymnasium import spaces
import numpy as np

class EEGClosedLoopEnv(gymnasium.Env):
    def __init__(self, mlp_model, scaler_X, scaler_y, z_target):
        super(EEGClosedLoopEnv, self).__init__()

        self.mlp = mlp_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.z_target = z_target
        self.n_features = len(z_target)

        # Define action and observation space
        # Actions: Δcurrent, Δfreq
        self.action_space = spaces.Box(low=np.array([-0.05, -0.5]),
                                       high=np.array([0.05, 0.5]), dtype=np.float32)
        # Observations: current z + stim params
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(self.n_features + 2,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.z_current = np.random.uniform(-0.2, 0.2, 15)
        self.stim_params = np.array([1.0, 10.0])
        self.steps = 0

        obs = np.concatenate([self.z_current, self.stim_params])
        return obs, {}

    def step(self, action):
        # Update stimulation parameters
        self.stim_params += action
        self.stim_params = np.clip(self.stim_params, self.action_space.low, self.action_space.high)

        # Predict next latent state using your trained MLP
        stim_onehot = np.zeros(2)
        stim_onehot[0] = 1
        X_input = np.hstack([self.z_current, stim_onehot]).reshape(1, -1)
        X_scaled = self.scaler_X.transform(X_input)
        dz_scaled = self.mlp.predict(X_scaled)
        dz = np.clip(self.scaler_y.inverse_transform(dz_scaled) * 0.005, -0.03, 0.03)

        self.z_current = np.clip(self.z_current + dz.flatten(), -3, 3)

        # Compute error and reward
        error = np.linalg.norm(self.z_target - self.z_current)
        reward = -error  # RL learns to minimize this

        self.steps += 1
        terminated = error < 0.2
        truncated = self.steps >= 200

        obs = np.concatenate([self.z_current, self.stim_params])
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass
