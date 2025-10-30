from stable_baselines3 import PPO
from closed_loop_enviornment import EEGClosedLoopEnv
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load MLP model and scalers
mlp = load_model("MLP_PCA_optimized_model.keras")
scaler_X, scaler_y = joblib.load("scalers.pkl")

z_target = np.zeros(15)

# Initialize environment
env = EEGClosedLoopEnv(mlp, scaler_X, scaler_y, z_target)

# Define RL agent (PPO = Proximal Policy Optimization)
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=25000)

# Save trained model
model.save("EEG_RL_Controller")
print(" RL controller training complete and saved!")
