from stable_baselines3 import PPO
from closed_loop_enviornment import EEGClosedLoopEnv
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load environment and trained RL model
mlp = load_model("MLP_PCA_optimized_model.keras")

scaler_X, scaler_y = joblib.load("scalers.pkl")
z_target = np.zeros(15)

env = EEGClosedLoopEnv(mlp, scaler_X, scaler_y, z_target)
model = PPO.load("EEG_RL_Controller")

# Evaluate performance
obs, _ = env.reset()

rewards = []
for t in range(200):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    rewards.append(reward)
    print(f"Cycle {t} | Reward: {reward:.3f}")
    if done:
        print(f" Converged at cycle {t}")
        break

print(f"\nAverage reward: {np.mean(rewards):.3f}")

