"""
Closed-Loop EEG Controller Simulation (MiSO-Inspired)
------------------------------------------------------
Uses trained MLP Δz predictor to simulate adaptive brain stimulation.

For each iteration:
  - Reads current latent state z_t
  - Predicts Δz (brain change) via MLP
  - Estimates new z_t+1 = z_t + Δz_pred
  - Computes error to target z_target
  - Updates stimulation parameters using ε–greedy + proportional rule
  - Repeats for multiple ε values to observe exploration effects

Author: Nethmi Malsha Ranathunga
Supervisor: Dr Farwa Abbas (SAHMRI)
Date: October 2025
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import load_model
import joblib


# Load trained MLP model + scalers

mlp = load_model("MLP_PCA_optimized_model.keras")
scaler_X, scaler_y = joblib.load("scalers.pkl")  # saved during training

# Hyperparameters

eta = 0.0025        # learning rate (controller)
Kp = 0.25
Ki= 0.02
dz_gain = 0.003# proportional gain
min_curr, max_curr = 0.5, 2.0   # mA limits
min_freq, max_freq = 5, 40      # Hz limits
min_freq, max_freq = 5, 40      # Hz limits
epsilons = [0.05, 0.1, 0.2, 0.3]
cycles = 200                    # number of closed-loop updates


# Target state (e.g., healthy baseline)

z_target = np.zeros(15)         # target latent vector (can use mean baseline)
z_init = np.random.uniform(-0.2, 0.2, 15)   # simulated starting latent state


# Function to run one simulation

def run_closed_loop(epsilon):
    z_current = z_init.copy()
    stim_params = np.array([1.0, 10.0])  # [current (mA), frequency (Hz)]
    history = []

    #controller memory terms

    prev_error = 0
    integral = 0


    for t in range(cycles):
        # One-hot encoding for 'During' phase (assuming 2 labels total)
        stim_onehot = np.zeros(2)
        stim_onehot[0] = 1  # or index that corresponds to "during"
        X_input = np.hstack([z_current, stim_onehot]).reshape(1, -1)
        X_scaled = scaler_X.transform(X_input)

        #predict change in z
        dz_scaled = mlp.predict(X_scaled)
        dz = np.clip(scaler_y.inverse_transform(dz_scaled) * dz_gain, -0.03, 0.03)
        z_next = np.clip(z_current + dz.flatten(), -3, 3)

        # #clip latent values
        # z_next = np.clip(z_next, -2, 2)

        # Compute error & average deviation
        error = z_target - z_next
        err_mean = error.mean()
        err_norm_raw = np.linalg.norm(error)
        err_norm_norm = err_norm_raw / np.sqrt(len(error))
        # Integral & control update
        integral = 0.7 * integral + err_mean

        # ε–greedy stimulation update
        exploration_scale = 0.02* np.exp(-t / 100)  # decays over time
        if random.random() < epsilon:
            stim_params += np.random.uniform(-exploration_scale, exploration_scale, stim_params.shape)
        else:
            # Proportional + integral control
            stim_params -= eta * (Kp * err_mean + Ki * integral)
            #we use minus above cause the control low needs to compute the difference between z target and z next .
            #so if current error state is above target, error becomes negative, and you need to reduce stimulation to bring it down.
        stim_params = 0.95 * stim_params + 0.03 * np.array([1.0, 10.0])
        # Safety bounds
        stim_params[0] = np.clip(stim_params[0], min_curr, max_curr)
        stim_params[1] = np.clip(stim_params[1], min_freq, max_freq)

        # Log current iteration
        history.append({
            "t": t,
            "epsilon": epsilon,
            "error_raw": err_norm_raw,
            "error_norm": err_norm_norm,
            "stim_current": stim_params[0],
            "stim_freq": stim_params[1]
        })
        z_current = z_next
        prev_error = err_mean
    return history


# Run for multiple epsilon values

all_results = []
for eps in epsilons:
    print(f"\nRunning closed-loop with ε={eps}")
    history = run_closed_loop(eps)
    all_results.extend(history)

# Convert to structured arrays for analysis

results_df = pd.DataFrame(all_results)
results_df.to_csv("closed_loop_simulation_results.csv", index=False)
print("\nSaved closed-loop results into closed_loop_simulation_results.csv")


# Visualization

plt.figure(figsize=(10,5))
for eps in epsilons:
    subset = results_df[results_df["epsilon"] == eps]
    plt.plot(subset["t"], subset["error_norm"], label=f"ε={eps}")
plt.title("Error Convergence under Different ε Exploration Rates")
plt.xlabel("Cycle")
plt.ylabel("‖z_target − z_t‖ (Latent State Error)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
for eps in epsilons:
    subset = results_df[results_df["epsilon"] == eps]
    plt.plot(subset["t"], subset["stim_current"], label=f"ε={eps}")
plt.title("Adaptive Stimulation Intensity Adjustment")
plt.xlabel("Cycle")
plt.ylabel("Stimulation Current (mA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Summary statistics
summary = results_df.groupby("epsilon")[["error_raw", "error_norm"]].agg(["mean", "min", "std"])

print("\nClosed-Loop Summary (reduce in error = better control)\n")
print(summary)
