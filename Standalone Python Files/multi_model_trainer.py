"""
Unified Multi-Model Trainer — Δz Regression on PCA Latent Transitions
---------------------------------------------------------------------
Compares optimized MLP (static mapping) and LSTM (temporal mapping)
for predicting continuous EEG latent transitions (Δz).

Author: Nethmi Malsha Ranathunga
Date: October 2025
"""
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
#load and merge data
latent_df = pd.read_csv("latent_factors_PCA.csv")
df = pd.read_csv("latent_transition_pairs_PCA.csv")

merged = latent_df.merge(df, on=["subject_id", "label"], suffixes=("_z", "_dz"))
merged = resample(merged, replace=True, n_samples=300, random_state=42)

# Feature Engineering

phase_onehot = pd.get_dummies(merged["label"], prefix="phase")

X = np.hstack([
    merged[[f"z{i}" for i in range(1, 16)]].values,
    phase_onehot.values
])
y = merged[[f"Δz{i}" for i in range(1, 16)]].values * 10  # scaled targets

# Standardize
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# Data Augmentation (adds Gaussian noise)

noise = np.random.normal(0, 0.02, X_train.shape)
X_train_aug = np.vstack([X_train, X_train + noise])
y_train_aug = np.vstack([y_train, y_train])
print(f"Augmented training size → {X_train_aug.shape}")

# Helper Function for Evaluation

def evaluate_model(name, y_true, y_pred):
    y_true = scaler_y.inverse_transform(y_true) / 10
    y_pred = scaler_y.inverse_transform(y_pred) / 10
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    ev = explained_variance_score(y_true, y_pred)
    print(f"\n{name} → R²={r2:.4f} | MSE={mse:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} | EV={ev:.4f}")
    return dict(Model=name, R2=r2, MSE=mse, RMSE=rmse, MAE=mae, EV=ev)

results = []
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# MLP Model (Optimized)

print("\n=== Training Optimized MLP Model ===")
mlp = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.15),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1])
])
mlp.compile(optimizer=Adam(1e-3), loss=Huber(), metrics=['mae'])
mlp.fit(X_train_aug, y_train_aug, validation_data=(X_val, y_val),
        epochs=250, batch_size=8, callbacks=[early_stop], verbose=1)

mlp_pred = mlp.predict(X_test)
results.append(evaluate_model("MLP", y_test, mlp_pred))


# LSTM Model (for completeness)

print("\n=== Training LSTM Model ===")
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

lstm = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1])
])
lstm.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
lstm.fit(X_train_lstm, y_train, validation_data=(X_val_lstm, y_val),
         epochs=150, batch_size=8, callbacks=[early_stop], verbose=1)

lstm_pred = lstm.predict(X_test_lstm)
results.append(evaluate_model("LSTM", y_test, lstm_pred))

# Save Results & Compare

results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison_latent_transitions_final.csv", index=False)
print("\nSaved metrics → model_comparison_latent_transitions_final.csv")
print(results_df)


# Visualization (Predicted vs True Δz₁)

y_true = scaler_y.inverse_transform(y_test) / 10
y_pred_rescaled = scaler_y.inverse_transform(mlp_pred) / 10

plt.figure(figsize=(6,6))
# MLP (already done)
plt.scatter(y_true[:,0], y_pred_rescaled[:,0], alpha=0.7, color='blue', label='MLP')

# LSTM
lstm_pred_rescaled = scaler_y.inverse_transform(lstm_pred) / 10
plt.scatter(y_true[:,0], lstm_pred_rescaled[:,0], alpha=0.7, color='orange', label='LSTM')

# Ideal line
plt.plot([y_true[:,0].min(), y_true[:,0].max()],
         [y_true[:,0].min(), y_true[:,0].max()],
         'r--', lw=2, label='Ideal Fit')

plt.title("Predicted vs True Δz₁ — MLP vs LSTM Comparison")
plt.xlabel("True Δz₁")
plt.ylabel("Predicted Δz₁")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f"model_comparison_latent_transitions_final.png")

# Save Model & Final Metrics

mlp.save("MLP_PCA_optimized_model.keras")
with open("MLP_PCA_results.txt", "w") as f:
    f.write(f"R²={results[0]['R2']:.4f}, MSE={results[0]['MSE']:.4f}, RMSE={results[0]['RMSE']:.4f}, "
            f"MAE={results[0]['MAE']:.4f}, EV={results[0]['EV']:.4f}\n")
print("\n Saved model → MLP_PCA_optimized_model.keras")

joblib.dump((scaler_X, scaler_y), "scalers.pkl")
print("Saved scalers → scalers.pkl")
