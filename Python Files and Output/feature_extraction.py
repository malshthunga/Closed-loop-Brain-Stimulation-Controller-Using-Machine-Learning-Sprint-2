"""
Feature Extraction Script (Optimized)
-------------------------------------
This version reduces dataset size by downsampling and summarizing time-series
features while preserving essential stimulation-related information.
"""

import os
import glob
import numpy as np
import pandas as pd
from feature_functions import get_feature_matrices


# Configuration
selected_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz']
Bands = ["delta", "theta", "alpha", "beta"]
Ratios = ["theta_alpha_ratio", "alpha_beta_ratio"]
base_features = [f"{ch}_{b}" for ch in selected_channels for b in Bands + Ratios]

# Downsampling factor — adjust this if needed
DOWNSAMPLE_FACTOR = 1000  # average every 1000 samples (~1s at fs=1000 Hz)


def downsample_features(matrix, factor=DOWNSAMPLE_FACTOR):
    """Downsample EEG feature matrix by averaging over windows of given factor."""
    if matrix.shape[1] < factor:
        return matrix  # too short to downsample
    n_windows = matrix.shape[1] // factor
    reshaped = matrix[:, :n_windows * factor].reshape(matrix.shape[0], n_windows, factor)
    return reshaped.mean(axis=2)  # mean per window


def main():
    #Find all .mat files
    data_dir = os.getcwd()
    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    if not file_paths:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")

    print(f"Found {len(file_paths)} .mat files:")
    for f in file_paths:
        print("  ", os.path.basename(f))

    all_subjects = []

    for subj_i, file_path in enumerate(file_paths, start=1):
        subject_id = os.path.splitext(os.path.basename(file_path))[0]
        file_tag = subject_id
        print("\n" + "=" * 75)
        print(f" Processing EEG file: {os.path.basename(file_path)}")
        print("=" * 75)

        fm = get_feature_matrices(file_path)

        # Keep only frontal channels
        if fm:
            for cond in fm.keys():
                mat_ = fm[cond]
                if mat_.shape[0] >= 35:
                    fm[cond] = mat_[:5, :]  # take first 5 channels Fp1–Fz

        print("\nFeature matrix summary:")
        for cond, mat_ in fm.items():
            print(f"  {cond}: {mat_.shape}")

        #Combine all conditions
        all_features, all_labels = [], []
        for label, cond in enumerate(["Pre-stimulation", "During-stimulation", "Post-stimulation"]):
            feats = fm.get(cond)
            if feats is not None and feats.size > 0:
                feats_ds = downsample_features(feats)
                all_features.append(feats_ds)
                all_labels.extend([label] * feats_ds.shape[0])

        if not all_features:
            print(f" No features extracted for {file_path}. Skipping.")
            continue

        # Align lengths before stacking
        min_len = min(feat.shape[1] for feat in all_features)
        all_features = [feat[:, :min_len] for feat in all_features]

        X_all = np.vstack(all_features)
        y_all = np.array(all_labels)

        #Compute activation (Δ = Post - Pre)
        pre_feats = fm.get("Pre-stimulation")
        post_feats = fm.get("Post-stimulation")

        if pre_feats is not None and post_feats is not None:
            n = min(pre_feats.shape[1], post_feats.shape[1])
            delta_feats = post_feats[:, :n] - pre_feats[:, :n]
            mean_activation = delta_feats.mean(axis=1)
            activation_map = dict(zip(selected_channels, mean_activation))
            most_active = max(activation_map, key=lambda ch: abs(activation_map[ch]))
            print("\nAverage Δactivation per channel:")
            for ch, val in activation_map.items():
                direction = "↑ excitation" if val > 0 else "↓ inhibition"
                print(f"  {ch}: {val:+.4f} ({direction})")
            print(f" Region showing largest stimulation effect: {most_active}")

        #Build and save compact CSV
        n_cols = X_all.shape[1]
        feature_names = [f"feature_{i + 1}" for i in range(n_cols)]
        df = pd.DataFrame(X_all, columns=feature_names)
        df["label"] = y_all
        df["subject_id"] = subject_id
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Save each subject
        save_path = f"{file_tag}_EEG_bandpower_features.csv"
        df.to_csv(save_path, index=False)
        all_subjects.append(df)
        print(f" Saved downsampled bandpower features: {save_path} ({df.shape})")

    # Combine all subjects
    combined_df = pd.concat(all_subjects, ignore_index=True)
    combined_df.to_csv("EEG_all_subjects_features.csv", index=False)
    print("\n Feature extraction completed for all subjects.")
    print(f"Combined dataset saved → EEG_all_subjects_features.csv (shape: {combined_df.shape})")


if __name__ == "__main__":
    print("=== Running Optimized Feature Extraction Stage ===")
    main()
