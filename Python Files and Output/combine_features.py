"""
Combine EEG Features
--------------------
Loads all *_EEG_bandpower_features.csv files, trims to top 5000
high-variance columns per file, and merges into a single
'combined_features.csv' dataset for later Factor Analysis.
"""

import os
import glob
import numpy as np
import pandas as pd

MAX_FEATURES = 5000
CHUNK_SIZE = 50000
OUT_PATH = "combined_features.csv"

def trim_by_variance(X):
    """Select top 5000 high-variance features efficiently."""
    n_features = X.shape[1]
    X = X.astype(np.float32, copy=False)
    variances = np.zeros(n_features, dtype=np.float32)
    for start in range(0, n_features, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_features)
        variances[start:end] = np.var(X[:, start:end], axis=0)
        print(f"  Processed {end}/{n_features} features...")
    top_idx = np.argsort(variances)[-MAX_FEATURES:]
    return X[:, top_idx]

def combine_feature_csvs():
    feature_files = glob.glob("*_EEG_bandpower_features.csv")
    print(f"Found {len(feature_files)} feature files.")
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    for csv_path in feature_files:
        print(f"\nProcessing {csv_path} ...")
        df = pd.read_csv(csv_path)

        if not {"label", "subject_id"}.issubset(df.columns):
            raise ValueError(f"{csv_path} missing 'label' or 'subject_id' columns")

        y = df["label"].values
        subj = df["subject_id"].values
        X = df.drop(columns=["label", "subject_id"]).values

        if X.shape[1] > MAX_FEATURES:
            print(f" Selecting top {MAX_FEATURES} of {X.shape[1]} features by variance...")
            X = trim_by_variance(X)
            print(f" Trimmed to {X.shape[1]} features.")

        # Save incrementally to avoid memory overload
        temp_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        temp_df["label"] = y
        temp_df["subject_id"] = subj
        temp_df.to_csv(OUT_PATH, mode="a", index=False, header=not os.path.exists(OUT_PATH))
        print(f"  → Appended {csv_path} to {OUT_PATH}")

    print(f"\n Combined dataset saved as {OUT_PATH}")

if __name__ == "__main__":
    combine_feature_csvs()
