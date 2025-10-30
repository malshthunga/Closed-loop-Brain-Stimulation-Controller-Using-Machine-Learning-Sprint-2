"""
Latent State Transition Computation
-----------------------------------
This script computes Δz vectors between EEG stimulation states (Pre, During, Post)
for each subject based on PCA latent representations.

Outputs:
- latent_transition_pairs.csv
"""

import pandas as pd
import numpy as np
import os

csv_path = "latent_factors_PCA.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found!")

df = pd.read_csv(csv_path)
print(f"Loaded latent dataset → {df.shape}")

# Ensure correct ordering
df = df.sort_values(by=["subject_id", "label"]).reset_index(drop=True)

# Extract unique subject IDs and latent feature names
latent_cols = [c for c in df.columns if c.startswith("z")]
subjects = df["subject_id"].unique()

pairs = []
for subj in subjects:
    subj_data = df[df["subject_id"] == subj]
    if set(subj_data["label"]) >= {0, 2}:  # has both Pre and Post
        pre = subj_data[subj_data["label"] == 0][latent_cols].mean().values
        post = subj_data[subj_data["label"] == 2][latent_cols].mean().values
        delta_post = post - pre
        pairs.append({
            "subject_id": subj,
            "label": 2,
            **{f"Δz{i+1}": v for i, v in enumerate(delta_post)}
        })
    if set(subj_data["label"]) >= {0, 1}:  # has both Pre and During
        pre = subj_data[subj_data["label"] == 0][latent_cols].mean().values
        dur = subj_data[subj_data["label"] == 1][latent_cols].mean().values
        delta_dur = dur - pre
        pairs.append({
            "subject_id": subj,
            "label": 1,
            **{f"Δz{i+1}": v for i, v in enumerate(delta_dur)}
        })

# Save transitions
transitions_df = pd.DataFrame(pairs)
transitions_df.to_csv("latent_transition_pairs_PCA.csv", index=False)
print("Saved latent_transition_pairs_PCA.csv")