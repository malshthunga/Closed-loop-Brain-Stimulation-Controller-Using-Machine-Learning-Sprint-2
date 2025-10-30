"""
Latent Space Compression using PCA
----------------------------------
This script:
1. Loads combined EEG feature data (with 'label' and 'subject_id').
2. Cleans missing/constant columns.
3. Scales features per subject to remove inter-subject bias.
4. Applies PCA to reduce dimensionality.
5. Saves the latent representation for MLP or downstream models.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

np.random.seed(42)
random.seed(42)

csv_path = "EEG_all_subjects_features.csv"
n_components = 15
visualize = True

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found!")

print(f"Loading dataset → {csv_path}")
df = pd.read_csv(csv_path)
print("Before cleaning:", df.shape)

# Clean missing/constant columns
df = df.dropna(axis=1, how="all")
nunique = df.nunique()
df = df.drop(columns=nunique[nunique <= 1].index)

if "label" not in df.columns or "subject_id" not in df.columns:
    raise ValueError("Missing 'label' or 'subject_id' column.")

labels = df["label"].astype(int).values
subject_ids = df["subject_id"].values
df = df.drop(columns=["label", "subject_id"])
df = df.apply(lambda x: x.fillna(x.mean()), axis=0)

# Scale features per subject
print("Scaling per subject...")
scaled_parts = []
for subj in np.unique(subject_ids):
    mask = subject_ids == subj
    subj_features = df.loc[mask]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(subj_features)
    scaled_parts.append(pd.DataFrame(scaled, index=subj_features.index, columns=subj_features.columns))
X_scaled = pd.concat(scaled_parts).sort_index()

# PCA
pca = PCA(n_components=n_components, random_state=42)
X_latent = pca.fit_transform(X_scaled)
print(f"Latent shape → {X_latent.shape}")

#Save outputs
latent_cols = [f"z{i+1}" for i in range(X_latent.shape[1])]
latent_df = pd.DataFrame(X_latent, columns=latent_cols)
latent_df["label"] = labels
latent_df["subject_id"] = subject_ids

latent_df.to_csv("latent_factors_PCA.csv", index=False)
print("Saved latent_factors_PCA.csv")

with open("latent_model_PCA.pkl", "wb") as f:
    pickle.dump(pca, f)
print("Saved PCA model → latent_model_PCA.pkl")

# t-SNE visualization
if visualize:
    print("Running t-SNE visualization...")
    tsne = TSNE(n_components=2, perplexity=20, random_state=42)
    latent_2d = tsne.fit_transform(X_latent)
    plt.figure(figsize=(7,6))
    for stim, color, name in zip([0,1,2], ["blue","orange","green"], ["Pre","During","Post"]):
        mask = labels == stim
        plt.scatter(latent_2d[mask,0], latent_2d[mask,1], c=color, label=name, alpha=0.6, s=40)
    plt.title("t-SNE of Latent EEG States (PCA Components)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("latent_space_PCA.png")
    plt.show()