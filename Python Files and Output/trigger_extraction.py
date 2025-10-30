"""
Trigger Event Extraction and Timeline Visualisation
---------------------------------------------------
This script performs the following steps:

1. Scans the working directory for all `.mat` EEG files.
2. Decodes trigger events from each file using the `decode_triggers()` helper function.
3. Combines all triggers into one dataframe.
4. Maps event codes to descriptive labels:
      • 2  → Block Start
      • 16 → Stim Start
      • 32 → Stim Stop
5. Displays event counts and recording duration statistics.
6. Generates a scatter-timeline plot showing when stimulation events occurred.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob
from helper_func import decode_triggers, load_eeg_dataset

#Find all .mat files
data_dir = os.getcwd()
file_paths = sorted(glob.glob(os.path.join(data_dir, '*.mat')))
if not file_paths:
    raise FileNotFoundError(f"No .mat files found in {data_dir}")

print(f"Found {len(file_paths)} .mat files:")
for f in file_paths:
    print("  ", os.path.basename(f))

#Decode triggers from all files
all_triggers = []
for fp in file_paths:
    try:
        df = decode_triggers(fp)
        df["subject_file"] = os.path.basename(fp)  # ← add subject ID for merging later
        all_triggers.append(df)
        print(f"Decoded {len(df)} triggers from {os.path.basename(fp)}")
    except Exception as e:
        print(f" Failed to decode {os.path.basename(fp)} → {e}")


if not all_triggers:
    raise RuntimeError("No trigger data could be decoded!")

# Combine all results
triggers_df = pd.concat(all_triggers, ignore_index=True)
triggers_df.to_csv("trigger_extraction.csv", index=False)
print("\n Combined triggers saved → all_subject_triggers.csv")

#Map codes to readable event names
code_map = {2: "Block Start", 16: "Stim Start", 32: "Stim Stop", 48: "Block Start"}
triggers_df["event_name"] = triggers_df["code"].map(code_map).fillna("Other")

print("\nTrigger Data Sample:")
print(triggers_df.head(10))

print("\nEvent counts across all files:")
print(triggers_df["event_name"].value_counts())

# Check EEG file shape for verification
# (Loads only first file just to confirm dimensions)
try:
    eeg_data, labels, fs, triggers = load_eeg_dataset(file_paths[0])
    print(f"\nExample EEG shape: {eeg_data.shape}, sampling rate: {fs} Hz")
except Exception as e:
    print(f"\n Could not load EEG signal from {file_paths[0]} → {e}")

#Visualize trigger timeline
plt.figure(figsize=(12, 4))
plt.scatter(triggers_df["time_s"], triggers_df["code"], c="royalblue", marker="|", s=200)
plt.yticks([2, 16, 32, 48], ["Block Start", "Stim Start", "Stim Stop", "Other"])
plt.xlabel("Time (s)")
plt.ylabel("Event")
plt.title("EEG Trigger Timeline (All Subjects)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("triggers_timeline.png")
plt.show()
