"""
CTT Behavioural and EEG Data Visualisation
------------------------------------------
Loads all EEG `.mat` files in the current directory, extracts
EEG and behavioural (CTT) data, ensures channel consistency,
and plots the first 10 seconds of behavioural performance with
stimulation markers.

Outputs:
- Prints dataset details (channels, sampling rates, etc.)
- Saves behavioural plots: CTT_Performance_<subject>.png
- Exports behavioural summary: CTT_behavioral_metrics.csv
"""

import h5py
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import plotly.graph_objects as go
from helper_func import load_eeg_dataset
import numpy as np
import os
import glob

# Locate and load EEG data

data_dir = os.getcwd()
pattern = os.path.join(data_dir, "*.mat")
file_paths = sorted(glob.glob(pattern))
if not file_paths:
    raise FileNotFoundError(f"No .mat files found in {data_dir}")

datasets = {}
all_labels = []

for fp in file_paths:
    eeg_data, labels, fs, triggers = load_eeg_dataset(fp)
    datasets[fp] = {"eeg": eeg_data, "labels": labels, "fs": fs, "triggers": triggers}
    all_labels.append(labels)
    print(f"Loaded {os.path.basename(fp)}: shape={eeg_data.shape}, fs={fs}, channels={len(labels)}")

# Check label consistency across files

unique_label_sets = [tuple(lbls) for lbls in all_labels]
reference_labels = unique_label_sets[0]
inconsistent_files = [
    os.path.basename(file_paths[i])
    for i, lbls in enumerate(unique_label_sets)
    if lbls != reference_labels
]

if inconsistent_files:
    print("\n Inconsistent channel labels found in these files:")
    for name in inconsistent_files:
        print("   ", name)
else:
    print("\n All files have consistent EEG channel labels.")

# Select relevant EEG channels

selected_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz']
chan_indices = [reference_labels.index(ch) for ch in selected_channels if ch in reference_labels]

print(f"\nSelected channels: {', '.join([reference_labels[i] for i in chan_indices])}")
eeg_data_selected = eeg_data[chan_indices, :]
print(f"Selected EEG shape: {eeg_data_selected.shape}  (channels × samples)")
print(f"Sampling rate (fs): {fs} Hz")

# Extract behavioural data

metrics_list = []

for fp in file_paths:
    print(f"\nProcessing behavioural data from: {os.path.basename(fp)}")
    try:
        with h5py.File(fp, "r") as f:
            if "DSamp" in f and "ptrackerPerf" in f["DSamp"]:
                CTT_data = np.squeeze(np.array(f["DSamp"]["ptrackerPerf"]))
                fs_ctt = int(np.array(f["DSamp"]["ptrackerfs"]).squeeze())
                print(f" Found behavioural data: DSamp/ptrackerPerf ({CTT_data.shape[0]} samples)")
            elif "DSamp" in f and "pTrackerPerf" in f["DSamp"]:
                CTT_data = np.squeeze(np.array(f["DSamp"]["pTrackerPerf"]))
                fs_ctt = int(np.array(f["DSamp"]["ptrackerfs"]).squeeze())
                print(f" Found behavioural data: DSamp/pTrackerPerf ({CTT_data.shape[0]} samples)")
            else:
                print(f" No behavioural data fields found in {fp}")
                continue
    except Exception as e:
        print(f" Failed to read behavioural data from {fp}: {e}")
        continue

    # Plot behavioural data (first 10 seconds)
    start_time, end_time = 0, 10
    start_indx, end_indx = int(fs_ctt * start_time), int(fs_ctt * end_time)
    time_in = np.arange(start_indx, end_indx) / fs_ctt
    dat_in = CTT_data[start_indx:end_indx]



    fig = go.Figure()
    fig.add_scatter(x=time_in, y=dat_in, mode='lines', name="CTT performance")

    # Add stimulation markers if available
    triggers_df = datasets[fp]["triggers"]
    if not triggers_df.empty:
        stim_times = triggers_df[triggers_df["event_name"] == "Stim Start"]["time_s"].values
        for t in stim_times:
            fig.add_vline(x=t, line=dict(color="red", dash="dash"), annotation_text="Stim Start")
    # Normalise EEG trigger times to match behavioural duration
    stim_times = triggers_df[triggers_df["event_name"] == "Stim Start"]["time_s"].values
    stim_times = stim_times[stim_times < len(CTT_data) / fs_ctt]  # only keep those within behavioural duration

    fig.update_layout(
        title=f"CTT Data (First 10 Seconds) - {os.path.basename(fp)}",
        xaxis_title="Time (sec)",
        yaxis_title="Performance",
        legend_title="Signal Type"
    )
    output_name = f"CTT_Performance_{os.path.basename(fp).replace('.mat', '')}.png"
    fig.write_image(output_name, width=1200, height=700)
    print(f" Saved plot → {output_name}")

    # Compute behavioural summary metrics

    metrics = {
        "subject_file": os.path.basename(fp),
        "mean_perf": np.mean(CTT_data),
        "std_perf": np.std(CTT_data),
        "max_perf": np.max(CTT_data),
        "min_perf": np.min(CTT_data),
        "performance_range": np.ptp(CTT_data),
        "sample_rate": fs_ctt,
    }
    metrics_list.append(metrics)

# Save metrics summary

metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv("CTT_behavioral_metrics.csv", index=False)
print("\Saved CTT_behavioral_metrics.csv with behavioural summary features.")
