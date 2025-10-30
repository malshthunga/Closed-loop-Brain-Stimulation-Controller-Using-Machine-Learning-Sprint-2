"""
helper_func.py
--------------
Utility functions for EEG dataset handling, preprocessing, and trigger decoding
used across the GX tES EEG Closed-Loop Brain Stimulation project.
"""

from scipy.signal import butter, sosfiltfilt, welch
import pandas as pd
import numpy as np
import h5py

def decode_triggers(file_path):
    """
    Final working trigger decoder for GX EEG DSamp HDF5 (.mat v7.3)
    ----------------------------------------------------------------
    Handles nested object references and ASCII-encoded numeric fields.
    Returns tidy DataFrame [subject_id, label, code, offset, time_s].
    """

    import os, h5py, numpy as np, pandas as pd
    from scipy.io import loadmat

    subject_id = os.path.basename(file_path).replace(".mat", "")
    print(f"Decoding triggers → {subject_id}")

    def ascii_to_str(arr):
        """Convert arrays of uint16 ASCII values (like [[48],[48],[49],[54]]) → '0016'."""
        try:
            flat = np.ravel(arr)
            chars = [chr(int(x)) for x in flat if 32 <= x <= 126]
            return "".join(chars)
        except Exception:
            return ""

    try:
        with h5py.File(file_path, "r") as f:
            trg = f["DSamp/triggers"]

            def deep_deref(ref):
                try:
                    obj = f[ref] if isinstance(ref, h5py.Reference) else ref
                    val = obj[()]
                    # If ASCII-encoded numeric, decode to string
                    if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.integer):
                        decoded = ascii_to_str(val)
                        return decoded
                    if hasattr(val, "dtype") and val.dtype.kind in ("S", "O"):
                        val = [v.decode() if isinstance(v, bytes) else str(v) for v in np.ravel(val)]
                        return val[0] if len(val) == 1 else val
                    if np.ndim(val) == 0:
                        return val.item()
                    return val
                except Exception:
                    return np.nan

            def deref_field(name):
                field = np.ravel(trg[name])
                return [deep_deref(r) for r in field]

            labels  = deref_field("Label")
            codes   = deref_field("code")
            offsets = deref_field("offset")
            times   = deref_field("time")

            # Convert codes to numbers if possible
            clean_codes = []
            for c in codes:
                if isinstance(c, str) and c.isdigit():
                    clean_codes.append(int(c))
                else:
                    try:
                        clean_codes.append(int(float(c)))
                    except Exception:
                        clean_codes.append(np.nan)

            # Convert other fields to float
            def to_float_list(x):
                out = []
                for v in x:
                    try:
                        out.append(float(v))
                    except Exception:
                        out.append(np.nan)
                return out

            offsets = to_float_list(offsets)
            times = to_float_list(times)

            print(f"  → Found {len(clean_codes)} triggers (codes sample: {clean_codes[:5]})")

    except Exception as e:
        print(f" HDF5 read failed for {file_path}: {e}")
        # fallback for legacy MAT v7
        mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        DSamp = mat.get("DSamp", None)
        if DSamp and hasattr(DSamp, "triggers"):
            triggers = DSamp.triggers
            labels = [getattr(t, "Label", "") for t in triggers]
            clean_codes = [getattr(t, "code", np.nan) for t in triggers]
            times = [getattr(t, "time", np.nan) for t in triggers]
            offsets = [getattr(t, "offset", np.nan) for t in triggers]
        else:
            raise RuntimeError(f"Could not find triggers in {file_path}")

    df = pd.DataFrame({
        "subject_id": subject_id,
        "label": [str(l).strip() for l in labels],
        "code": clean_codes,
        "offset": offsets,
        "time_s": times
    }).dropna(subset=["code"]).reset_index(drop=True)

    return df


def load_eeg_dataset(file_path):
    """
    Load MATLAB v7.3 (HDF5) EEG dataset from GX tES EEG project.
    Works with new-format .mat files that contain DSamp/EEGdata, DSamp/label, etc.
    Returns:
        eeg_data (np.ndarray): EEG signals (channels x time)
        labels (list): Channel names (e.g. Fp1, Fp2, F3, ...)
        fs (float): Sampling frequency
        triggers (pd.DataFrame): DataFrame of trigger events [code, label, time_s, sample_index, event_name]
    """
    try:
        with h5py.File(file_path, "r") as f:
            # Extract EEG data
            eeg_data = np.array(f["DSamp"]["EEGdata"]).T  # shape (samples, channels)
            fs = float(np.array(f["DSamp"]["fs"])[0][0])

            #Extract channel labels
            labels_ref = f["DSamp"]["label"][0]
            labels = []
            for ref in labels_ref:
                obj = f[ref]
                data = obj[()]
                if isinstance(data, np.ndarray) and data.dtype == np.uint16:
                    decoded = "".join([chr(int(x)) for x in data.flatten() if 32 <= x <= 126])
                    labels.append(decoded.strip())
                elif isinstance(data, (bytes, str)):
                    labels.append(data.decode() if isinstance(data, bytes) else data)
                else:
                    labels.append(str(data))

            # Extract triggers
            if "DSamp/triggers" in f:
                trg = f["DSamp/triggers"]

                def decode_ascii(ref):
                    val = f[ref][()]
                    if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.integer):
                        chars = [chr(int(x)) for x in val.flatten() if 32 <= x <= 126]
                        return "".join(chars)
                    return ""

                codes = [decode_ascii(r) for r in np.ravel(trg["code"])]
                labels_trg = [decode_ascii(r) for r in np.ravel(trg["Label"])]
                times = []
                for ref in np.ravel(trg["time"]):
                    try:
                        times.append(float(f[ref][()][0][0]))
                    except Exception:
                        times.append(np.nan)

                df_triggers = pd.DataFrame({
                    "code": [int(c) if c.isdigit() else np.nan for c in codes],
                    "label": labels_trg,
                    "time_s": times
                }).dropna(subset=["code"]).reset_index(drop=True)

                # Add sample index and event mapping
                df_triggers["sample_index"] = (df_triggers["time_s"] * fs).astype(int)
                code_map = {2: "Block Start", 16: "Stim Start", 32: "Stim Stop"}
                df_triggers["event_name"] = df_triggers["code"].map(code_map).fillna("Unknown")

            else:
                df_triggers = pd.DataFrame(columns=["code", "label", "time_s", "sample_index", "event_name"])

    except Exception as e:
        print(f" HDF5 read failed for {file_path}: {e}")
        raise RuntimeError("Could not load EEG dataset. Ensure this is a MATLAB v7.3 HDF5 file.")

    print(f" Loaded EEG: {eeg_data.shape} | fs={fs} Hz | {len(labels)} channels")
    print(f"   Channels: {labels[:5]} ...")
    print(f"   Triggers: {len(df_triggers)} events ({df_triggers['event_name'].unique()})")

    # Transpose so output is channels x time (to match your old format)
    return eeg_data.T, labels, fs, df_triggers



# Bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
        Apply Butterworth bandpass filter to EEG data
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=1)


def extract_stim_events(triggers, fs):
    """
    Universal trigger extractor compatible with GX_tES_Importing_From_MAT.ipynb style data.
    Works for both:
      - MATLAB v7.3 (HDF5 Group via h5py)
      - MATLAB structs / dicts (v7.0–7.2)
    Returns: DataFrame [code, label, time_s, sample_index, event_name]
    """
    events = []

    # --- DICT (v7.0–7.2) CASE ---
    if isinstance(triggers, dict):
        codes = np.ravel(triggers.get("code", []))
        times = np.ravel(triggers.get("time", []))
        labels = triggers.get("Label", [])
        for i in range(len(codes)):
            try:
                code = int(np.array(codes[i]).squeeze())
            except Exception:
                code = -1
            label = ""
            try:
                if isinstance(labels, (list, np.ndarray)) and len(labels) > i:
                    arr = np.array(labels[i]).flatten()
                    label = "".join(chr(c) for c in arr if c > 0)
            except Exception:
                pass
            try:
                time_s = float(np.array(times[i]).squeeze())
            except Exception:
                time_s = np.nan
            events.append({
                "code": code,
                "label": label,
                "time_s": time_s,
                "sample_index": int(time_s * fs) if np.isfinite(time_s) else 0
            })


    df = pd.DataFrame(events)

    # Ensure time and index columns
    if "time_s" not in df:
        df["time_s"] = np.nan
    if "sample_index" not in df:
        df["sample_index"] = (df["time_s"].fillna(0) * fs).astype(int)

    # Map event codes to names
    code_map = {2: "Block Start", 16: "Stim Start", 32: "Stim Stop"}
    df["event_name"] = df["code"].map(code_map).fillna("Unknown")

    # Sort by time
    df = df.sort_values(by="time_s", ascending=True).reset_index(drop=True)

    print(f" Extracted {len(df)} trigger events: {df['event_name'].value_counts().to_dict()}")
    return df

#epoch extraction
def extract_epochs(eeg_data, triggers_df, fs=1000,
                   prestimulation_time=30, duration_time=30, poststimulation_time=30,
                   during_offset=5, post_offset=35):
    """
    Extract pre-, during-, and post-stimulation epochs from EEG data.

    Parameters
    ----------
    eeg_data : np.array (channels x time)
    triggers_df : DataFrame with 'label' and 'sample_index'
    fs : int (Hz), sampling frequency
    """
    stim_starts = triggers_df[triggers_df['event_name'] == 'Stim Start']['sample_index'].values
    epochs = {"Pre-stimulation": [], "During-stimulation": [], "Post-stimulation": []}

    for t in stim_starts:
        pre_start = int(t - prestimulation_time * fs)
        pre_end = int(t - during_offset * fs)
        dur_start = int(t + during_offset * fs)
        dur_end = int(t + (duration_time + during_offset) * fs)
        post_start = int(t + (duration_time + post_offset) * fs)
        post_end = int(post_start + poststimulation_time * fs)
        if pre_start < 0 or post_end > eeg_data.shape[1]:
            continue
        epochs["Pre-stimulation"].append(eeg_data[:, pre_start:pre_end])
        epochs["During-stimulation"].append(eeg_data[:, dur_start:dur_end])
        epochs["Post-stimulation"].append(eeg_data[:, post_start:post_end])
    return epochs



#epochs function
#baseline correction
def baseline_correction(epoch, fs=1000, baseline_dur=1):
    baseline_samples = int(baseline_dur * fs)
    baseline = np.mean(epoch[:, :baseline_samples], axis=1, keepdims=True)
    return epoch - baseline

#bandpower computation (Welch's Method)
def bandpower(epoch, fs, band):
    """Compute band power for one channel's epoch using Welch PSD."""
    fmin, fmax = band
    freqs, psd = welch(epoch, fs=fs, nperseg=fs*2)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.sum(psd[idx_band]) * freq_res

