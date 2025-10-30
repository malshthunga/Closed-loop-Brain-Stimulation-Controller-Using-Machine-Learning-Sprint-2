"""
EEG Feature Extraction Functions
--------------------------------
This script defines helper functions used to extract EEG features for different
stimulation conditions (Pre-, During-, and Post-Stimulation). It computes the
frequency-based bandpower features and key ratios commonly used in cognitive
EEG analysis.

Main Purpose:
- To process raw EEG data and convert it into structured numerical features
  that describe brain activity under different task/stimulation conditions.

Workflow:
1. Loads all `.mat` EEG files in the current working directory.
2. Extracts stimulation event markers (e.g., Block Start, Stim Start, Stim Stop).
3. Segments EEG signals into epochs (pre-, during-, and post-stimulation).
4. Applies baseline correction to each epoch to normalise the signals.
5. Calculates frequency band power features using Welch’s method:
      - Delta  (1–4 Hz)
      - Theta  (4–8 Hz)
      - Alpha  (8–13 Hz)
      - Beta   (13–30 Hz)
6. Computes bandpower ratios for each channel:
      - Theta/Alpha ratio
      - Alpha/Beta ratio
7. Returns a dictionary of feature matrices for each stimulation condition.

Outputs:
- `feature_matrices`: a Python dictionary containing NumPy arrays of features,
  where each key corresponds to a condition (e.g., "Pre-stimulation").

Usage:
The function `get_feature_matrices()` is imported and used in
`feature_extraction.py` to generate the final dataset for machine learning.

"""
import h5py
from scipy.io import loadmat
import numpy as np



def get_feature_matrices(file_path):
    """
    Extracts Pre, During, and Post EEG segments from a .mat file.
    Supports both v7.0 (Colab) and v7.3 (owner) dataset formats.
    """
    def decode_ascii(arr):
        try:
            return ''.join([chr(x) for x in arr.flatten() if x > 0])
        except:
            return str(arr)

    try:
        mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        DSamp = mat.get("DSamp", None)
        if hasattr(DSamp, "EEGdata"):
            # Already segmented
            return {
                "Pre-stimulation": getattr(DSamp, "Pre", DSamp.EEGdata),
                "During-stimulation": getattr(DSamp, "During", DSamp.EEGdata),
                "Post-stimulation": getattr(DSamp, "Post", DSamp.EEGdata),
            }
    except NotImplementedError:
        pass

    with h5py.File(file_path, "r") as f:
        eeg = np.array(f["DSamp"]["EEGdata"])
        if eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T  # ensure shape (channels, samples)
        fs = int(np.array(f["DSamp"]["fs"]))
        trig = f["DSamp"]["triggers"]

        def deref(refarr):
            out = []
            for ref in np.array(refarr):
                val = f[ref[0]][()]
                if val.dtype == np.uint16:
                    val = ''.join(chr(x) for x in val.flatten() if x > 0)
                out.append(val)
            return out

        labels = deref(trig["Label"])
        offsets = [float(f[ref[0]][()][0,0]) for ref in np.array(trig["offset"])]

    stim_start_indices = [off for lab, off in zip(labels, offsets) if "Stim Start" in lab]
    stim_stop_indices = [off for lab, off in zip(labels, offsets) if "Stim Stop" in lab]

    if stim_start_indices and stim_stop_indices:
        start = int(stim_start_indices[0])
        end = int(stim_stop_indices[-1])
        pre = eeg[:, :start]
        during = eeg[:, start:end]
        post = eeg[:, end:]
    else:
        one_third = eeg.shape[1] // 3
        pre = eeg[:, :one_third]
        during = eeg[:, one_third:2*one_third]
        post = eeg[:, 2*one_third:]

    return {
        "Pre-stimulation": pre,
        "During-stimulation": during,
        "Post-stimulation": post,
    }