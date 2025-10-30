# Closed-Loop Brain Stimulation

## Project Information

- **Author:** Nethmi Malsha Ranathunga
- **Supervisor:** Dr. Farwa Abbas (SAHMRI)
- **Date:** November 2025

---

### Script Structure

The project is organized into two main sections:

1. **Section 1 (Data_Extraction):**
    - Helper Functions
    - Feature Functions 
    - Feature extraction
    - Trigger extraction
    - CTT behavioral tasks
    - Latent space compression
    - Latent state transition

2. **Section 2 (AI_Model):**
    - Multi Model training
    - Closed-Loop Environment
    - Closed-Loop Controller Simulation
    - Combine Features
    - Train RL Agent
    - Test RL Agent

---

## Prerequisites

- Python 3.x
- All dependencies listed in the import sections of each Python file
- EEG data files in `.mat` format (for Data_Extraction mode)

> **Note:** These scripts are designed to run within the same folder as the data and exports from other scripts (everything in one folder).

---

## Usage Instructions

### Option 1: Data_Extraction Mode (with new data)

Use this mode to process new EEG data from scratch.

1. Copy the standalone Python files from the `Standalone Python Files` folder into a working directory
2. Place your EEG data files (`.mat` format) into the same directory as the Python files
3. Ensure all required dependencies are installed
   - Dependencies are listed in the import sections of each Python file
4. Run the `main.py` file:
   ```python
   python main.py
   ```
5. When prompted to run both sections, type **`yes`** to execute all scripts
   - This will run both the data extraction scripts and the model scripts

### Option 2: AI_Model Mode (with pre-extracted features)

Use this mode to train and test models using the already extracted features from the test dataset.

1. Use the Python files and data located in the `Python Files and Output` folder
2. Ensure all required dependencies are installed
   - Dependencies are listed in the import sections of each Python file
3. Run the `main.py` file:
   ```python
   python main.py
   ```
4. When prompted to run both sections, type **`no`** to run only the model scripts
   - This will skip the data extraction and run only the AI model training and simulation

---

## Project Structure

```
Closed-Loop-Brain-Stimulation/
├── Python Files and Output/     # Main working directory with already extracted features
│   └── main.py                  # Main entry point
│
└── Standalone Python Files/     # Standalone scripts for new data processing
    └── main.py                  # Main entry point
```

## References

### Dataset

**Primary Dataset Source:**
- [GX tES EEG Dataset (Zenodo)](https://zenodo.org/records/15572614)

**How to Download the Data:**

1. **Via GitHub Examples:**
   - Follow instructions and examples from the [GX_tES_EEG_Physio_Behavior repository](https://github.com/ngebodh/GX_tES_EEG_Physio_Behavior)
   - View examples at: [https://github.com/ngebodh/GX_tES_EEG_Physio_Behavior/tree/master/examples](https://github.com/ngebodh/GX_tES_EEG_Physio_Behavior/tree/master/examples)

2. **Direct Download from Zenodo:**
   - Visit the [Zenodo dataset page](https://zenodo.org/records/15572614)
   - Scroll to the "Files" section
   - Click the "Download" button for each `.mat` file

---

### Citations for Primary Dataset Source:
Listed below as required by Dataset Source 

1. **Gebodh, N., Esmaeilpour, Z., Datta, A. et al.** (2021). Dataset of concurrent EEG, ECG, and behavior with multiple doses of transcranial electrical stimulation. *Sci Data* 8, 274.  
   [https://doi.org/10.1038/s41597-021-01046-y](https://doi.org/10.1038/s41597-021-01046-y)

2. **Gebodh N, Miskovic V, Laszlo S, Datta A, Bikson M.** (2024). Frontal HD-tACS enhances behavioral and EEG biomarkers of vigilance in continuous attention task. *Brain Stimul* 17(3):683–6.  
   [https://doi.org/10.1016/j.brs.2024.05.009](https://doi.org/10.1016/j.brs.2024.05.009)

---

### Additional Resources

- **MiSO Implementation (Adaptation Inspiration):**  
  [https://github.com/yuumii-san/MiSO](https://github.com/yuumii-san/MiSO)  
  *MicroStimulation Optimization for brain stimulation*

- **Sample .mat File Import Example:**  
  [GX_tES_Importing_From_MAT.ipynb](https://github.com/ngebodh/GX_tES_EEG_Physio_Behavior/blob/master/examples/GX_tES_Importing_From_MAT.ipynb)  
  *Jupyter notebook demonstrating how to import and work with the dataset*