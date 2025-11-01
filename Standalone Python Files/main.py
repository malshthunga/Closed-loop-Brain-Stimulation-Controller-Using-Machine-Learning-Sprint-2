# show how trained reinforcement learning controller works
# runs closed loop simulation
# output performance metrics and graphs
"""
Main Script – Closed-Loop EEG Controller (RL Integration)
----------------------------------------------------------
This script orchestrates the complete EEG closed-loop brain stimulation AI Model:
  Section 1 (Data_Extraction): Feature extraction, trigger decoding, latent compression
  Section 2 (AI_Model): Model training, RL agent training, closed-loop simulation

Author: Nethmi Malsha Ranathunga
Supervisor: Dr. Farwa Abbas (SAHMRI)
Date: October 2025
"""

import os
import sys
import subprocess
import importlib.util

# Change to the script directory to ensure relative imports work
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run_script(script_path, description):
    """
    Execute a Python script and handle errors gracefully.
    """
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print('=' * 80)

    try:
        # Load and execute the script
        spec = importlib.util.spec_from_file_location("module", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # If the script has a main function, run it
        if hasattr(module, 'main'):
            module.main()

        print(f"\n✓ Completed: {description}")
        return True
    except Exception as e:
        print(f"\n✗ ERROR in {script_path}: {e}")
        print(f"   Continuing to next script...")
        return False


def data_extraction_section():
    """
    Section 1: Data Extraction Pipeline
    Extracts EEG features, triggers, behavioral data, and compresses to latent space.
    """
    print("\n" + "=" * 80)
    print("SECTION 1: DATA EXTRACTION")
    print("=" * 80)
    print("This section will:")
    print("  1. Load EEG helper functions")
    print("  2. Decode trigger events")
    print("  3. Process behavioral (CTT) task data")
    print("  4. Extract EEG features from .mat files")
    print("  5. Combine features for final dataset")
    print("  6. Compute latent state transitions")
    print("  7. Compress to PCA latent space")
    print("\n This process can take several minutes depending on dataset size.")

    scripts = [
        ("helper_func.py", "Helper Functions"),
        ("trigger_extraction.py", "Trigger Event Decoding"),
        ("ctt_behavioral_task.py", "CTT Behavioral Analysis"),
        ("feature_functions.py", "Feature Functions"),
        ("feature_extraction.py", "EEG Feature Extraction"),
        ("combine_features.py", "Combine Features"),
        ("latent_space_compression_PCA.py", "PCA Latent Compression"),
        ("latent_state_transition_pca.py", "Latent State Transitions")
    ]

    results = []
    for script_path, description in scripts:
        if os.path.exists(script_path):
            success = run_script(script_path, description)
            results.append((description, success))
        else:
            print(f"\n  WARNING: {script_path} not found. Skipping...")
            results.append((description, False))

    print("\n" + "=" * 80)
    print("DATA EXTRACTION SECTION COMPLETE")
    print("=" * 80)
    print("Summary:")
    for desc, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED/SKIPPED"
        print(f"  {status}: {desc}")
    print()


def ai_model_section():
    """
    Section 2: AI Model Training & Closed-Loop Simulation
    Trains MLP/LSTM models, RL agent, and runs closed-loop controller simulation.
    """
    print("\n" + "=" * 80)
    print("SECTION 2: AI MODEL TRAINING & SIMULATION")
    print("=" * 80)
    print("This section will:")
    print("  1. Train MLP & LSTM models for latent transition prediction")
    print("  2. Run closed-loop controller simulation")
    print("  3. Set up closed-loop environment for Reinforcement Learning")
    print("  4. Train RL agent (PPO)")
    print("  5. Test RL agent performance")

    scripts = [
        ("multi_model_trainer.py", "Multi-Model Trainer"),
        ("closed_loop_controller.py", "Closed-Loop Controller Simulation"),
        ("closed_loop_enviornment.py", "Closed-Loop Environment"),
        ("train_rl_agent.py", "Train RL Agent"),
        ("test_rl_agent.py", "Test RL Agent"),
    ]

    results = []
    for script_path, description in scripts:
        if os.path.exists(script_path):
            success = run_script(script_path, description)
            results.append((description, success))
        else:
            print(f"\n  WARNING: {script_path} not found. Skipping...")
            results.append((description, False))

    print("\n" + "=" * 80)
    print("AI MODEL SECTION COMPLETE")
    print("=" * 80)
    print("Summary:")
    for desc, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED/SKIPPED"
        print(f"  {status}: {desc}")
    print()


def main():
    """
    Main execution function with user prompt for data extraction.
    """
    print("\n" + "=" * 80)
    print("EEG CLOSED-LOOP BRAIN STIMULATION")
    print("=" * 80)
    print("Author: Nethmi Malsha Ranathunga")
    print("Supervisor: Dr. Farwa Abbas (SAHMRI)")
    print("=" * 80)

    # Prompt user about data extraction
    print("\n" + "-" * 80)
    print("DATA EXTRACTION OPTIONS")
    print("-" * 80)
    print("Data extraction includes:")
    print("  EEG feature extraction from .mat files")
    print("  Trigger event decoding")
    print("  Behavioral (CTT) task analysis")
    print("  Latent space compression (PCA)")
    print()
    print("WARNING: Data extraction can take several minutes.")
    print("Only run this if:")
    print("  This is your first time running the model")
    print("  You have new/updated .mat files")
    print("  Previously extracted data is missing or corrupted")
    print("-" * 80)

    while True:
        user_input = input("\nWould you like to complete data extraction? (yes/no): ").strip().lower()

        if user_input in ['yes', 'y']:
            print("\n✓ Running Data Extraction + AI Model")
            data_extraction_section()
            ai_model_section()
            break
        elif user_input in ['no', 'n']:
            print("\n✓ Skipping data extraction. Running AI MODEL only.")
            ai_model_section()
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print("Check the output files and visualizations generated in the working directory.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
