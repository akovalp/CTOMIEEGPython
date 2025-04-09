# 3_noise_covariance.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pprint import pprint
import mne

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.join(SCRIPT_DIR, "Participants")

# FUNCTION - 1 : Get the subjects


def get_subjects(participants_dir, conditions=["EC", "EO"], bands=["alpha", "beta", "theta"]):
    """
    for each subject returns the paths of raw fif files before polhemeus registration for specified conditions This will hopefully get the every file path we need for this file most of the path are redundant but we will keep them for now 
    """
    subjects = []
    for item in os.listdir(participants_dir):
        if item.startswith('sub-') and os.path.isdir(os.path.join(participants_dir, item)):
            subjects.append(item)

    subjects_dict = {}

    for subject in subjects:
        subjects_dict[subject] = {}
        subjects_dict[subject]["subject_id"] = subject
        for condition in conditions:
            # Input directories and files
            eeg_dir = os.path.join(
                participants_dir, subject, "EEG_Before_Chan", condition)
            eeg_path = os.path.join(
                eeg_dir, subject + "_" + condition + ".set")
            subjects_dict[subject][condition] = eeg_path

            # Save directories
            eeg_save_dir = os.path.join(
                participants_dir, subject, "EEG_After_Chan", condition)
            subjects_dict[subject][f"{condition}_save_dir"] = eeg_save_dir
            eeg_saved_file = os.path.join(eeg_save_dir, "raw.fif")
            subjects_dict[subject][f"{condition}_saved_file"] = eeg_saved_file

            # Other paths
            localizer_path = os.path.join(
                participants_dir, subject, "Localizer", f"{subject}.mat")
            subjects_dict[subject]["localizer_path"] = localizer_path
            Polhemus_dir = os.path.join(participants_dir, subject, "Polhemus")
            subjects_dict[subject]["Polhemus_dir"] = Polhemus_dir

            # Coreg Paths
            smri_path = os.path.join(
                participants_dir, subject, "SMRI", f"{subject}_ses-01_inv-2_mp2rage.nii.gz")
            subjects_dict[subject]["smri_path"] = smri_path
            recon_dir = os.path.join(participants_dir, subject, "Recon")
            subjects_dict[subject]["recon_dir"] = recon_dir
            forward_path = os.path.join(
                recon_dir, subject, "rhino", "model-fwd.fif")
            subjects_dict[subject]["forward_path"] = forward_path

            # Bandpass Filter Paths
            for band in bands:
                filtered_path = os.path.join(
                    participants_dir, subject, "EEG_After_Chan", condition, f"{band}_{condition}.fif")
                subjects_dict[subject][f"{condition}_{band}_path"] = filtered_path

    return subjects_dict


# FUNCTION - 2 : Compute the noise covariance
def compute_and_save_noise_covariance(subjects_dict):
    """
    Compute and save noise covariance matrices for each subject, condition and frequency band.
    Always recomputes and overwrites existing files.

    Args:
        subjects_dict: Dictionary containing subject data paths
    """
    # Define frequency bands and conditions
    bands = ['alpha', 'beta', 'theta']
    conditions = ['EC', 'EO']

    for subject_id, subject_data in tqdm(subjects_dict.items()):
        print(f"\nProcessing subject: {subject_id}")

        # Create output directory for noise covariance matrices
        noise_cov_dir = os.path.join(
            SCRIPT_DIR, "Noise_Covariances", subject_id)
        os.makedirs(noise_cov_dir, exist_ok=True)
        print(f"Created/verified noise covariance directory: {noise_cov_dir}")

        # Process each condition and frequency band
        for condition in conditions:
            print(f"\nProcessing condition: {condition}")
            for band in bands:
                # Get path to filtered data
                raw_path = subject_data[f"{condition}_{band}_path"]
                print(f"\nProcessing band: {band}")
                print(f"Reading filtered data from: {raw_path}")

                # Define output file path
                noise_cov_path = os.path.join(
                    noise_cov_dir, f"{condition}_{band}_noise_cov.fif")
                print(
                    f"Output noise covariance will be saved to: {noise_cov_path}")

                # Check if input file exists
                if os.path.exists(raw_path):
                    print(f"Confirmed input file exists: {raw_path}")
                    # Check if file already exists - but always recompute
                    if os.path.exists(noise_cov_path):
                        print(
                            f"Noise covariance file exists for {condition}_{band} for {subject_id}, but will be recomputed and overwritten")

                    print(
                        f"Computing noise covariance for {condition}_{band} for {subject_id}")

                    try:
                        # Load raw data
                        print(f"Loading raw data from: {raw_path}")
                        raw = mne.io.read_raw_fif(raw_path, preload=True)
                        print(
                            f"Successfully loaded raw data, shape: {raw.get_data().shape}")

                        # Compute noise covariance
                        print(f"Computing noise covariance with method='empirical'")
                        noise_cov = mne.compute_raw_covariance(
                            raw, tmin=0, tmax=None, method='empirical')
                        print(f"Noise covariance computed successfully")

                        # Save noise covariance
                        print(f"Saving noise covariance to: {noise_cov_path}")
                        noise_cov.save(noise_cov_path, overwrite=True)
                        print(
                            f"Successfully saved noise covariance to {noise_cov_path}")
                    except Exception as e:
                        print(
                            f"Error processing {condition}_{band} for {subject_id}: {str(e)}")
                        print(f"Failed file path: {raw_path}")
                else:
                    print(f"File not found: {raw_path}")

        print(f"Completed processing for {subject_id}")

# =#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=#


subjects_dict = get_subjects(MAIN_DIR)
compute_and_save_noise_covariance(subjects_dict)
