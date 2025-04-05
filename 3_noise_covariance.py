# 3_noise_covariance.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import mne


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
