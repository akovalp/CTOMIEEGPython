# 2_band_filter.py
import os
from pprint import pprint
import mne


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAIN_DIR = os.path.join(SCRIPT_DIR, "Participants")
frequency_bands = {
    'theta': (4, 8),    # 4-8 Hz
    'alpha': (8, 14),   # 8-14 Hz
    'beta': (14, 30),   # 14-30 Hz
}
# FUNCTION - 1 : Same function as in 1_coreg_forward.py but we will add the bandpass filter paths which for each condition will be added to the


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


# FUNCTION - 2 : Bandpass Filtering

def filter_and_save_frequency_bands(subjects_dict):
    """
    Apply bandpass filters to EEG data for different frequency bands and save the filtered data.

    Parameters:
        subjects_dict: Dictionary containing paths for each subject's data files
    """
    for subject in subjects_dict:
        print(f"\n{'='*20} Processing Subject: {subject} {'='*20}")

        # Verify files exist before loading
        ec_file = subjects_dict[subject]["EC_saved_file"]
        eo_file = subjects_dict[subject]["EO_saved_file"]

        print(f"Loading Eyes Closed data from: {ec_file}")
        print(f"Loading Eyes Open data from: {eo_file}")

        # Critical error fix: Need to reload raw data for each band to avoid cumulative filtering
        for band in ["alpha", "beta", "theta"]:
            # Reload raw data for each band to avoid cumulative filtering
            EC_raw = mne.io.read_raw_fif(ec_file, preload=True)
            EO_raw = mne.io.read_raw_fif(eo_file, preload=True)

            l_freq, h_freq = frequency_bands[band]
            print(f"\n{'-'*50}")
            print(
                f"BAND: {band.upper()} | Frequency range: {l_freq}-{h_freq} Hz")
            print(f"{'-'*50}")

            # Filter data
            print(
                f"Applying {band} bandpass filter ({l_freq}-{h_freq} Hz) to EC data...")
            EC_raw.filter(l_freq=l_freq, h_freq=h_freq, method="iir")

            print(
                f"Applying {band} bandpass filter ({l_freq}-{h_freq} Hz) to EO data...")
            EO_raw.filter(l_freq=l_freq, h_freq=h_freq, method="iir")

            # Save filtered data
            ec_save_path = subjects_dict[subject][f"EC_{band}_path"]
            eo_save_path = subjects_dict[subject][f"EO_{band}_path"]

            print(f"Saving EC {band} filtered data to: {ec_save_path}")
            EC_raw.save(ec_save_path, overwrite=True)

            print(f"Saving EO {band} filtered data to: {eo_save_path}")
            EO_raw.save(eo_save_path, overwrite=True)


subjects_dict = get_subjects(MAIN_DIR)
pprint(subjects_dict)
filter_and_save_frequency_bands(subjects_dict)
