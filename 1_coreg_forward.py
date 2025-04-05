from scipy import io
import shutil
import sys
import os
from pprint import pprint
from osl_ephys import source_recon
import numpy as np
import os.path as op
import mne
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# MAIN_DIR is outside the project, so keep as absolute path
MAIN_DIR = os.path.join(SCRIPT_DIR, "Participants")

# Use relative paths for directories within the project
PLOTS_DIR = os.path.join(SCRIPT_DIR, "Plots")
PLOTS_2D = os.path.join(PLOTS_DIR, "2D_electrode_positions_plots")
ELECTRODE_AND_FORWARD_PLOTS = os.path.join(
    PLOTS_DIR, "3D_electrode_and_forward")

fsl_dir = '~/fsl'  # This is the path to the fsl installation
source_recon.setup_fsl(fsl_dir)
# FUNCTION - 1 : Get the necessary paths for each subject


def get_subjects(participants_dir, conditions=["EC", "EO"]):
    """
    for each subject returns the paths of raw fif files before polhemeus registration for specified conditions This will hopefully get the every file path we need
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
    return subjects_dict


# FUNCTION - 2 : Polhemus Registration and Fix


def polhemus_registration_and_fix(subject_data):
    """
    Performs Polhemus registration and fixes electrode positions for EEG data.
    """

    eo_path = subject_data['EO']
    print(f"USING {eo_path} FOR EYES OPEN CONDITION")
    ec_path = subject_data['EC']
    print(f"USING {ec_path} FOR EYES CLOSED CONDITION")
    localizer_path = subject_data['localizer_path']
    subject_id = os.path.basename(eo_path).split("_")[0]
    raw_eo = mne.io.read_raw_eeglab(eo_path, preload=True)
    raw_ec = mne.io.read_raw_eeglab(ec_path, preload=True)
    polhemus_dir = subject_data['Polhemus_dir']
    ec_save_dir = subject_data['EC_save_dir']
    eo_save_dir = subject_data['EO_save_dir']
    X = io.loadmat(localizer_path, simplify_cells=True)
    print(f"Channel locations: for {subject_id}")
    ch_pos = {}
    for i in range(len(X["Channel"]) - 1):
        key = X["Channel"][i]["Name"].split("_")[2]
        if key[:2] == "FP":
            key = "Fp" + key[2]
        value = X["Channel"][i]["Loc"]
        ch_pos[key] = value
        x, y, z = value
        print(f"{key:<8} {x:>10.4f} {y:>10.4f} {z:>10.4f}")
    print("----------------------------------------")
    print("\nHead Points:")
    print("-" * 50)
    hp = X["HeadPoints"]["Loc"]
    nas = np.mean([hp[:, 0], hp[:, 3]], axis=0) - np.array([-0.02, 0.01, 0.01])
    lpa = np.mean([hp[:, 1], hp[:, 4]], axis=0) - np.array([-0.02, 0.01, 0.01])
    rpa = np.mean([hp[:, 2], hp[:, 5]], axis=0) - np.array([-0.02, 0.01, 0.01])
    print(f"NAS: {nas[0]:>10.4f} {nas[1]:>10.4f} {nas[2]:>10.4f}")
    print(f"LPA: {lpa[0]:>10.4f} {lpa[1]:>10.4f} {lpa[2]:>10.4f}")
    print(f"RPA: {rpa[0]:>10.4f} {rpa[1]:>10.4f} {rpa[2]:>10.4f}")
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos, nasion=nas, lpa=lpa, rpa=rpa)
    raw_eo.set_montage(montage)
    raw_ec.set_montage(montage)
    nas = np.array([-nas[1], nas[0], nas[2]])
    lpa = np.array([-lpa[1], lpa[0], lpa[2]])
    rpa = np.array([-rpa[1], rpa[0], rpa[2]])
    for name, point in [("lpa", lpa), ("nasion", nas), ("rpa", rpa)]:
        filename = os.path.join(polhemus_dir, f"polhemus_{name}.txt")
        np.savetxt(filename, point, fmt='%.7e')
        print(f"Saved {name} to {filename}")
    ec_raw_path = os.path.join(ec_save_dir, "raw.fif")
    eo_raw_path = os.path.join(eo_save_dir, "raw.fif")
    raw_ec.save(ec_raw_path, overwrite=True)
    raw_eo.save(eo_raw_path, overwrite=True)


# FUNCTION - 3 : Plotting 2D electrode positions

def plot_2d_electrode_positions(subject_data):
    """
    Plots 2D electrode positions for eyes open and closed conditions.

    This function loads the raw EEG data for both conditions, creates a montage
    """

    fig = plt.figure(figsize=(10, 10))
    # One of them is enough since for both files the montage is the same
    raw = mne.io.read_raw_fif(subject_data["EC_saved_file"])
    mne.viz.plot_montage(
        raw.get_montage(), show_names=True)
    plt.title(
        f"2D Electrode Positions for participant {subject_data['subject_id']}")
    plt.savefig(os.path.join(
        PLOTS_2D, f"{subject_data['subject_id']}_2D_electrode_positions.png"))
    plt.close()

# FUNCTION - 4 : Plotting Forward Model (Interactive Plotly HTML)


def plot_forward_model(subject_data):
    """
    Plots the forward inside the subject's electrode positions to check the alignment
    """
    forward_path = subject_data["forward_path"]
    fwd = mne.read_forward_solution(forward_path)
    src = fwd['src'][0]
    all_cords = src['rr']
    used_indices = src['vertno']
    forward_cords = all_cords[used_indices]
    raw = mne.io.read_raw_fif(subject_data["EC_saved_file"])
    chan_pos = raw.get_montage().get_positions()
    electrode_names = list(chan_pos["ch_pos"].keys())
    chan_x = []
    chan_y = []
    chan_z = []
    for el in chan_pos["ch_pos"]:
        chan_x.append(chan_pos["ch_pos"][el][0])
        chan_y.append(chan_pos["ch_pos"][el][1])
        chan_z.append(chan_pos["ch_pos"][el][2])
    chan_x = np.array(chan_x)
    chan_y = np.array(chan_y)
    chan_z = np.array(chan_z)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=forward_cords[:, 0],
        y=forward_cords[:, 1],
        z=forward_cords[:, 2],
        mode='markers',
        marker=dict(color='dodgerblue', size=3, opacity=0.8),
        name='Source Space'
    ))
    fig.add_trace(go.Scatter3d(
        x=chan_x,
        y=chan_y,
        z=chan_z,
        mode='markers+text',
        marker=dict(color='red', size=5, opacity=0.9),
        text=electrode_names,
        textposition="top center",
        hoverinfo='text',
        name='Electrodes'
    ))
    fig.update_layout(
        title="EEG Electrode Positions and Source Space for participant " +
        subject_data['subject_id'],
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    fig.write_html(os.path.join(
        ELECTRODE_AND_FORWARD_PLOTS, f"{subject_data['subject_id']}_forward_model.html"))
# =#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=#


subjects_data = get_subjects(MAIN_DIR)
pprint(subjects_data)

# SECTION 1 : Polhemus Registration and Fix
for subject in subjects_data:
    polhemus_registration_and_fix(subjects_data[subject])
    plot_2d_electrode_positions(subjects_data[subject])
# !SECTION

# SECTION 2 : Compute Surfaces
for subject in subjects_data:
    smri_path = subjects_data[subject]["smri_path"]
    recon_dir = subjects_data[subject]["recon_dir"]
    # Using one since for the coreg or fwd computation it does not matter
    raw_path = subjects_data[subject]["EO_save_dir"] + "/raw.fif"
    print(f"Computing surfaces for {subject}")
    source_recon.rhino.compute_surfaces(
        smri_path,
        recon_dir,
        subject,
        include_nose=False,
        do_mri2mniaxes_xform=True,
        recompute_surfaces=False,
    )
# !SECTION

# SECTION 3 : Copy Polhemus Points to Recon Folder

for subject in subjects_data:
    polhemus_dir = subjects_data[subject]["Polhemus_dir"]
    nasion_path = os.path.join(polhemus_dir, "polhemus_nasion.txt")
    lpa_path = os.path.join(polhemus_dir, "polhemus_lpa.txt")
    rpa_path = os.path.join(polhemus_dir, "polhemus_rpa.txt")
    coreg_dir = os.path.join(
        subjects_data[subject]["recon_dir"], subject, "rhino", "coreg")

    # Copy the fiducial files to the coreg directory
    shutil.copy(nasion_path, os.path.join(coreg_dir, "polhemus_nasion.txt"))
    shutil.copy(lpa_path, os.path.join(coreg_dir, "polhemus_lpa.txt"))
    shutil.copy(rpa_path, os.path.join(coreg_dir, "polhemus_rpa.txt"))
    print(f"Copied fiducial files to {coreg_dir} for subject {subject}")
# !SECTION

# SECTION 4 : Coregistration
for subject in subjects_data:
    source_recon.rhino.coreg(
        subjects_data[subject]["EO_save_dir"] + "/raw.fif",
        subjects_data[subject]["recon_dir"],
        subject,
        use_headshape=False,
        use_nose=False,
    )
# !SECTION

# SECTION 5 : Forward Model
gridstep = 7
for subject in subjects_data:
    source_recon.rhino.forward_model(
        subjects_data[subject]["recon_dir"],
        subject,
        gridstep=gridstep,
        meg=False,
        eeg=True,
        model="Triple Layer",
        verbose=True,
    )
    plot_forward_model(subjects_data[subject])
# !SECTION
