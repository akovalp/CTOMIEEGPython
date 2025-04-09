import os
import numpy as np
import pandas as pd
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_raw
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
import plotly.graph_objects as go
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.join(SCRIPT_DIR, "Participants")
# This will be transformed for each subject to their own head shape combined with their electrode spaces
ATLAS_FILE = "/Users/alpmac/Code Works/downloaderdeneme/selected_atlas_areas.txt"
CSV_DIR = os.path.join(SCRIPT_DIR, "CSV")
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
            # Save directories
            eeg_save_dir = os.path.join(
                participants_dir, subject, "EEG_After_Chan", condition)
            subjects_dict[subject][f"{condition}_save_dir"] = eeg_save_dir
            eeg_saved_file = os.path.join(eeg_save_dir, "raw.fif")
            subjects_dict[subject][f"{condition}_saved_file"] = eeg_saved_file

            # Coreg Paths
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

            # Noise Cov Paths
            noise_cov_dir = os.path.join(
                SCRIPT_DIR, "Noise_Covariances", subject)
            subjects_dict[subject]["noise_cov_dir"] = noise_cov_dir
            for condition in conditions:
                for band in bands:
                    noise_cov_path = os.path.join(
                        noise_cov_dir, f"{condition}_{band}_noise_cov.fif")
                    subjects_dict[subject][f"{condition}_{band}_noise_cov_path"] = noise_cov_path

            # Transformation Matrices
            coreg_coreg_dir = os.path.join(
                participants_dir, subject, "Recon", subject, "rhino", "coreg")
            coreg_surfaces_dir = os.path.join(
                participants_dir, subject, "Recon", subject, "rhino", "surfaces")
            mni_mri_t_path = os.path.join(
                coreg_surfaces_dir, "mni_mri-trans.fif")
            subjects_dict[subject]["mni_mri_t_path"] = mni_mri_t_path
            head_mri_t_path = os.path.join(
                coreg_coreg_dir, "head_scaledmri-trans.fif")
            subjects_dict[subject]["head_mri_t_path"] = head_mri_t_path

    return subjects_dict


# Function 2 : Transform atlas Coordinates to HEAD
def transform_atlas_coordinates_and_select_vertices(subject_data, save=False):
    """
    Transform atlas coordinates to head coordinates.
    """
    # 1) LOAD ATLAS COORDINATES
    atlas_df = pd.read_csv(ATLAS_FILE, delimiter="\t", header=None)
    region_names = atlas_df[4].values
    atlas_coords_mni_mm = atlas_df[[1, 2, 3]].values.astype(float) * 1000

    # 2) LOAD TRANSFORMS
    mni_mri_t = mne.read_trans(subject_data["mni_mri_t_path"])
    print("-"*100)
    print("\nMNI->MRI transform matrix:")
    print(mni_mri_t['trans'])
    print("-"*100)
    # Load and invert head->MRI transform to get MRI->head
    head_mri_t = mne.read_trans(subject_data["head_mri_t_path"])
    mri_head_t = mne.transforms.invert_transform(head_mri_t)
    print("\nMRI->Head transform matrix:")
    print(mri_head_t['trans'])
    print("-"*100)
    # 3) APPLY TRANSFORMATIONS
    # First transform: MNI -> MRI (coordinates still in mm)
    atlas_coords_mri_mm = mne.transforms.apply_trans(
        mni_mri_t['trans'], atlas_coords_mni_mm)

    # Second transform: MRI -> Head (coordinates still in mm)
    atlas_coords_head_mm = mne.transforms.apply_trans(
        mri_head_t['trans'], atlas_coords_mri_mm)

    # 4) CONVERT TO METERS and Save the Atlas Coordinates to a df
    atlas_coords_head_m = atlas_coords_head_mm / 1000.0
    atlas_coords_df = pd.DataFrame(
        atlas_coords_head_m, columns=['X', 'Y', 'Z'])
    atlas_coords_df['Region'] = region_names

    print(
        f"For {subject_data['subject_id']} Transformed Atlas Coordinates to Head Coordinates")
    print(atlas_coords_df)
    print("-"*100)

    # 5) SELECT THE VERTEX INDICES THAT ARE IN THE HEAD
    selected_verticies = {}
    forward_solution = mne.read_forward_solution(subject_data["forward_path"])
    src = forward_solution['src'][0]
    all_cords = src['rr']
    print(all_cords.shape)
    used_indices = src['vertno']
    forward_cords = all_cords[used_indices]

    for i, node in enumerate(atlas_coords_head_m):
        distances = np.linalg.norm(forward_cords - node, axis=1)
        closest_index = np.argmin(distances)
        vertex_number = used_indices[closest_index]
        selected_verticies[region_names[i]] = vertex_number
        print(f"Atlas node '{region_names[i]}':")
        print(f"  Atlas coordinates: {node}")
        print(f"  Closest vertex: {vertex_number}")
        print(f"  Distance: {distances[closest_index]:.4f} meters")
        print(f"  Vertex coordinates: {forward_cords[closest_index]}\n")

    return {
        'atlas_coords_head_m': atlas_coords_head_m,
        'selected_vertices': selected_verticies,
        'forward_cords': forward_cords,
        'used_indices': used_indices
    }


def visualize_atlas_and_vertices(transform_results, title=None, save_path=None):

    atlas_coords_head_m = transform_results['atlas_coords_head_m']
    selected_vertices = transform_results['selected_vertices']
    forward_cords = transform_results['forward_cords']
    used_indices = transform_results['used_indices']

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all forward solution vertices in light blue
    ax.scatter(forward_cords[:, 0], forward_cords[:, 1], forward_cords[:, 2],
               c='lightblue', s=1, alpha=0.5, label='Forward Solution Vertices')

    # Plot atlas nodes in red
    ax.scatter(atlas_coords_head_m[:, 0], atlas_coords_head_m[:, 1], atlas_coords_head_m[:, 2],
               c='red', s=100, label='Atlas Nodes')

    # Plot selected vertices in green
    selected_indices = [np.where(used_indices == vertex)[0][0]
                        for vertex in selected_vertices.values()]
    selected_coords = forward_cords[selected_indices]
    ax.scatter(selected_coords[:, 0], selected_coords[:, 1], selected_coords[:, 2],
               c='green', s=100, label='Selected Vertices')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()

    if title:
        plt.title(title)
    else:
        plt.title('Forward Solution with Atlas Nodes and Selected Vertices')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def compute_source_time_courses_for_subject(subject_data, selected_vertices, csv_output_dir):
    """
    Compute inverse operators and source time courses for a subject across frequency bands
    and conditions (EC/EO). Save time series data as CSV files.

    Parameters:
        subject_data: Dictionary containing all paths for the subject
        selected_vertices: Dictionary mapping region names to vertex indices
        csv_output_dir: Directory where CSV files will be saved
    """
    import os
    import numpy as np
    import pandas as pd
    import mne
    from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
    from tqdm import tqdm

    # Create subject-specific output directory
    subject_id = subject_data['subject_id']
    participant_csv_dir = os.path.join(csv_output_dir, subject_id)
    os.makedirs(participant_csv_dir, exist_ok=True)

    # Parameters for source reconstruction
    snr = 3.0
    lambda2 = 1.0 / snr**2
    method = "eLORETA"  # Using eLORETA method
    print("-"*100)

    print(f"\nProcessing subject: {subject_id}\n")

    print("-"*100)

    # Check for forward solution
    if not os.path.exists(subject_data["forward_path"]):
        print(f"Warning: No forward solution available for {subject_id}")
        return

    # Load forward solution
    try:
        print("-"*100)
        print(
            f"\nForward solution path that will be used: {subject_data['forward_path']}\n")
        print("-"*100)
        fwd = mne.read_forward_solution(subject_data["forward_path"])
        print(f"Loaded forward solution from {subject_data['forward_path']}")
    except Exception as e:
        print(f"Error loading forward solution for {subject_id}: {e}")
        return

    # Define frequency bands and conditions
    bands = ['alpha', 'beta', 'theta']
    conditions = ['EC', 'EO']  # Using uppercase to match existing structure

    # Process each condition and frequency band
    for condition in conditions:
        for band in bands:
            try:
                # Define paths for raw data and noise covariance
                raw_path = subject_data[f"{condition}_{band}_path"]
                print("-"*100)
                print(f"\nRaw path that will be used: {raw_path}\n")
                print("-"*100)
                noise_cov_path = subject_data[f"{condition}_{band}_noise_cov_path"]
                print("-"*100)
                print(
                    f"\nNoise covariance path that will be used: {noise_cov_path}\n")
                print("-"*100)

                # Check if files exist
                if not os.path.exists(raw_path):
                    print(f"Warning: Raw file not found at {raw_path}")
                    continue

                if not os.path.exists(noise_cov_path):
                    print(
                        f"Warning: Noise covariance file not found at {noise_cov_path}")
                    continue
                print("-"*100)
                print(f"\nProcessing {condition}_{band} for {subject_id}\n")
                print("-"*100)

                # Load raw data and noise covariance
                print(
                    f"\nLoading {condition}_{band} raw data from: {raw_path}\n")
                print("-"*100)
                raw = mne.io.read_raw_fif(raw_path, preload=True)

                # Set EEG reference using projection (CRITICAL FIX)
                raw.set_eeg_reference(projection=True)

                print(
                    f"\nLoading {condition}_{band} noise covariance from: {noise_cov_path}\n")
                print("-"*100)
                noise_cov = mne.read_cov(noise_cov_path)

                # Compute rank from noise covariance
                rank = mne.compute_rank(noise_cov, info=raw.info)
                print(f"\nComputed rank for {condition}_{band}: {rank}\n")

                # Create the inverse operator
                print(f"\nComputing {condition}_{band} inverse operator...\n")
                inverse_operator = make_inverse_operator(
                    raw.info,
                    fwd,
                    noise_cov,
                    loose="auto",
                    depth=0.8,
                    rank=rank
                )

                print(
                    f"\nComputing {condition}_{band} source time course...\n")

                # Apply inverse operator to raw data
                stc = apply_inverse_raw(
                    raw,
                    inverse_operator,
                    lambda2=lambda2,
                    method=method,
                    pick_ori='vector',
                    verbose=True
                )

                # Print dimensions
                print("-"*100)
                print(f"\n{condition}_{band} Data dimensions check:")
                print(f"Time vector length: {len(stc.times)}")
                print(f"stc.data shape: {stc.data.shape}")
                print("-"*100)

                # Dictionary to store time series data for each atlas region
                selected_time_series = {}

                # Iterate through each region and its corresponding vertex
                for region, vertex_no in selected_vertices.items():
                    # Find where this vertex exists in the source time course (stc) data
                    vertex_idx = np.where(stc.vertices[0] == vertex_no)[0]

                    # Check if the vertex was found
                    if len(vertex_idx) == 0:
                        print(
                            f"Warning: Vertex {vertex_no} for region '{region}' not found in stc.vertices!")
                        continue

                    # Get the index position of this vertex in the stc data
                    idx = vertex_idx[0]

                    # Extract the time series for this vertex (shape: 3, n_times)
                    data = stc.data[idx, :, :]

                    # Compute the magnitude of the activity at each time point
                    intensity = np.sqrt(np.sum(data**2, axis=0))

                    # Store the computed time series for this region
                    selected_time_series[region] = intensity

                # Create a DataFrame with time as the first column
                time_vector = stc.times
                combined_df = pd.DataFrame({'Time': time_vector})

                # Add each region's time series as a new column
                for region, intensity in selected_time_series.items():
                    combined_df[region] = intensity

                # Scale intensity values by 10^14 before saving to avoid small e-15 values
                for column in combined_df.columns:
                    if column != 'Time':  # Skip the time column
                        combined_df[column] = combined_df[column] * 1e14

                # Save the results to a CSV file
                output_filename = os.path.join(
                    participant_csv_dir, f"{subject_id}_{condition}_{band}_intensity_time_series.csv")
                combined_df.to_csv(
                    output_filename, index=False, float_format='%.5e')
                print(
                    f"Saved {condition}_{band} combined time series to {output_filename}")

                # Clear memory
                del stc, combined_df, selected_time_series

            except Exception as e:
                print(
                    f"Error processing {condition}_{band} for {subject_id}: {e}")
                import traceback
                traceback.print_exc()


def visualize_source_space_interactive(subject_data, transform_results, output_dir):
    """
    Creates an interactive 3D visualization of the source space, atlas nodes, 
    selected vertices, and electrode positions using Plotly.

    Parameters:
        subject_data: Dictionary containing subject information and paths
        transform_results: Results from transform_atlas_coordinates_and_select_vertices
        output_dir: Directory to save the HTML file
    """
    import os
    import numpy as np
    import mne
    import plotly.graph_objects as go

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract data from transform_results
    atlas_coords_head_m = transform_results['atlas_coords_head_m']
    selected_vertices = transform_results['selected_vertices']
    forward_cords = transform_results['forward_cords']
    used_indices = transform_results['used_indices']

    # Get electrode positions from raw file
    raw = mne.io.read_raw_fif(subject_data["EC_saved_file"])
    chan_pos = raw.get_montage().get_positions()
    electrode_names = list(chan_pos["ch_pos"].keys())

    # Extract electrode coordinates
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

    # Get coordinates of selected vertices
    selected_indices = [np.where(used_indices == vertex)[0][0]
                        for vertex in selected_vertices.values()]
    selected_coords = forward_cords[selected_indices]

    # Create the figure
    fig = go.Figure()

    # Add forward solution vertices (source space)
    fig.add_trace(go.Scatter3d(
        x=forward_cords[:, 0],
        y=forward_cords[:, 1],
        z=forward_cords[:, 2],
        mode='markers',
        marker=dict(color='blue', size=2, opacity=0.4),
        name='Source Space'
    ))

    # Add atlas nodes
    fig.add_trace(go.Scatter3d(
        x=atlas_coords_head_m[:, 0],
        y=atlas_coords_head_m[:, 1],
        z=atlas_coords_head_m[:, 2],
        mode='markers+text',
        marker=dict(color='red', size=6, opacity=0.9),
        text=[f"{i}" for i in range(len(atlas_coords_head_m))],
        textposition="top center",
        name='Atlas Nodes'
    ))

    # Add selected vertices
    region_names = list(selected_vertices.keys())
    fig.add_trace(go.Scatter3d(
        x=selected_coords[:, 0],
        y=selected_coords[:, 1],
        z=selected_coords[:, 2],
        mode='markers+text',
        marker=dict(color='green', size=6, opacity=0.9),
        text=region_names,
        textposition="top center",
        name='Selected Vertices'
    ))

    # Add electrode positions
    fig.add_trace(go.Scatter3d(
        x=chan_x,
        y=chan_y,
        z=chan_z,
        mode='markers+text',
        marker=dict(color='orange', size=5, symbol='diamond', opacity=0.9),
        text=electrode_names,
        textposition="top center",
        hoverinfo="text",
        name='Electrodes'
    ))

    # Update layout
    fig.update_layout(
        title=f"Source Space Visualization for {subject_data['subject_id']}",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode='data'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Save the figure as HTML
    output_file = os.path.join(
        output_dir, f"{subject_data['subject_id']}_source_space_visualization.html")
    fig.write_html(output_file)

    print(f"Saved interactive visualization to {output_file}")
    return fig

# =#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=##=#=#


PLOTS_DIR = os.path.join(SCRIPT_DIR, "Plots")
FINAL_SOURCE_SPACE_DIR = os.path.join(PLOTS_DIR, "3D_Final_Source_Space")
os.makedirs(FINAL_SOURCE_SPACE_DIR, exist_ok=True)

subjects_dict = get_subjects(MAIN_DIR)

for subject in subjects_dict:
    print(f"Analyzing: {subject}")
    transform_results = transform_atlas_coordinates_and_select_vertices(
        subjects_dict[subject])

    # Create interactive visualization
    visualize_source_space_interactive(
        subjects_dict[subject],
        transform_results,
        FINAL_SOURCE_SPACE_DIR
    )

    compute_source_time_courses_for_subject(
        subjects_dict[subject],
        transform_results['selected_vertices'],
        CSV_DIR
    )
