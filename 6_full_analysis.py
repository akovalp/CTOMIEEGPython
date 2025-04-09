import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns
import os
import time
from tqdm import tqdm
# SECTION : DEFINING REGION STRUCTURE
column_mapping = {
    # Keep Time column as is
    'Time': 'Time',

    # Occipital Regions
    'L_V1_ROI L': 'V1_Left',
    'R_V1_ROI R': 'V1_Right',
    'L_V6A_ROI L': 'V6A_Left',
    'R_V6A_ROI R': 'V6A_Right',
    'L_V4_ROI L': 'V4_Left',
    'R_V4_ROI R': 'V4_Right',

    # REVIEW: L_5L, L_6mp, L_4 seem to be other occipital regions were not used in the analysis
    'L_5L_ROI L': 'Area5L_Left',
    'R_5L_ROI R': 'Area5L_Right',
    'L_6mp_ROI L': 'Area6mp_Left',
    'R_6mp_ROI R': 'Area6mp_Right',
    'L_4_ROI L': 'Area4_Left',
    'R_4_ROI R': 'Area4_Right',

    # Parietal Regions
    'L_PFm_ROI L': 'PFm_Left',
    'R_PFm_ROI R': 'PFm_Right',
    'L_PF_ROI L': 'PF_Left',
    'R_PF_ROI R': 'PF_Right',
    'L_STV_ROI L': 'STV_Left',
    'R_STV_ROI R': 'STV_Right',

    # Temporal Regions
    'L_STGa_ROI L': 'STGa_Left',
    'R_STGa_ROI R': 'STGa_Right',
    'L_TE1a_ROI L': 'TE1a_Left',
    'R_TE1a_ROI R': 'TE1a_Right',
    'L_TA2_ROI L': 'TA2_Left',
    'R_TA2_ROI R': 'TA2_Right',

    # Frontal Regions
    'L_10d_ROI L': '10d_Left',
    'R_10d_ROI R': '10d_Right',
    'L_10pp_ROI L': '10pp_Left',
    'R_10pp_ROI R': '10pp_Right',
    'L_p10p_ROI L': 'p10p_Left',
    'R_p10p_ROI R': 'p10p_Right'
}
brain_regions = {
    # REVIEW - I have removed the other occipital regions that are not in the description
    'occipital': ['V1_Left', 'V1_Right', 'V6A_Left', 'V6A_Right', 'V4_Left', 'V4_Right'],
    'parietal': ['PFm_Left', 'PFm_Right', 'PF_Left', 'PF_Right', 'STV_Left', 'STV_Right'],
    'temporal': ['STGa_Left', 'STGa_Right', 'TE1a_Left', 'TE1a_Right', 'TA2_Left', 'TA2_Right'],
    'frontal': ['10d_Left', '10d_Right', '10pp_Left', '10pp_Right', 'p10p_Left', 'p10p_Right']
}
# Print the number of nodes in each region
for region, nodes in brain_regions.items():
    print(f"{region}: {len(nodes)}")

# Hemisphere-based categorization
hemisphere_regions = {
    'left': [col for col in column_mapping.values() if col != 'Time' and col.endswith('_Left')],
    'right': [col for col in column_mapping.values() if col != 'Time' and col.endswith('_Right')]
}
# Print the number of nodes in each hemisphere
print(f"Left hemisphere: {len(hemisphere_regions['left'])}")
print(f"Right hemisphere: {len(hemisphere_regions['right'])}")

# List of participants to exclude from analysis
excluded_participants = [
    '010052',
    '010061',
    '010065',
    '010069',
    '010070',
    '010073',
    '010074',
    '010207',
    '010287',
    '010288',
    '010305'
]

young_groups = ['Male_20-25', 'Female_20-25']
older_groups = [
    'Male_60-65',
    'Female_60-65',
    'Male_65-70',
    'Female_65-70',
    'Male_70-75',
    'Female_70-75',
    'Female_75-80',
    'Male_75-80'
]
base_dir = '/Users/alpmac/PerinelliFixed/Grouped'


def get_all_subjects(group):  # Standard function to get all subjects in a given group
    """
    This function returns a list of all subjects in a given group so we have two (main) groups to compare (young and older but also female and male there is an imbalance in the number of subjects in each group )
    parameters:
    """
    group_dir = os.path.join(base_dir, group)

    # Check if the group directory exists
    if not os.path.exists(group_dir):
        print(f"Warning: Group directory not found: {group_dir}")
        return []

    # List all subdirectories (subjects) in the group directory
    subjects = [d for d in os.listdir(group_dir)
                if os.path.isdir(os.path.join(group_dir, d)) and d.startswith('sub-')]

    # Filter out excluded participants
    filtered_subjects = []
    excluded_count = 0
    excluded_subjects = []

    for subject in subjects:
        # Extract the ID portion from 'sub-XXXXXX'
        subject_id = subject.split('-')[1]
        if subject_id not in excluded_participants:
            filtered_subjects.append(subject)
        else:
            excluded_count += 1
            excluded_subjects.append(subject)
            print(
                f"EXCLUDED PARTICIPANT: {subject} (ID: {subject_id}) from group {group}")

    if excluded_count > 0:
        print(
            f"Total participants excluded from {group}: {excluded_count}/{len(subjects)}")
        print(
            f"Excluded subjects from {group}: {', '.join(excluded_subjects)}")

    return filtered_subjects


def load_participant_data(group, subject_id, condition, band):
    """
    Load participant data for a specific condition and frequency band. These are intensity time series files. for each band and condition there is a csv file.
    """
    file_path = os.path.join(base_dir, group, subject_id,
                             f"{subject_id}_{condition}_{band}_intensity_time_series.csv")

    # Check file existence
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        df = df.rename(columns=column_mapping)
        return df
    except pd.errors.ParserError as e:
        print(f"Parser error loading {file_path}: {str(e)}")
        # Try with Python engine for problematic files
        try:
            df = pd.read_csv(file_path, engine='python')
            df = df.rename(columns=column_mapping)
            return df
        except Exception as e2:
            print(f"Failed with Python engine too: {str(e2)}")
            return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None


def process_subject_data(group, subject_id, conditions=['EC', 'EO']):
    """
    Process data for a single subject, both EC and EO conditions

    Parameters:
    -----------
    group : str
        The demographic group folder name
    subject_id : str
        Subject identifier
    conditions : list
        List of conditions to process (default: ['EC', 'EO'])

    Returns:
    --------
    dict
        Nested dictionary with structure:
        {condition: {region: {band: power}}}
    """
    results = {'group': group, 'subject_id': subject_id}

    for condition in conditions:
        condition_results = {}

        # Process each frequency band
        for band in ['alpha', 'beta', 'theta']:
            # Load data for this condition and band
            df = load_participant_data(group, subject_id, condition, band)

            if df is None:
                continue

            # Process each brain region
            for region_name, columns in brain_regions.items():
                if region_name not in condition_results:
                    condition_results[region_name] = {}

                # Calculate average power for each region
                try:
                    print(columns)
                    # region_data = df[columns][45000:90000].mean(axis=0).mean()
                    region_data = np.mean(
                        np.square(1e11*np.array((df[columns][45000:90000]))))
                    # Store results
                    condition_results[region_name][band] = region_data
                except Exception as e:
                    print(
                        f"Error calculating power for {region_name}, {band}: {str(e)}")

        results[condition] = condition_results
        print(results)

    return results


def process_all_subjects(groups_list):
    """
    Process all subjects in the given groups

    Parameters:
    -----------
    groups_list : list
        List of demographic group names

    Returns:
    --------
    list
        List of dictionaries with processed data for each subject
    """
    all_results = []

    # Process each group
    for group in groups_list:
        subjects = get_all_subjects(group)

        if not subjects:
            print(f"No subjects found for group {group}")
            continue

        # Process each subject in this group
        for i, subject_id in enumerate(subjects):
            print(
                f"Processing subject {i+1}/{len(subjects)} in group {group}: {subject_id}")

            try:
                subject_results = process_subject_data(group, subject_id)
                all_results.append(subject_results)
            except Exception as e:
                print(f"Error processing subject {subject_id}: {str(e)}")

    print(f"Processed data for {len(all_results)} subjects total")
    return all_results


def create_analysis_dataframe(all_results):
    """
    Convert processed results to a pandas DataFrame for statistical analysis

    Parameters:
    -----------
    all_results : list
        List of dictionaries with processed data

    Returns:
    --------
    pandas.DataFrame
        Long-format DataFrame with columns:
        [subject_id, group, age_group, condition, region, band, power]
    """
    rows = []

    for subject_data in all_results:
        subject_id = subject_data['subject_id']
        group = subject_data['group']

        # Determine age group
        age_group = 'young' if group in young_groups else 'older'

        # Process each condition
        for condition in ['EC', 'EO']:
            if condition not in subject_data:
                print(f"No {condition} data for subject {subject_id}")
                continue

            # Process each region
            for region, band_data in subject_data[condition].items():
                # Process each frequency band
                for band, power in band_data.items():
                    # Create a row for this data point
                    row = {
                        'subject_id': subject_id,
                        'group': group,
                        'age_group': age_group,
                        'condition': condition,
                        'region': region,
                        'band': band,
                        'power': power
                    }
                    rows.append(row)

    # Create DataFrame from the rows
    df = pd.DataFrame(rows)
    print(f"Created dataframe with {len(df)} rows")

    return df


def verify_excluded_participants():  # Not Important function
    """
    Verify that all excluded participants exist in the dataset and identify where they are located.
    This function searches through all directories to find each excluded participant.
    """
    print("=" * 80)
    print("VERIFYING EXCLUDED PARTICIPANTS")
    print("=" * 80)

    # Get all directories in the base_dir
    all_groups = []
    try:
        all_groups = [d for d in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, d))]
    except Exception as e:
        print(f"Error reading base directory {base_dir}: {str(e)}")
        return

    # Track where each excluded participant is found
    found_participants = {participant_id: []
                          for participant_id in excluded_participants}

    # Check each group directory for the excluded participants
    for group in all_groups:
        group_dir = os.path.join(base_dir, group)
        try:
            subjects = [d for d in os.listdir(group_dir)
                        if os.path.isdir(os.path.join(group_dir, d)) and d.startswith('sub-')]

            for subject in subjects:
                subject_id = subject.split('-')[1]
                if subject_id in excluded_participants:
                    found_participants[subject_id].append((group, subject))
        except Exception as e:
            print(f"Error checking group directory {group_dir}: {str(e)}")

    # Report results
    print("Verification results for excluded participants:")
    not_found = []

    for participant_id, locations in found_participants.items():
        if locations:
            print(
                f"Participant ID {participant_id} found in {len(locations)} location(s):")
            for group, subject in locations:
                print(f"  - {group}/{subject}")
        else:
            print(f"Participant ID {participant_id} NOT FOUND in any group")
            not_found.append(participant_id)

    if not_found:
        print(
            f"WARNING: {len(not_found)}/{len(excluded_participants)} excluded participants were not found:")
        for participant_id in not_found:
            print(f"  - {participant_id}")
    else:
        print(
            f"All {len(excluded_participants)} excluded participants were found in the dataset")

    print("=" * 80)


def main():
    print("Starting EEG analysis")

    # First, verify that all excluded participants exist in the dataset
    verify_excluded_participants()

    # Get all groups from the base directory to ensure we catch exclusions in all groups
    all_groups = [d for d in os.listdir(base_dir)
                  if os.path.isdir(os.path.join(base_dir, d))]

    # Keep track of exclusions for the final report
    excluded_tracking = {group: [] for group in all_groups}

    # Scan all groups to identify which excluded participants were found in each group
    print("Scanning all groups for excluded participants")
    for group in all_groups:
        group_dir = os.path.join(base_dir, group)
        if not os.path.exists(group_dir):
            continue

        all_subjects = [d for d in os.listdir(group_dir)
                        if os.path.isdir(os.path.join(group_dir, d)) and d.startswith('sub-')]

        for subject in all_subjects:
            subject_id = subject.split('-')[1]
            if subject_id in excluded_participants:
                excluded_tracking[group].append((subject, subject_id))

    # Display excluded participants found in each group
    print("Participants marked for exclusion by group:")
    for group in all_groups:
        if excluded_tracking[group]:
            print(
                f"Group {group}: {len(excluded_tracking[group])} participants to exclude")
            for subject, subject_id in excluded_tracking[group]:
                print(f"  - {subject} (ID: {subject_id})")

    # Load data for young and older participants
    print("Processing young participants")
    young_results = process_all_subjects(young_groups)
    print(f"Processed {len(young_results)} young subjects")

    print("Processing older participants")
    old_results = process_all_subjects(older_groups)
    print(f"Processed {len(old_results)} older subjects")

    # Create combined dataframes
    print("Creating analysis dataframes")
    young_df = create_analysis_dataframe(young_results)
    old_df = create_analysis_dataframe(old_results)

    # Process and save separate CSVs for each combination of age_group, band, and condition
    print("Saving filtered CSV files")
    for age_group, df in [('young', young_df), ('older', old_df)]:
        for band in ['alpha', 'beta', 'theta']:
            for condition in ['EC', 'EO']:
                # Filter the dataframe for this specific combination
                filtered_df = df[(df['band'] == band) & (
                    df['condition'] == condition)]

                # Multiply power values by 10^11
                filtered_df = filtered_df.copy()
                filtered_df['power'] = filtered_df['power'] * 1

                # Save to CSV
                filename = f"{age_group}_{band}_{condition}.csv"
                filtered_df.to_csv(filename, index=False)
                print(f"Saved {filename} with {len(filtered_df)} records")

    # Also save the full dataframes for reference
    print("Saving complete datasets")
    young_df_scaled = young_df.copy()
    young_df_scaled['power'] = young_df_scaled['power'] * 1
    young_df_scaled.to_csv('young_results_all.csv', index=False)

    old_df_scaled = old_df.copy()
    old_df_scaled['power'] = old_df_scaled['power'] * 1
    old_df_scaled.to_csv('old_results_all.csv', index=False)

    # Print final exclusion summary
    print("=" * 80)
    print("SUMMARY OF EXCLUDED PARTICIPANTS")
    print("=" * 80)

    # First, summarize by participant ID
    participant_locations = {pid: [] for pid in excluded_participants}
    for group, excluded_list in excluded_tracking.items():
        for subject, subject_id in excluded_list:
            participant_locations[subject_id].append(group)

    print("Excluded participants by ID:")
    for pid, groups in participant_locations.items():
        if groups:
            print(
                f"Participant ID {pid} excluded from {len(groups)} group(s): {', '.join(groups)}")
        else:
            print(f"Participant ID {pid} was not found in any group")

    # Then, summarize by group
    total_excluded = 0
    print("\nExcluded participants by group:")
    for group, excluded_list in excluded_tracking.items():
        if excluded_list:
            print(f"Group: {group}")
            for subject, subject_id in excluded_list:
                print(f"  - {subject} (ID: {subject_id})")
            print(
                f"  Total: {len(excluded_list)} participants excluded from {group}")
            total_excluded += len(excluded_list)

    print(f"\nTOTAL PARTICIPANTS EXCLUDED: {total_excluded}")
    print("=" * 80)

    # Note if any excluded participants weren't found
    not_found = [pid for pid, groups in participant_locations.items()
                 if not groups]
    if not_found:
        print(
            f"WARNING: {len(not_found)} excluded IDs were not found in any group:")
        for pid in not_found:
            print(f"  - {pid}")

    print("Complete dataset saved to CSV files")
    print("EEG Analysis Script Completed Successfully")


if __name__ == "__main__":
    print("EEG Analysis Script Started")
    try:
        main()
    except Exception as e:
        print(f"Script failed with error: {str(e)}")
        raise
