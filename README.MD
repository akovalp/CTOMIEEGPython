# EEG Source Reconstruction and Analysis Pipeline

## Overview

This repository contains a complete pipeline for EEG source reconstruction and analysis, designed to analyze power shifts and connectivity changes in healthy aging during resting-state EEG, following the methodology from Perinelli et al. (2022). The pipeline performs source reconstruction using the `osl_ephys` package, applies frequency band filtering, computes noise covariance matrices, and analyzes power distribution across brain regions.

The pipeline focuses on analyzing EEG data in three frequency bands (theta: 4-8 Hz, alpha: 8-14 Hz, beta: 14-30 Hz) across four brain areas (occipital, parietal, temporal, frontal) for Eyes Closed (EC) and Eyes Open (EO) conditions. It allows comparison between young and older adult groups.

## Pipeline Components

The repository consists of six main scripts that should be run sequentially:

1. **1_coreg_forward.py**: Handles electrode registration, computes surfaces, and creates forward models
2. **2_band_filter.py**: Applies band-pass filters to the EEG data (theta, alpha, beta bands)
3. **3_noise_covariance.py**: Computes noise covariance matrices for each subject, condition, and frequency band
4. **4_compute_source.py**: Transforms atlas coordinates and computes source time courses
5. **5_organize_participants.py**: Organizes participants into demographic groups based on age and gender
6. **6_full_analysis.py**: Performs final power analysis across brain regions and age groups

## Required Dependencies

- MNE-Python
- NumPy, Pandas
- OSL-EPHYS (Oxford Software Library for Electrophysiology)
- Matplotlib, Seaborn, Plotly
- Pingouin (for statistical analysis)
- tqdm (for progress bars)

## Expected Folder Structure

Before running the pipeline, your data should be organized as follows:

```
Participants/
├── sub-XXXXXX/
│   ├── EEG_Before_Chan/
│   │   ├── EC/
│   │   │   └── sub-XXXXXX_EC.set
│   │   └── EO/
│   │       └── sub-XXXXXX_EO.set
│   ├── EEG_After_Chan/   # Initially empty, will be populated by the pipeline
│   │   ├── EC/
│   │   └── EO/
│   ├── Localizer/
│   │   └── sub-XXXXXX.mat
│   ├── Polhemus/
│   │   └── [polhemus files]
│   ├── Recon/    # Initially empty, will be populated by the pipeline
│   └── SMRI/
│       └── sub-XXXXXX_ses-01_inv-2_mp2rage.nii.gz
```

**Important Notes:**
- The `Recon` folder must be empty at the beginning as it will be populated by the pipeline.
- The `EEG_After_Chan` folder should exist with empty `EC` and `EO` subfolders, which will be populated during processing.

## Running the Pipeline

### 1. Coregistration and Forward Modeling

```bash
python 1_coreg_forward.py
```

This script performs the following operations:
- Reads raw EEG data and localizer information
- Performs Polhemus registration to align electrode positions
- Computes head surfaces from structural MRI data
- Creates coregistration between EEG and MRI coordinates
- Computes forward models for source reconstruction
- Generates 2D and 3D visualizations of electrode positions

**Output:**
- Transformed EEG files with correct montage information
- Forward models for source reconstruction
- Visualization plots of electrode positions

### 2. Band-pass Filtering

```bash
python 2_band_filter.py
```

This script applies band-pass filters to the EEG data:
- Theta band (4-8 Hz)
- Alpha band (8-14 Hz)
- Beta band (14-30 Hz)

**Output:**
- Filtered EEG files for each frequency band and condition (EC/EO)

### 3. Noise Covariance Computation

```bash
python 3_noise_covariance.py
```

This script computes noise covariance matrices for each subject, condition, and frequency band, which are required for inverse solutions in source reconstruction.

**Output:**
- Noise covariance matrices saved in the `Noise_Covariances` directory

### 4. Source Reconstruction

```bash
python 4_compute_source.py
```

This script performs source reconstruction:
- Transforms atlas coordinates to head space
- Selects appropriate vertices in the source space
- Computes source time courses for selected regions
- Creates visualizations of the source space

**Output:**
- CSV files containing source time series for each region, condition, and frequency band
- 3D visualizations of the source space

### 5. Participant Organization

```bash
python 5_organize_participants.py
```

This script organizes participants into demographic groups based on age and gender:
- Young groups (20-25 years)
- Older groups (60-65, 65-70, 70-75, 75-80 years)
- Separated by gender (Male/Female)

**Output:**
- Organized participant directories in the `Grouped` folder

### 6. Full Analysis

```bash
python 6_full_analysis.py
```

This script performs the final analysis:
- Calculates power for each brain region, condition, and frequency band
- Creates analysis dataframes for statistical comparison
- Generates CSV files with power values
- Compares young vs. older adult groups

**Output:**
- CSV files containing power results
- Statistical analysis results

## Atlas Information

The pipeline uses atlas regions defined in the `selected_atlas_areas.txt` file, with 24 nodes across four brain areas:

1. **Occipital**: V1, V6A, V4 (both left and right)
2. **Parietal**: PFm, PF, STV (both left and right)
3. **Temporal**: STGa, TE1a, TA2 (both left and right)
4. **Frontal**: 10d, 10pp, p10p (both left and right)

## Demographics

Demographics information should be provided in a `demographics.csv` file with columns:
- ID
- Gender (1=female, 2=male)
- Age (categorical: 20-25, 60-65, 65-70, 70-75, 75-80)



## Sample Analysis

Sample analysis pipeline that will output graphs related to power differences can be found in `analysis.ipynb`

## References

This pipeline implements the methodology from:
Perinelli, A., Assecondi, S., Tagliabue, C. F., & Mazza, V. (2022). Power shift and connectivity changes in healthy aging during resting-state EEG. NeuroImage, 256, 119247.

## Notes on OSL-EPHYS

The pipeline relies on the OSL-EPHYS package for source reconstruction, which provides tools for:
- Creating and working with head models
- Computing source reconstruction using eLORETA
- Handling coordinate transformations between different spaces

Make sure OSL-EPHYS is properly installed and configured before running the pipeline.