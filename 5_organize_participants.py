import os
import csv
import shutil
import pandas as pd
from collections import Counter
from pathlib import Path
# Paths
demographics_file = "/Users/alpmac/Desktop/PerinelliFixed/demographics.csv"
combined_dir = "/Users/alpmac/PerinelliFixed/CSV"
output_dir = "/Users/alpmac/PerinelliFixed/Grouped"
# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read demographics data
demographics = {}
gender_map = {"1": "Female", "2": "Male"}

with open(demographics_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        subject_id = row['ID']
        gender = gender_map.get(row['Gender_ 1=female_2=male'], "Unknown")
        age = row['Age']
        demographics[subject_id] = {'gender': gender, 'age': age}

# Get list of existing participants in combined directory
existing_subjects = [d for d in os.listdir(combined_dir) if os.path.isdir(
    os.path.join(combined_dir, d)) and d.startswith('sub-')]

# Create a DataFrame for analysis
data = []
for subject in existing_subjects:
    if subject in demographics:
        data.append({
            'subject_id': subject,
            'gender': demographics[subject]['gender'],
            'age': demographics[subject]['age']
        })

df = pd.DataFrame(data)

# Count frequency of each gender-age combination
gender_age_counts = Counter([(row['gender'], row['age'])
                            for _, row in df.iterrows()])

# Print the top combinations
print("Most common gender-age combinations:")
for (gender, age), count in gender_age_counts.most_common(10):
    print(f"{gender}, {age}: {count}")

# Define the 6 groups we want to create
groups = [
    ('Female', '20-25'),
    ('Male', '20-25'),
    ('Female', '60-65'),
    ('Male', '60-65'),
    ('Female', '65-70'),
    ('Male', '65-70'),
    ('Female', '70-75'),
    ('Male', '70-75'),
    ('Female', '75-80'),
    ('Male', '75-80'),
]

# Create symbolic links for each subject in the appropriate group folder
for group_gender, group_age in groups:
    # Create group directory
    group_dir = os.path.join(output_dir, f"{group_gender}_{group_age}")
    os.makedirs(group_dir, exist_ok=True)

    # Find matching subjects
    matching_subjects = [subject for subject in existing_subjects
                         if subject in demographics
                         and demographics[subject]['gender'] == group_gender
                         and demographics[subject]['age'] == group_age]

    print(
        f"\nCreating links for {group_gender}_{group_age} ({len(matching_subjects)} subjects):")

    # Create symbolic links
    for subject in matching_subjects:
        src = os.path.join(combined_dir, subject)
        dst = os.path.join(group_dir, subject)

        # Create symbolic link if it doesn't exist
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
                print(f"  Created link for {subject}")
            except Exception as e:
                print(f"  Error creating link for {subject}: {e}")

print("\nDone! Organized participants into gender-age group folders.")
