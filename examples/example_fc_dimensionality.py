"""
Example: Compute FC dimensionality from vertex-wise functional connectivity.

This script demonstrates how to compute dimensionality of FC patterns using
singular value participation ratio (SVPR).
"""

import numpy as np
import fc_dimensionality

# ============================================================================
# Configuration
# ============================================================================

# Paths
FC_DIR = '/home/ln275/f_mc1689_1/multitask_generalization/data/derivatives/FC_new/'
OUTPUT_DIR = '/home/ln275/f_mc1689_1/multitask_generalization/derivatives/dimensionality/'

# Subjects (resting-state FC subjects)
SUBJECT_IDS = ['02', '03', '06', '08', '10', '12', '14', '18', '20',
               '22', '24', '25', '26', '27', '28', '29', '30', '31']

# ============================================================================
# Example 1: Compute FC dimensionality for all subjects
# ============================================================================

print("Example 1: Computing FC dimensionality for all subjects...")
print("=" * 70)

all_fc_dimensionalities = fc_dimensionality.compute_fc_dimensionality_all_subjects(
    subject_ids=SUBJECT_IDS,
    fc_dir=FC_DIR,
    output_dir=OUTPUT_DIR
)

print(f"\nFC Dimensionality shape: {all_fc_dimensionalities.shape}")
print(f"  (n_subjects, n_parcels) = ({len(SUBJECT_IDS)}, 360)")

# Summary statistics
print(f"\nFC Dimensionality statistics (across all subjects/regions):")
print(f"  Mean: {np.nanmean(all_fc_dimensionalities):.2f}")
print(f"  Std: {np.nanstd(all_fc_dimensionalities):.2f}")
print(f"  Range: [{np.nanmin(all_fc_dimensionalities):.2f}, {np.nanmax(all_fc_dimensionalities):.2f}]")

# Per-subject average
subject_means = np.nanmean(all_fc_dimensionalities, axis=1)
print(f"\nPer-subject mean FC dimensionality:")
for sub_idx, subject_id in enumerate(SUBJECT_IDS[:5]):  # Show first 5
    print(f"  Subject {subject_id}: {subject_means[sub_idx]:.2f}")
print(f"  ...")

print("\n" + "=" * 70)
print("Completed!")

# ============================================================================
# Example 2: Load and inspect results
# ============================================================================

print("\nExample 2: Loading and inspecting results...")
print("=" * 70)

# Load saved results
dimensionality_file = f'{OUTPUT_DIR}allsub_FC_dimensionality.pkl'
fc_dims = fc_dimensionality.load_fc_dimensionality(dimensionality_file)

print(f"\nLoaded FC dimensionality shape: {fc_dims.shape}")

# Distribution across parcels (averaged across subjects)
parcel_means = np.nanmean(fc_dims, axis=0)

print(f"\nFC dimensionality distribution across parcels:")
print(f"  10th percentile: {np.nanpercentile(parcel_means, 10):.2f}")
print(f"  25th percentile: {np.nanpercentile(parcel_means, 25):.2f}")
print(f"  Median: {np.nanmedian(parcel_means):.2f}")
print(f"  75th percentile: {np.nanpercentile(parcel_means, 75):.2f}")
print(f"  90th percentile: {np.nanpercentile(parcel_means, 90):.2f}")

# Find regions with lowest/highest dimensionality
lowest_parcels = np.argsort(parcel_means)[:5]
highest_parcels = np.argsort(parcel_means)[-5:]

print(f"\nParcels with LOWEST FC dimensionality (most low-dimensional):")
for parcel_idx in lowest_parcels:
    print(f"  Parcel {parcel_idx + 1}: {parcel_means[parcel_idx]:.2f}")

print(f"\nParcels with HIGHEST FC dimensionality (least low-dimensional):")
for parcel_idx in highest_parcels:
    print(f"  Parcel {parcel_idx + 1}: {parcel_means[parcel_idx]:.2f}")

print("\n" + "=" * 70)
print("All examples completed!")
