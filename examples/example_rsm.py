"""
Example: Compute representational similarity matrices (RSMs) for MDTB data.

This script demonstrates how to compute cross-validated RSMs using cosine
similarity between task condition activation patterns.
"""

import numpy as np
import rsm_computation

# ============================================================================
# Configuration
# ============================================================================

# Paths
BETA_DIR = '/home/ln275/f_mc1689_1/MDTB/derivatives/postprocessing/betasByTaskCondition/'
RSM_OUTPUT_DIR = '/home/ln275/f_mc1689_1/MDTB/derivatives/RSM_ActFlow/'
GLASSER_FILE = '/home/ln275/f_mc1689_1/MDTB/docs/files/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'

# Subjects
SUBJECT_IDS = ['02', '03', '04', '06', '08', '09', '10', '12', '14', '15',
               '18', '20', '22', '25', '27', '29', '31', '17', '19', '21',
               '24', '26', '28', '30']

# Parameters
SPACE = 'parcellated'  # or 'vertex'

# ============================================================================
# Example 1: Process single subject
# ============================================================================

print("Example 1: Processing single subject...")
print("=" * 70)

rsm_computation.process_subject(
    subject_id='02',
    beta_dir=BETA_DIR,
    output_dir=RSM_OUTPUT_DIR,
    glasser_file=GLASSER_FILE,
    space=SPACE
)

print("\nCompleted single subject!")
print("=" * 70)

# ============================================================================
# Example 2: Process all subjects
# ============================================================================

print("\nExample 2: Batch processing all subjects...")
print("=" * 70)

rsm_computation.process_all_subjects(
    subject_ids=SUBJECT_IDS,
    beta_dir=BETA_DIR,
    output_dir=RSM_OUTPUT_DIR,
    glasser_file=GLASSER_FILE,
    space=SPACE
)

print("\nCompleted batch processing!")
print("=" * 70)

# ============================================================================
# Example 3: Load and inspect RSMs
# ============================================================================

print("\nExample 3: Loading and inspecting RSMs...")
print("=" * 70)

# Load RSMs for all subjects
all_rsms = rsm_computation.load_all_subject_rsms(SUBJECT_IDS, RSM_OUTPUT_DIR)

print(f"\nAll subjects RSMs shape: {all_rsms.shape}")
print(f"  (n_subjects, n_parcels, n_conditions, n_conditions)")
print(f"  = ({len(SUBJECT_IDS)}, 360, 96, 96)")

# Inspect RSM for one subject, one region
subject_idx = 0
parcel_idx = 0

rsm = all_rsms[subject_idx, parcel_idx, :, :]

print(f"\nRSM for subject {SUBJECT_IDS[subject_idx]}, parcel {parcel_idx + 1}:")
print(f"  Shape: {rsm.shape}")
print(f"  Range: [{rsm.min():.3f}, {rsm.max():.3f}]")
print(f"  Mean: {rsm.mean():.3f}")
print(f"  Diagonal mean: {np.diag(rsm).mean():.3f}")

# Visualize RSM
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8, 7))
im = ax.imshow(rsm, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_title(f'RSM: Subject {SUBJECT_IDS[subject_idx]}, Parcel {parcel_idx + 1}')
ax.set_xlabel('Task Condition')
ax.set_ylabel('Task Condition')
plt.colorbar(im, ax=ax, label='Cosine Similarity')
plt.tight_layout()
plt.savefig(f'{RSM_OUTPUT_DIR}example_RSM_sub{SUBJECT_IDS[subject_idx]}_parcel{parcel_idx + 1}.png', dpi=150)
print(f"\nSaved example RSM plot to {RSM_OUTPUT_DIR}")

print("\n" + "=" * 70)
print("All examples completed!")

# ============================================================================
# Example 4: Compute representational dimensionality
# ============================================================================

print("\n" + "=" * 70)
print("Example 4: Computing representational dimensionality...")
print("=" * 70)

# Compute dimensionality for all subjects
all_dimensionalities = rsm_computation.compute_dimensionality_all_subjects(
    subject_ids=SUBJECT_IDS,
    rsm_dir=RSM_OUTPUT_DIR,
    output_dir=RSM_OUTPUT_DIR
)

print(f"\nDimensionality shape: {all_dimensionalities.shape}")
print(f"  (n_subjects, n_parcels) = ({len(SUBJECT_IDS)}, 360)")

# Summary statistics
print(f"\nDimensionality statistics (across all subjects/regions):")
print(f"  Mean: {np.mean(all_dimensionalities):.2f}")
print(f"  Std: {np.std(all_dimensionalities):.2f}")
print(f"  Range: [{np.min(all_dimensionalities):.2f}, {np.max(all_dimensionalities):.2f}]")

# Per-subject average
subject_means = np.mean(all_dimensionalities, axis=1)
print(f"\nPer-subject mean dimensionality:")
for sub_idx, subject_id in enumerate(SUBJECT_IDS[:5]):  # Show first 5
    print(f"  Subject {subject_id}: {subject_means[sub_idx]:.2f}")
print(f"  ...")

print("\n" + "=" * 70)
print("All examples completed!")
