"""
Example: Compute vertex-wise FC for a single subject.

This script demonstrates the complete pipeline for estimating vertex-wise
functional connectivity using PC regression with CV-optimized hyperparameters.
"""

import numpy as np
import nibabel as nib
import data_utils
import vertexwise_fc_pcr

# ============================================================================
# Configuration
# ============================================================================

# Paths
PROJ_DIR = '/home/ln275/f_mc1689_1/multitask_generalization/'
FC_DIR = PROJ_DIR + 'data/derivatives/FC_new/'
OUTPUT_DIR = FC_DIR + 'vertexwiseFC/'
HELPFILES_DIR = PROJ_DIR + 'docs/files/'
GLASSER_PARCELS_DIR = HELPFILES_DIR + 'dilated_glasser_parcel_dscalar_files/'

# Subject to process (use as example)
SUBJECT_ID = '02'

# Parameters
N_PARCELS = 360
N_VERTICES = 59412

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")

# Load resting-state fMRI data
sub_vert_data = data_utils.load_rsfmri(SUBJECT_ID, space='vertex')
sub_parcel_data = data_utils.load_rsfmri(SUBJECT_ID, space='parcellated')

print(f"Vertex data shape: {sub_vert_data.shape}")
print(f"Parcel data shape: {sub_parcel_data.shape}")

# Load Glasser parcellation
glasser = data_utils.load_glasser_parcellation()
print(f"Glasser parcellation shape: {glasser.shape}")

# Load parcel-level FC (graphical lasso output)
# This is the FC "skeleton" that defines which connections to estimate at vertex level
all_subjects_parcel_fc = data_utils.load_all_subjects_parcel_fc(FC_DIR)
# Get this subject's parcel FC (assuming subjects are in order)
subject_idx = ['02','03','06','08','10','12','14','18','20',
               '22','24','25','26','27','28','29','30','31'].index(SUBJECT_ID)
parcel_fc = all_subjects_parcel_fc[subject_idx, :, :]

print(f"Parcel FC shape: {parcel_fc.shape}")
print(f"Parcel FC sparsity: {np.sum(parcel_fc == 0) / parcel_fc.size:.2%} zeros")

# ============================================================================
# Run Vertex-wise FC Pipeline
# ============================================================================

print("\n" + "=" * 70)
print("RUNNING VERTEX-WISE FC ESTIMATION WITH CV-OPTIMIZED PCs")
print("=" * 70 + "\n")

vertexwise_fc_pcr.process_subject(
    subject_id=SUBJECT_ID,
    sub_vert_data=sub_vert_data,
    sub_parcel_data=sub_parcel_data,
    parcel_fc=parcel_fc,
    glasser=glasser,
    glasser_parcels_dir=GLASSER_PARCELS_DIR,
    output_dir=OUTPUT_DIR,
    n_parcels=N_PARCELS
)

print("\n" + "=" * 70)
print("COMPLETED!")
print("=" * 70)

# ============================================================================
# Verify Output
# ============================================================================

print("\nVerifying saved output...")

import pickle

output_file = f'{OUTPUT_DIR}/sub{SUBJECT_ID}_vertFC_CVoptimal_nPCs.pkl'
with open(output_file, 'rb') as f:
    vertFC_dict = pickle.load(f)
    vertFC_rsq_arr = pickle.load(f)
    optimal_nPC = pickle.load(f)

print(f"\nResults summary:")
print(f"- Optimal nPCs: {optimal_nPC}")
print(f"- Number of target parcels: {len(vertFC_dict)}")
print(f"- Example FC shape (target parcel 0): {vertFC_dict[0].shape}")
print(f"- Mean R-squared across parcels: {np.mean(vertFC_rsq_arr):.4f}")
print(f"- R-squared range: [{np.min(vertFC_rsq_arr):.4f}, {np.max(vertFC_rsq_arr):.4f}]")
