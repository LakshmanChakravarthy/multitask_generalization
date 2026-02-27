"""
Example: Activity Flow Modeling Pipeline.

Demonstrates how to:
1. Load task betas and FC
2. Predict target activations using activity flow
3. Compute predicted RSMs
4. Compute predicted dimensionality
5. Compare observed vs predicted
"""

import numpy as np
import pickle
import actflow_prediction
import rsm_computation  # Reuse existing module!

# ============================================================================
# Configuration
# ============================================================================

# Paths
PROJ_DIR = '/home/ln275/f_mc1689_1/multitask_generalization/'
FC_DIR = PROJ_DIR + 'data/derivatives/FC_new/'
TASK_DIR = '/home/ln275/f_mc1689_1/MDTB/derivatives/postprocessing/betasByTaskCondition/'
OUTPUT_DIR = PROJ_DIR + 'derivatives/actflow/'
GLASSER_FILE = PROJ_DIR + 'docs/files/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
GLASSER_PARCELS_DIR = PROJ_DIR + 'docs/files/dilated_glasser_parcel_dscalar_files/'

# Subjects (only those with resting-state FC)
SUBJECT_IDS = ['02', '03', '06', '08', '10', '12', '14', '18', '20',
               '22', '24', '25', '26', '27', '28', '29', '30', '31']

# Task subset: 96 active visual tasks
TASK_SUBSET = np.array([
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,36,37,38,39,40,
    41,42,43,44,45,46,49,50,51,52,53,54,55,56,60,61,62,63,64,65,
    66,67,68,69,70,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,
    87,88,89,91,92,93,94,95,96,97,98,99,100,101,102,104,105,106,
    107,108,109,110,111,112,113,114,115,120,121,122,123,124,125
])

# ============================================================================
# Example 1: Single Subject Activity Flow - Complete Pipeline
# ============================================================================

print("Example 1: Activity Flow for Single Subject - Complete Pipeline")
print("=" * 70)

subject_id = '02'

# Step 1: Load task betas (same approach as RSM computation)
print(f"\nStep 1: Loading task betas for subject {subject_id}...")
print("  (Using same loading approach as RSM computation)")
task_betas = actflow_prediction.load_mdtb_task_betas(
    subject_id=subject_id,
    beta_dir=TASK_DIR,
    task_subset_indices=TASK_SUBSET,
    space='vertex'
)
print(f"  Task betas shape: {task_betas.shape}")
print(f"  (2 session groups, {TASK_SUBSET.shape[0]} conditions, n_vertices)")

# Step 2: Load vertex-wise FC (from vertexwise_fc_pcr.py output)
print(f"\nStep 2: Loading vertex-wise FC...")
fc_file = f'{FC_DIR}sub{subject_id}_vertFC_CVoptimal_nPCs.pkl'
with open(fc_file, 'rb') as f:
    fc_dict = pickle.load(f)
    fc_rsq = pickle.load(f)
    optimal_nPC = pickle.load(f)

print(f"  Optimal nPCs: {optimal_nPC}")
print(f"  Number of target regions: {len(fc_dict)}")

# Step 3: Load parcel-level FC (from graphical_lasso_cv.py output)
print(f"\nStep 3: Loading parcel-level FC...")
with open(f'{FC_DIR}allSubFC_parCorr.pkl', 'rb') as f:
    all_parcel_fc = pickle.load(f)

subject_idx = SUBJECT_IDS.index(subject_id)
parcel_fc = all_parcel_fc[subject_idx, :, :]
print(f"  Parcel FC shape: {parcel_fc.shape}")
print(f"  Sparsity: {np.sum(parcel_fc == 0) / parcel_fc.size:.1%} zeros")

# Step 4: Load Glasser parcellation
print(f"\nStep 4: Loading Glasser parcellation...")
import nibabel as nib
glasser = np.squeeze(nib.load(GLASSER_FILE).get_fdata())
print(f"  Glasser shape: {glasser.shape}")

# Step 5: Run complete activity flow pipeline
print(f"\nStep 5: Running activity flow predictions...")
results = actflow_prediction.process_subject_actflow(
    subject_id=subject_id,
    task_betas=task_betas,
    fc_dict=fc_dict,
    parcel_fc=parcel_fc,
    glasser=glasser,
    glasser_parcels_dir=GLASSER_PARCELS_DIR,
    rsm_computation_module=rsm_computation,  # Reuses existing functions!
    output_dir=OUTPUT_DIR
)

print("\n✓ Pipeline completed successfully!")
print(f"  - Predicted betas: {len(results['predicted_betas'])} regions")
print(f"  - Predicted RSMs shape: {results['predicted_rsms'].shape}")
print(f"  - Predicted dimensionality shape: {results['predicted_dimensionality'].shape}")

print("\nCompleted!")
print("=" * 70)

# ============================================================================
# Example 2: Compare Observed vs Predicted
# ============================================================================

print("\nExample 2: Comparing Observed vs Predicted")
print("=" * 70)

# Load observed RSMs (computed earlier)
print("Loading observed RSMs...")
with open(f'{OUTPUT_DIR}../RSM_ActFlow/{subject_id}_RSM_activeVisual.pkl', 'rb') as f:
    observed_rsms = pickle.load(f)

# Load predicted RSMs (from activity flow)
print("Loading predicted RSMs...")
with open(f'{OUTPUT_DIR}{subject_id}_predicted_RSM_actflow.pkl', 'rb') as f:
    predicted_rsms = pickle.load(f)

print(f"\nObserved RSMs shape: {observed_rsms.shape}")
print(f"Predicted RSMs shape: {predicted_rsms.shape}")

# Compute correlation between observed and predicted RSMs
print("\nComputing RSM correlations...")
rsm_correlations = np.zeros(360)

for parcel_idx in range(360):
    obs_flat = observed_rsms[parcel_idx].flatten()
    pred_flat = predicted_rsms[parcel_idx].flatten()
    rsm_correlations[parcel_idx] = np.corrcoef(obs_flat, pred_flat)[0, 1]

print(f"\nRSM Prediction Accuracy:")
print(f"  Mean correlation: {np.mean(rsm_correlations):.3f}")
print(f"  Std correlation: {np.std(rsm_correlations):.3f}")
print(f"  Range: [{np.min(rsm_correlations):.3f}, {np.max(rsm_correlations):.3f}]")

# Compare dimensionality
print("\nLoading dimensionality...")
with open(f'{OUTPUT_DIR}../RSM_ActFlow/allsub_representational_dimensionality.pkl', 'rb') as f:
    all_observed_dim = pickle.load(f)

observed_dim = all_observed_dim[subject_idx, :]

with open(f'{OUTPUT_DIR}{subject_id}_predicted_dimensionality_actflow.pkl', 'rb') as f:
    predicted_dim = pickle.load(f)

print(f"\nObserved dimensionality:")
print(f"  Mean: {np.mean(observed_dim):.2f}")
print(f"  Range: [{np.min(observed_dim):.2f}, {np.max(observed_dim):.2f}]")

print(f"\nPredicted dimensionality:")
print(f"  Mean: {np.mean(predicted_dim):.2f}")
print(f"  Range: [{np.min(predicted_dim):.2f}, {np.max(predicted_dim):.2f}]")

# Correlation
dim_corr = np.corrcoef(observed_dim, predicted_dim)[0, 1]
print(f"\nDimensionality correlation: {dim_corr:.3f}")

print("\nCompleted!")
print("=" * 70)

# ============================================================================
# Example 3: Visualize Predictions
# ============================================================================

print("\nExample 3: Visualizing Predictions")
print("=" * 70)

import matplotlib.pyplot as plt

# Plot observed vs predicted for one region
parcel_idx = 100

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Observed RSM
im1 = axes[0].imshow(observed_rsms[parcel_idx], cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_title(f'Observed RSM\n(Parcel {parcel_idx + 1})')
axes[0].set_xlabel('Task Condition')
axes[0].set_ylabel('Task Condition')
plt.colorbar(im1, ax=axes[0])

# Predicted RSM
im2 = axes[1].imshow(predicted_rsms[parcel_idx], cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_title(f'Predicted RSM\n(Parcel {parcel_idx + 1})')
axes[1].set_xlabel('Task Condition')
axes[1].set_ylabel('Task Condition')
plt.colorbar(im2, ax=axes[1])

# Scatter plot
axes[2].scatter(observed_rsms[parcel_idx].flatten(), 
               predicted_rsms[parcel_idx].flatten(),
               alpha=0.3, s=5)
axes[2].plot([-1, 1], [-1, 1], 'r--', alpha=0.5)
axes[2].set_xlabel('Observed RSM')
axes[2].set_ylabel('Predicted RSM')
axes[2].set_title(f'r = {rsm_correlations[parcel_idx]:.3f}')
axes[2].set_xlim([-1, 1])
axes[2].set_ylim([-1, 1])
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}actflow_example_parcel{parcel_idx}.png', dpi=150)
print(f"\nSaved figure to {OUTPUT_DIR}actflow_example_parcel{parcel_idx}.png")

print("\n" + "=" * 70)
print("All examples completed!")

# ============================================================================
# Example 4: Double Cross-Validation
# ============================================================================

print("\nExample 4: Double Cross-Validation - Obs × Pred")
print("=" * 70)

print("\nRunning double cross-validated activity flow...")
double_cv_results = actflow_prediction.process_subject_actflow_double_cv(
    subject_id=subject_id,
    task_betas=task_betas,
    fc_dict=fc_dict,
    parcel_fc=parcel_fc,
    glasser=glasser,
    glasser_parcels_dir=GLASSER_PARCELS_DIR,
    rsm_computation_module=rsm_computation,
    output_dir=OUTPUT_DIR
)

print("\n✓ Double-CV completed!")
print(f"\nResults:")
print(f"  - Double-CV RSM_1 (Obs-Sess0 × Pred-Sess1): {double_cv_results['double_cv_rsms_1'].shape}")
print(f"  - Double-CV RSM_2 (Obs-Sess1 × Pred-Sess0): {double_cv_results['double_cv_rsms_2'].shape}")
print(f"  - Double-CV dimensionality (averaged): {double_cv_results['double_cv_dimensionality_avg'].shape}")

# Compare dimensionality estimates
obs_dim = observed_dim  # From Example 2
pred_dim = double_cv_results['predicted_dimensionality']
double_cv_dim = double_cv_results['double_cv_dimensionality_avg']

print(f"\nDimensionality comparison:")
print(f"  Observed mean: {np.mean(obs_dim):.2f}")
print(f"  Predicted (standard) mean: {np.mean(pred_dim):.2f}")
print(f"  Predicted (double-CV) mean: {np.mean(double_cv_dim):.2f}")

print(f"\nCorrelations:")
print(f"  Obs vs Pred (standard): {np.corrcoef(obs_dim, pred_dim)[0,1]:.3f}")
print(f"  Obs vs Pred (double-CV): {np.corrcoef(obs_dim, double_cv_dim)[0,1]:.3f}")

# Visualize double-CV RSMs
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

parcel_idx = 100

# Double-CV RSM_1
im1 = axes[0].imshow(double_cv_results['double_cv_rsms_1'][parcel_idx], 
                     cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_title(f'Double-CV RSM_1\n(Obs-Sess0 × Pred-Sess1)')
axes[0].set_xlabel('Task Condition')
axes[0].set_ylabel('Task Condition')
plt.colorbar(im1, ax=axes[0])

# Double-CV RSM_2
im2 = axes[1].imshow(double_cv_results['double_cv_rsms_2'][parcel_idx], 
                     cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_title(f'Double-CV RSM_2\n(Obs-Sess1 × Pred-Sess0)')
axes[1].set_xlabel('Task Condition')
axes[1].set_ylabel('Task Condition')
plt.colorbar(im2, ax=axes[1])

# Scatter: RSM_1 vs RSM_2
axes[2].scatter(double_cv_results['double_cv_rsms_1'][parcel_idx].flatten(),
               double_cv_results['double_cv_rsms_2'][parcel_idx].flatten(),
               alpha=0.3, s=5)
axes[2].plot([-1, 1], [-1, 1], 'r--', alpha=0.5)
axes[2].set_xlabel('RSM_1')
axes[2].set_ylabel('RSM_2')
rsm_corr = np.corrcoef(double_cv_results['double_cv_rsms_1'][parcel_idx].flatten(),
                       double_cv_results['double_cv_rsms_2'][parcel_idx].flatten())[0,1]
axes[2].set_title(f'RSM_1 vs RSM_2\nr = {rsm_corr:.3f}')
axes[2].set_xlim([-1, 1])
axes[2].set_ylim([-1, 1])
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}doubleCV_RSMs_parcel{parcel_idx}.png', dpi=150)
print(f"\nSaved double-CV RSM figure to {OUTPUT_DIR}")

print("\n" + "=" * 70)
print("All examples completed!")
