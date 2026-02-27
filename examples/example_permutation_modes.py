"""
Example: Activity Flow Permutation Tests with Different Modes.

Demonstrates both permutation strategies:
1. Full permutation: Shuffles all FC values
2. Column-wise permutation: Preserves target vertex properties
"""

import numpy as np
import pickle
import actflow_permutation
import rsm_computation

# ============================================================================
# Configuration
# ============================================================================

# Paths (example - adjust to your setup)
OUTPUT_DIR = '/home/ln275/f_mc1689_1/multitask_generalization/derivatives/actflow/'
FC_DIR = '/home/ln275/f_mc1689_1/multitask_generalization/data/derivatives/FC_new/'
RSM_DIR = '/home/ln275/f_mc1689_1/MDTB/derivatives/RSM_ActFlow/'
GLASSER_FILE = '/home/ln275/f_mc1689_1/multitask_generalization/docs/files/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
GLASSER_PARCELS_DIR = '/home/ln275/f_mc1689_1/multitask_generalization/docs/files/dilated_glasser_parcel_dscalar_files/'

SUBJECT_ID = '02'
N_PERMUTATIONS = 100

# ============================================================================
# Example: Compare Permutation Modes
# ============================================================================

print("Comparing Permutation Modes")
print("=" * 70)

# Create example FC matrix to demonstrate difference
print("\nDemonstration with example FC matrix:")
fc_example = np.random.randn(10, 20)  # 10 target vertices, 20 source vertices

print(f"Original FC shape: {fc_example.shape}")
print(f"Original target in-degrees (sum across sources):")
original_indegrees = np.sum(fc_example, axis=1)
print(f"  {original_indegrees[:5]}")  # Show first 5

# Full permutation
fc_full_perm = actflow_permutation.permute_fc_matrix(fc_example, permute_mode='full')
full_perm_indegrees = np.sum(fc_full_perm, axis=1)
print(f"\nFull permutation - target in-degrees:")
print(f"  {full_perm_indegrees[:5]}")
print(f"  ✗ Changed! (Expected - all values shuffled)")

# Column-wise permutation
fc_col_perm = actflow_permutation.permute_fc_matrix(fc_example, permute_mode='columns')
col_perm_indegrees = np.sum(fc_col_perm, axis=1)
print(f"\nColumn-wise permutation - target in-degrees:")
print(f"  {col_perm_indegrees[:5]}")
print(f"  ✓ Preserved! (Each column shuffled independently)")

print("\n" + "=" * 70)

# ============================================================================
# Example: Run Both Permutation Modes (Commented - Computationally Heavy)
# ============================================================================

print("\nExample: Running both permutation modes")
print("(Uncomment to run - computationally intensive)")
print("=" * 70)

example_code = """
# Load required data (same as actflow_prediction examples)
# task_betas = ...
# fc_dict = ...
# parcel_fc = ...
# observed_rsms = ...
# glasser = ...

# Mode 1: Full permutation (null hypothesis: any connectivity structure)
results_full = actflow_permutation.permutation_test_single_subject(
    subject_id=SUBJECT_ID,
    task_betas=task_betas,
    fc_dict=fc_dict,
    parcel_fc=parcel_fc,
    observed_rsms=observed_rsms,
    glasser=glasser,
    glasser_parcels_dir=GLASSER_PARCELS_DIR,
    rsm_computation_module=rsm_computation,
    n_permutations=N_PERMUTATIONS,
    permute_mode='full',  # ← Full permutation
    output_dir=OUTPUT_DIR
)

# Mode 2: Column-wise permutation (null hypothesis: specific source arrangement doesn't matter)
results_columns = actflow_permutation.permutation_test_single_subject(
    subject_id=SUBJECT_ID,
    task_betas=task_betas,
    fc_dict=fc_dict,
    parcel_fc=parcel_fc,
    observed_rsms=observed_rsms,
    glasser=glasser,
    glasser_parcels_dir=GLASSER_PARCELS_DIR,
    rsm_computation_module=rsm_computation,
    n_permutations=N_PERMUTATIONS,
    permute_mode='columns',  # ← Column-wise permutation
    output_dir=OUTPUT_DIR
)

# Compare results
print("\\nComparing null distributions:")
print(f"Full permutation - mean d_trans_hat: {np.mean(results_full['permuted_d_trans_hat']):.3f}")
print(f"Column-wise permutation - mean d_trans_hat: {np.mean(results_columns['permuted_d_trans_hat']):.3f}")
"""

print(example_code)

print("\n" + "=" * 70)

# ============================================================================
# Interpretation Guide
# ============================================================================

print("\nInterpretation Guide:")
print("=" * 70)
print("""
Two Permutation Strategies:

1. FULL PERMUTATION (permute_mode='full'):
   - Shuffles ALL FC values randomly
   - Null hypothesis: Any connectivity pattern is equally likely
   - Tests: Does the overall FC structure matter?
   - Changes: Everything (target properties, source properties, structure)
   - Use when: Testing if connectivity itself provides information
   
2. COLUMN-WISE PERMUTATION (permute_mode='columns'):
   - Shuffles within each source (column) independently
   - Preserves: Target vertex in-degree and in-degree distribution
   - Null hypothesis: Specific source→target arrangement doesn't matter
   - Tests: Does the specific wiring pattern matter, controlling for target strength?
   - Use when: You want to control for basic connectivity properties
   
Why Column-wise Matters:
- Preserves target properties (degree, strength)
- More conservative test
- Tests if SPECIFIC connectivity pattern matters beyond overall strength
- Similar to controlling for node degree in network analysis

Typical Workflow:
1. Run full permutation → Establishes that connectivity matters
2. Run column-wise permutation → Tests if specific wiring pattern matters
3. Compare: If both significant, pattern matters beyond just connectivity strength

Example Interpretation:
- Full perm p < 0.01, Column perm p < 0.01:
  → Specific wiring pattern matters (strong evidence)
  
- Full perm p < 0.01, Column perm p > 0.05:
  → Connectivity strength matters, but specific pattern doesn't
  → Target properties drive results
  
- Full perm p > 0.05:
  → Connectivity doesn't provide useful information
""")

print("=" * 70)
print("Example completed!")
