"""
Example: Activity Flow Distance Metrics.

Demonstrates computation of transformation and prediction distances to evaluate
whether connectivity does meaningful transformation of source information.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import actflow_metrics

# ============================================================================
# Configuration
# ============================================================================

# Paths
OUTPUT_DIR = '/home/ln275/f_mc1689_1/multitask_generalization/derivatives/actflow/'
RSM_DIR = '/home/ln275/f_mc1689_1/MDTB/derivatives/RSM_ActFlow/'
FC_DIR = '/home/ln275/f_mc1689_1/multitask_generalization/data/derivatives/FC_new/'

# Subject
SUBJECT_ID = '02'
SUBJECT_IDS = ['02', '03', '06', '08', '10', '12', '14', '18', '20',
               '22', '24', '25', '26', '27', '28', '29', '30', '31']

# ============================================================================
# Example 1: Compute Distance Metrics for Single Subject
# ============================================================================

print("Example 1: Computing Distance Metrics for Single Subject")
print("=" * 70)

# Load observed RSMs
print(f"\nLoading observed RSMs for subject {SUBJECT_ID}...")
with open(f'{RSM_DIR}{SUBJECT_ID}_RSM_activeVisual.pkl', 'rb') as f:
    observed_rsms = pickle.load(f)

# Load predicted RSMs
print(f"Loading predicted RSMs...")
with open(f'{OUTPUT_DIR}{SUBJECT_ID}_predicted_RSM_actflow.pkl', 'rb') as f:
    predicted_rsms = pickle.load(f)

# Load parcel FC
print(f"Loading parcel-level FC...")
with open(f'{FC_DIR}allSubFC_parCorr.pkl', 'rb') as f:
    all_parcel_fc = pickle.load(f)

subject_idx = SUBJECT_IDS.index(SUBJECT_ID)
parcel_fc = all_parcel_fc[subject_idx, :, :]

# Compute all distances
results = actflow_metrics.process_subject_distances(
    subject_id=SUBJECT_ID,
    observed_rsms=observed_rsms,
    predicted_rsms=predicted_rsms,
    parcel_fc=parcel_fc,
    output_dir=OUTPUT_DIR
)

print("\n✓ Distance computation completed!")
print("=" * 70)

# ============================================================================
# Example 2: Visualize Distance Metrics
# ============================================================================

print("\nExample 2: Visualizing Distance Metrics")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Distribution of each distance metric
axes[0, 0].hist(results['d_trans'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('d_trans (Obs Target from Obs Sources)')
axes[0, 0].set_ylabel('Number of Parcels')
axes[0, 0].set_title('Transformation Distance')
axes[0, 0].axvline(np.mean(results['d_trans']), color='red', linestyle='--', 
                   label=f"Mean: {np.mean(results['d_trans']):.3f}")
axes[0, 0].legend()

axes[0, 1].hist(results['d_trans_hat'], bins=30, alpha=0.7, color='coral', edgecolor='black')
axes[0, 1].set_xlabel('d_trans_hat (Pred Target from Obs Sources)')
axes[0, 1].set_ylabel('Number of Parcels')
axes[0, 1].set_title('Predicted Transformation Distance')
axes[0, 1].axvline(np.mean(results['d_trans_hat']), color='red', linestyle='--',
                   label=f"Mean: {np.mean(results['d_trans_hat']):.3f}")
axes[0, 1].legend()

axes[1, 0].hist(results['d_pred'], bins=30, alpha=0.7, color='mediumseagreen', edgecolor='black')
axes[1, 0].set_xlabel('d_pred (Pred Target from Obs Target)')
axes[1, 0].set_ylabel('Number of Parcels')
axes[1, 0].set_title('Prediction Distance')
axes[1, 0].axvline(np.mean(results['d_pred']), color='red', linestyle='--',
                   label=f"Mean: {np.mean(results['d_pred']):.3f}")
axes[1, 0].legend()

# Transformation evidence
axes[1, 1].hist(results['transformation_evidence'], bins=30, alpha=0.7, 
               color='mediumpurple', edgecolor='black')
axes[1, 1].set_xlabel('Transformation Evidence (d_trans_hat - d_pred)')
axes[1, 1].set_ylabel('Number of Parcels')
axes[1, 1].set_title('Evidence for Meaningful Transformation')
axes[1, 1].axvline(0, color='black', linestyle='--', alpha=0.5, label='No evidence')
axes[1, 1].axvline(np.mean(results['transformation_evidence']), color='red', linestyle='--',
                   label=f"Mean: {np.mean(results['transformation_evidence']):.3f}")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}distance_distributions_{SUBJECT_ID}.png', dpi=150)
print(f"\nSaved distance distributions to {OUTPUT_DIR}")

# ============================================================================
# Example 3: Compare d_trans_hat vs d_pred
# ============================================================================

print("\nExample 3: Comparing Predicted Transformation vs Prediction Distance")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot: d_trans_hat vs d_pred
axes[0].scatter(results['d_trans_hat'], results['d_pred'], 
               alpha=0.6, s=50, c=results['transformation_evidence'],
               cmap='RdBu_r', vmin=-0.5, vmax=0.5, edgecolor='black', linewidth=0.5)
axes[0].plot([0, 2], [0, 2], 'k--', alpha=0.5, label='Unity line')
axes[0].set_xlabel('d_trans_hat (Pred Target from Sources)')
axes[0].set_ylabel('d_pred (Pred from Obs Target)')
axes[0].set_title('Transformation vs Prediction Distance')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(axes[0].collections[0], ax=axes[0], label='Transformation Evidence')

# Count parcels by quadrant
above_unity = np.sum(results['d_trans_hat'] > results['d_pred'])
below_unity = np.sum(results['d_trans_hat'] <= results['d_pred'])

axes[0].text(0.05, 0.95, f"Above unity: {above_unity} parcels\n(Evidence for transformation)",
            transform=axes[0].transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Bar plot: proportion with positive evidence
positive_evidence = np.sum(results['transformation_evidence'] > 0)
negative_evidence = np.sum(results['transformation_evidence'] <= 0)

axes[1].bar(['Positive\nEvidence', 'Negative\nEvidence'], 
           [positive_evidence, negative_evidence],
           color=['mediumseagreen', 'coral'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Number of Parcels')
axes[1].set_title('Transformation Evidence Summary')
axes[1].set_ylim([0, 360])

# Add percentages
for i, (label, count) in enumerate([('Positive', positive_evidence), 
                                     ('Negative', negative_evidence)]):
    pct = count / 360 * 100
    axes[1].text(i, count + 10, f'{count}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}transformation_evidence_{SUBJECT_ID}.png', dpi=150)
print(f"\nSaved transformation evidence plot to {OUTPUT_DIR}")

print("\n" + "=" * 70)
print("All examples completed!")

# ============================================================================
# Example 4: Interpretation Guide
# ============================================================================

print("\nInterpretation Guide:")
print("=" * 70)
print("""
Three Distance Metrics:

1. d_trans (Transformation Distance):
   - Cosine distance between observed target RSM and observed source RSMs
   - Measures: How different is the target from its inputs?
   - High values → Target has transformed away from sources

2. d_trans_hat (Predicted Transformation Distance):
   - Cosine distance between predicted target RSM and observed source RSMs  
   - Measures: Does the model predict similar transformation?
   - Should match d_trans if model captures transformation

3. d_pred (Prediction Distance):
   - Cosine distance between predicted target RSM and observed target RSM
   - Measures: How accurate is the prediction?
   - Low values → Good prediction

Transformation Evidence (d_trans_hat - d_pred):

If d_trans_hat > d_pred (POSITIVE evidence):
   → Predicted target is closer to observed target than to sources
   → Evidence that connectivity does meaningful transformation
   → Model captures information integration, not just averaging

If d_trans_hat ≈ d_pred (NEUTRAL):
   → Ambiguous about transformation
   
If d_trans_hat < d_pred (NEGATIVE evidence):
   → Predicted target closer to sources than to observed target
   → Model fails to capture transformation
   → Connectivity might just be averaging inputs
""")

print("=" * 70)
