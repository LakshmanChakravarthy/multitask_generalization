"""
Example: Compute FC gradients from parcel-level functional connectivity.

This script demonstrates how to compute principal gradients of FC using PCA
on thresholded, averaged connectivity matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fc_gradients

# ============================================================================
# Configuration
# ============================================================================

# Paths
FC_FILE = '/home/ln275/f_mc1689_1/multitask_generalization/data/derivatives/FC_new/allSubFC_parCorr.pkl'
OUTPUT_DIR = '/home/ln275/f_mc1689_1/multitask_generalization/derivatives/FC_gradients/'

# Parameters
N_COMPONENTS = 2  # Number of gradients to extract
THRESHOLD_PERCENTILE = 80  # Keep top 80% of connections

# ============================================================================
# Example 1: Compute FC gradients from file
# ============================================================================

print("Example 1: Computing FC gradients...")
print("=" * 70)

gradients, explained_variance, mean_fc = fc_gradients.load_and_compute_fc_gradients(
    fc_file=FC_FILE,
    n_components=N_COMPONENTS,
    threshold_percentile=THRESHOLD_PERCENTILE
)

print(f"\nGradient 1 range: [{gradients[:, 0].min():.3f}, {gradients[:, 0].max():.3f}]")
print(f"Gradient 2 range: [{gradients[:, 1].min():.3f}, {gradients[:, 1].max():.3f}]")

# Save results
output_file = f'{OUTPUT_DIR}fc_gradients_top{THRESHOLD_PERCENTILE}pct.npz'
fc_gradients.save_fc_gradients(gradients, explained_variance, mean_fc, output_file)

print("\nCompleted!")
print("=" * 70)

# ============================================================================
# Example 2: Visualize gradients
# ============================================================================

print("\nExample 2: Visualizing FC gradients...")
print("=" * 70)

# Plot gradient distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gradient 1
axes[0].hist(gradients[:, 0], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Gradient 1 Value')
axes[0].set_ylabel('Number of Parcels')
axes[0].set_title(f'FC Gradient 1 (Explains {explained_variance[0]:.1%} variance)')
axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)

# Gradient 2
axes[1].hist(gradients[:, 1], bins=30, color='coral', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Gradient 2 Value')
axes[1].set_ylabel('Number of Parcels')
axes[1].set_title(f'FC Gradient 2 (Explains {explained_variance[1]:.1%} variance)')
axes[1].axvline(0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fc_gradient_distributions.png', dpi=150)
print(f"Saved gradient distributions to {OUTPUT_DIR}fc_gradient_distributions.png")

# Scatter plot: Gradient 1 vs Gradient 2
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

scatter = ax.scatter(gradients[:, 0], gradients[:, 1], 
                    c=np.arange(360), cmap='viridis', 
                    s=50, alpha=0.6, edgecolor='black', linewidth=0.5)
ax.set_xlabel(f'Gradient 1 ({explained_variance[0]:.1%} variance)')
ax.set_ylabel(f'Gradient 2 ({explained_variance[1]:.1%} variance)')
ax.set_title('FC Gradients: Principal Axes of Connectivity Variation')
ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax, label='Parcel Index')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fc_gradient_scatter.png', dpi=150)
print(f"Saved gradient scatter plot to {OUTPUT_DIR}fc_gradient_scatter.png")

print("\nCompleted!")
print("=" * 70)

# ============================================================================
# Example 3: Identify extreme parcels along gradients
# ============================================================================

print("\nExample 3: Identifying extreme parcels along gradients...")
print("=" * 70)

# Find parcels at extremes of each gradient
n_extreme = 5

# Gradient 1
grad1_low = np.argsort(gradients[:, 0])[:n_extreme]
grad1_high = np.argsort(gradients[:, 0])[-n_extreme:]

print(f"\nGradient 1 - Lowest values (parcels):")
for idx in grad1_low:
    print(f"  Parcel {idx + 1}: {gradients[idx, 0]:.3f}")

print(f"\nGradient 1 - Highest values (parcels):")
for idx in grad1_high:
    print(f"  Parcel {idx + 1}: {gradients[idx, 0]:.3f}")

# Gradient 2
grad2_low = np.argsort(gradients[:, 1])[:n_extreme]
grad2_high = np.argsort(gradients[:, 1])[-n_extreme:]

print(f"\nGradient 2 - Lowest values (parcels):")
for idx in grad2_low:
    print(f"  Parcel {idx + 1}: {gradients[idx, 1]:.3f}")

print(f"\nGradient 2 - Highest values (parcels):")
for idx in grad2_high:
    print(f"  Parcel {idx + 1}: {gradients[idx, 1]:.3f}")

print("\n" + "=" * 70)
print("All examples completed!")

# ============================================================================
# Example 4: Load saved gradients
# ============================================================================

print("\nExample 4: Loading saved gradients...")
print("=" * 70)

# Load from file
loaded_gradients, loaded_variance, loaded_mean_fc = fc_gradients.load_fc_gradients(output_file)

print(f"\nLoaded gradients shape: {loaded_gradients.shape}")
print(f"Explained variance: {loaded_variance}")
print(f"Mean FC shape: {loaded_mean_fc.shape}")

# Verify they match
assert np.allclose(loaded_gradients, gradients), "Loaded gradients don't match!"
print("\n✓ Verification passed: Loaded gradients match computed gradients")

print("\n" + "=" * 70)
print("All examples completed!")
