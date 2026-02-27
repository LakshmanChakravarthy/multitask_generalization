"""
Example: Compute task betas for MDTB dataset.

This script demonstrates how to estimate beta weights for task conditions
using ridge regression with cross-validation.
"""

import task_glm

# ============================================================================
# Configuration
# ============================================================================

# Paths
DATA_DIR = '/home/ln275/f_mc1689_1/MDTB/qunex_mdtb/'
OUTPUT_DIR = '/home/ln275/f_mc1689_1/MDTB/derivatives/postprocessing/betasByTaskCondition/'
TASK_INFO_FILE = '/home/ln275/f_mc1689_1/MDTB/data/derivatives/allSubTaskConditionInfo.pkl'

# Processing parameters
SPACE = 'parcellated'  # or 'vertex'
N_RUNS = 8

# ============================================================================
# Example 1: Process single run
# ============================================================================

print("Example 1: Processing single run...")
print("=" * 70)

task_glm.compute_task_betas_single_run(
    subject_id='02',
    session_id='a1',
    run_id=1,
    space=SPACE,
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    task_info_file=TASK_INFO_FILE
)

print("\nCompleted single run!")
print("=" * 70)

# ============================================================================
# Example 2: Process all runs for one subject/session
# ============================================================================

print("\nExample 2: Processing all runs for subject 02, session a1...")
print("=" * 70)

subject_id = '02'
session_id = 'a1'

for run_id in range(1, N_RUNS + 1):
    print(f"\nProcessing run {run_id}...")
    try:
        task_glm.compute_task_betas_single_run(
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            space=SPACE,
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            task_info_file=TASK_INFO_FILE
        )
    except FileNotFoundError:
        print(f"Files not found for run {run_id}, skipping...")
        continue

print("\nCompleted all runs!")
print("=" * 70)

# ============================================================================
# Example 3: Process all subjects (batch processing)
# ============================================================================

print("\nExample 3: Batch processing all subjects...")
print("=" * 70)

# Define subjects and sessions
subject_ids = ['02', '03', '04', '06']  # Subset for demo
session_ids = ['a1', 'a2', 'b1', 'b2']

# Process all
task_glm.process_all_subjects(
    subject_ids=subject_ids,
    session_ids=session_ids,
    n_runs=N_RUNS,
    space=SPACE,
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    task_info_file=TASK_INFO_FILE
)

print("\nCompleted batch processing!")
print("=" * 70)

# ============================================================================
# Example 4: Load and inspect results
# ============================================================================

print("\nExample 4: Loading and inspecting results...")
print("=" * 70)

import h5py
import numpy as np

# Load results from Example 1
output_file = f'{OUTPUT_DIR}02_a1_tfMRI_{SPACE}_betaseries_bold1.h5'

with h5py.File(output_file, 'r') as f:
    betas = f['betas'][:]
    residuals = f['residuals'][:]

print(f"\nBetas shape: {betas.shape}")
print(f"  (n_task_conditions x n_parcels/vertices)")
print(f"\nResiduals shape: {residuals.shape}")
print(f"  (n_timepoints x n_parcels/vertices)")

# Load task condition names
task_index = np.loadtxt(f'{OUTPUT_DIR}02_a1_tfMRI_{SPACE}_betaseries_bold1_taskIndex.csv',
                        delimiter=',', dtype=str)
print(f"\nNumber of task conditions: {len(task_index)}")
print(f"Example conditions: {task_index[:5]}")

print("\n" + "=" * 70)
print("All examples completed!")
