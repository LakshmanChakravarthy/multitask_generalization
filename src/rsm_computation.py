"""
Representational Similarity Matrix (RSM) computation for MDTB task data.

Computes cross-validated RSMs using cosine similarity between task condition
activation patterns, separately for each brain region. Also includes functions
for computing representational dimensionality from RSMs.
"""

import numpy as np
import pickle
import h5py
import nibabel as nib
from pathlib import Path


# Task condition subset: 96 active visual tasks (excluding passive and auditory)
TASK_CONDITION_SUBSET = np.array([
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,36,37,38,39,40,
    41,42,43,44,45,46,49,50,51,52,53,54,55,56,60,61,62,63,64,65,
    66,67,68,69,70,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,
    87,88,89,91,92,93,94,95,96,97,98,99,100,101,102,104,105,106,
    107,108,109,110,111,112,113,114,115,120,121,122,123,124,125
])


def load_subject_betas(subject_id, beta_dir, space='parcellated'):
    """
    Load all task betas for a subject across all sessions and runs.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    beta_dir : str
        Directory containing task beta HDF5 files
    space : str, default='parcellated'
        Data space ('parcellated' or 'vertex')
        
    Returns
    -------
    betas_a : ndarray, shape (n_conditions, n_regions)
        Betas from sessions a1+a2 (concatenated across runs)
    betas_b : ndarray, shape (n_conditions, n_regions)
        Betas from sessions b1+b2 (concatenated across runs)
    condition_names : list
        Task condition names
    """
    
    session_ids = ['a1', 'a2', 'b1', 'b2']
    n_runs = 8
    
    # Collect betas separately for a-sessions and b-sessions
    betas_a_all = []
    betas_b_all = []
    all_condition_names = []
    
    for session_id in session_ids:
        for run_id in range(1, n_runs + 1):
            filename = f'{beta_dir}{subject_id}_{session_id}_tfMRI_{space}_betaseries_bold{run_id}.h5'
            
            # Load betas
            with h5py.File(filename, 'r') as f:
                betas = f['betas'][:]  # Shape: (n_conditions, n_regions)
            
            # Load condition names
            task_index_file = f'{beta_dir}{subject_id}_{session_id}_tfMRI_{space}_betaseries_bold{run_id}_taskIndex.csv'
            condition_names = np.loadtxt(task_index_file, delimiter=',', dtype=str)
            
            # Separate a-sessions from b-sessions
            if session_id in ['a1', 'a2']:
                betas_a_all.append(betas)
            else:  # b1, b2
                betas_b_all.append(betas)
            
            all_condition_names.extend(condition_names)
    
    # Concatenate across runs within each session type
    betas_a = np.vstack(betas_a_all)  # Shape: (n_conditions_a, n_regions)
    betas_b = np.vstack(betas_b_all)  # Shape: (n_conditions_b, n_regions)
    
    return betas_a, betas_b, all_condition_names


def select_task_subset(betas, task_indices=TASK_CONDITION_SUBSET):
    """
    Select subset of task conditions (96 active visual tasks).
    
    Parameters
    ----------
    betas : ndarray, shape (n_conditions_total, n_regions)
        Beta weights for all task conditions
    task_indices : ndarray, default=TASK_CONDITION_SUBSET
        Indices of task conditions to keep
        
    Returns
    -------
    betas_subset : ndarray, shape (96, n_regions)
        Beta weights for selected task conditions
    """
    
    return betas[task_indices, :]


def compute_rsm_single_region(betas_session1, betas_session2):
    """
    Compute cross-validated RSM for a single region using cosine similarity.
    
    Uses the efficient approach: normalize vectors, then dot product = cosine similarity.
    
    Parameters
    ----------
    betas_session1 : ndarray, shape (n_conditions, n_vertices)
        Beta patterns from first session (e.g., a1+a2)
    betas_session2 : ndarray, shape (n_conditions, n_vertices)
        Beta patterns from second session (e.g., b1+b2)
        
    Returns
    -------
    rsm : ndarray, shape (n_conditions, n_conditions)
        Cross-validated representational similarity matrix
    """
    
    # Normalize each task condition pattern (row-wise L2 normalization)
    # This makes the dot product equal to cosine similarity
    betas_session1_norm = betas_session1 / np.linalg.norm(betas_session1, axis=1, keepdims=True)
    betas_session2_norm = betas_session2 / np.linalg.norm(betas_session2, axis=1, keepdims=True)
    
    # Dot product of normalized vectors = cosine similarity
    # Shape: (n_conditions, n_conditions)
    rsm = np.dot(betas_session1_norm, betas_session2_norm.T)
    
    return rsm


def compute_rsm_all_regions(betas_a, betas_b, glasser):
    """
    Compute RSMs for all brain regions.
    
    Parameters
    ----------
    betas_a : ndarray, shape (n_conditions, n_vertices)
        Betas from sessions a1+a2
    betas_b : ndarray, shape (n_conditions, n_vertices)
        Betas from sessions b1+b2
    glasser : ndarray, shape (n_vertices,)
        Glasser parcellation labels
        
    Returns
    -------
    rsms : ndarray, shape (n_parcels, n_conditions, n_conditions)
        RSMs for each parcel
    """
    
    n_parcels = 360
    n_conditions = betas_a.shape[0]
    
    rsms = np.zeros((n_parcels, n_conditions, n_conditions))
    
    for roi_idx in range(n_parcels):
        print(f'Computing RSM for parcel {roi_idx + 1}/{n_parcels}')
        
        # Get vertices in this parcel
        parcel_vertices = np.where(glasser == roi_idx + 1)[0]
        
        # Extract betas for this region
        region_betas_a = betas_a[:, parcel_vertices]
        region_betas_b = betas_b[:, parcel_vertices]
        
        # Compute cross-validated RSM
        rsms[roi_idx] = compute_rsm_single_region(region_betas_a, region_betas_b)
    
    return rsms


def process_subject(subject_id, beta_dir, output_dir, glasser_file=None, space='parcellated'):
    """
    Complete RSM computation pipeline for one subject.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    beta_dir : str
        Directory containing task beta HDF5 files
    output_dir : str
        Output directory for RSMs
    glasser_file : str, optional
        Path to Glasser parcellation file
    space : str, default='parcellated'
        Data space
        
    Returns
    -------
    None (saves RSMs to file)
    """
    
    print(f"\nProcessing subject {subject_id}")
    print("=" * 70)
    
    # Load Glasser parcellation
    if glasser_file is None:
        glasser_file = '/home/ln275/f_mc1689_1/MDTB/docs/files/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
    
    glasser = np.squeeze(nib.load(glasser_file).get_fdata())
    
    # Load subject's task betas
    print("Loading task betas...")
    betas_a, betas_b, condition_names = load_subject_betas(subject_id, beta_dir, space)
    print(f"Loaded {betas_a.shape[0]} conditions from a-sessions")
    print(f"Loaded {betas_b.shape[0]} conditions from b-sessions")
    
    # Select 96 active visual task conditions
    print("\nSelecting 96 active visual task conditions...")
    betas_a_subset = select_task_subset(betas_a)
    betas_b_subset = select_task_subset(betas_b)
    print(f"Selected subset shape: {betas_a_subset.shape}")
    
    # Compute RSMs for all regions
    print("\nComputing cross-validated RSMs for all regions...")
    rsms = compute_rsm_all_regions(betas_a_subset, betas_b_subset, glasser)
    print(f"RSMs shape: {rsms.shape}")
    
    # Save results
    output_file = f'{output_dir}{subject_id}_RSM_activeVisual.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(rsms, f)
    
    print(f"\nSaved RSMs to: {output_file}")
    print("=" * 70)


def process_all_subjects(subject_ids, beta_dir, output_dir, **kwargs):
    """
    Batch process RSMs for multiple subjects.
    
    Parameters
    ----------
    subject_ids : list of str
        Subject identifiers
    beta_dir : str
        Directory containing task beta files
    output_dir : str
        Output directory for RSMs
    **kwargs : dict
        Additional arguments passed to process_subject
    """
    
    for subject_id in subject_ids:
        try:
            process_subject(subject_id, beta_dir, output_dir, **kwargs)
        except FileNotFoundError as e:
            print(f"Files not found for subject {subject_id}: {e}")
            continue


def load_all_subject_rsms(subject_ids, rsm_dir):
    """
    Load RSMs for all subjects into a single array.
    
    Parameters
    ----------
    subject_ids : list of str
        Subject identifiers
    rsm_dir : str
        Directory containing RSM pickle files
        
    Returns
    -------
    all_rsms : ndarray, shape (n_subjects, n_parcels, 96, 96)
        RSMs for all subjects
    """
    
    all_rsms = []
    
    for subject_id in subject_ids:
        filename = f'{rsm_dir}{subject_id}_RSM_activeVisual.pkl'
        with open(filename, 'rb') as f:
            rsms = pickle.load(f)
        all_rsms.append(rsms)
    
    all_rsms = np.array(all_rsms)
    
    return all_rsms


# Utility function for computing RSM from parcellated data directly
def compute_rsm_parcellated(betas_a, betas_b):
    """
    Compute RSM directly from parcellated data (no need to loop over regions).
    
    Parameters
    ----------
    betas_a : ndarray, shape (n_conditions, n_parcels)
        Betas from sessions a1+a2
    betas_b : ndarray, shape (n_conditions, n_parcels)
        Betas from sessions b1+b2
        
    Returns
    -------
    rsms : ndarray, shape (n_parcels, n_conditions, n_conditions)
        RSMs for each parcel
    """
    
    n_conditions, n_parcels = betas_a.shape
    rsms = np.zeros((n_parcels, n_conditions, n_conditions))
    
    for parcel_idx in range(n_parcels):
        print(f'Computing RSM for parcel {parcel_idx + 1}/{n_parcels}')
        
        # Extract betas for this parcel (reshape to 2D for compute_rsm_single_region)
        region_betas_a = betas_a[:, parcel_idx:parcel_idx+1]  # Shape: (n_conditions, 1)
        region_betas_b = betas_b[:, parcel_idx:parcel_idx+1]
        
        # Compute RSM
        rsms[parcel_idx] = compute_rsm_single_region(region_betas_a, region_betas_b)
    
    return rsms


# ============================================================================
# Representational Dimensionality Analysis
# ============================================================================

def get_dimensionality(rsm):
    """
    Compute representational dimensionality from an RSM.
    
    Uses participation ratio of eigenvalues: (Σλ)² / Σλ²
    
    This measures the effective number of dimensions needed to capture
    the representational structure. Higher values indicate representations
    are spread across more dimensions (less low-dimensional).
    
    Parameters
    ----------
    rsm : ndarray, shape (n_conditions, n_conditions)
        Representational similarity matrix (should be square and symmetric)
        
    Returns
    -------
    dimensionality : float
        Participation ratio of eigenvalues
        
    Notes
    -----
    For a perfectly 1-dimensional representation: dimensionality ≈ 1
    For uniform high-dimensional representation: dimensionality ≈ n_conditions
    """
    
    # Compute eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(rsm)
    
    # Take real part (RSMs should be real symmetric, but numerical errors can introduce tiny imaginary parts)
    eigenvalues = np.real(eigenvalues)
    
    # Participation ratio: (Σλ)² / Σλ²
    dimensionality = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
    
    return dimensionality


def compute_dimensionality_all_regions(rsms):
    """
    Compute representational dimensionality for all regions.
    
    Parameters
    ----------
    rsms : ndarray, shape (n_parcels, n_conditions, n_conditions)
        RSMs for all parcels
        
    Returns
    -------
    dimensionalities : ndarray, shape (n_parcels,)
        Dimensionality value for each parcel
    """
    
    n_parcels = rsms.shape[0]
    dimensionalities = np.zeros(n_parcels)
    
    for parcel_idx in range(n_parcels):
        dimensionalities[parcel_idx] = get_dimensionality(rsms[parcel_idx])
    
    return dimensionalities


def compute_dimensionality_all_subjects(subject_ids, rsm_dir, output_dir):
    """
    Compute representational dimensionality for all subjects and save.
    
    Parameters
    ----------
    subject_ids : list of str
        Subject identifiers
    rsm_dir : str
        Directory containing RSM pickle files
    output_dir : str
        Output directory for dimensionality results
        
    Returns
    -------
    all_dimensionalities : ndarray, shape (n_subjects, n_parcels)
        Dimensionality values for all subjects and parcels
    """
    
    n_subjects = len(subject_ids)
    n_parcels = 360
    
    all_dimensionalities = np.zeros((n_subjects, n_parcels))
    
    for sub_idx, subject_id in enumerate(subject_ids):
        print(f'Computing dimensionality for subject {subject_id} ({sub_idx + 1}/{n_subjects})')
        
        # Load RSMs
        rsm_file = f'{rsm_dir}{subject_id}_RSM_activeVisual.pkl'
        with open(rsm_file, 'rb') as f:
            rsms = pickle.load(f)
        
        # Compute dimensionality for all regions
        dimensionalities = compute_dimensionality_all_regions(rsms)
        all_dimensionalities[sub_idx, :] = dimensionalities
    
    # Save results
    output_file = f'{output_dir}allsub_representational_dimensionality.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_dimensionalities, f)
    
    print(f'\nSaved dimensionality results to: {output_file}')
    
    return all_dimensionalities
