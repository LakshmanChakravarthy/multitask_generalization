"""
Functional Connectivity (FC) Dimensionality Analysis.

Computes dimensionality of FC patterns using singular value participation ratio (SVPR).
This measures the effective dimensionality of connectivity patterns from source regions
to a target region.
"""

import numpy as np
import pickle


def get_svpr(fc_matrix):
    """
    Compute singular value participation ratio (SVPR) for FC dimensionality.
    
    Uses participation ratio of singular values: (Σs)² / Σs²
    
    This measures the effective number of dimensions in the connectivity pattern.
    Higher values indicate FC is spread across more dimensions (less low-dimensional).
    
    Parameters
    ----------
    fc_matrix : ndarray, shape (n_target_vertices, n_source_vertices)
        Functional connectivity matrix from source vertices to target vertices
        
    Returns
    -------
    dimensionality : float
        Participation ratio of singular values
        
    Notes
    -----
    For a rank-1 connectivity pattern: dimensionality ≈ 1
    For full-rank pattern: dimensionality ≈ min(n_target, n_source)
    """
    
    # Compute SVD
    U, S, V_T = np.linalg.svd(fc_matrix, full_matrices=False)
    
    # Take real part (FC matrices should be real, but numerical errors can introduce tiny imaginary parts)
    S = np.real(S)
    
    # Participation ratio: (Σs)² / Σs²
    dimensionality = np.sum(S)**2 / np.sum(S**2)
    
    return dimensionality


def compute_fc_dimensionality_all_regions(fc_dict):
    """
    Compute FC dimensionality for all target regions.
    
    Parameters
    ----------
    fc_dict : dict
        Dictionary with keys = target region indices (0-359)
        Values = FC matrices of shape (n_target_vertices, n_source_vertices)
        
    Returns
    -------
    dimensionalities : ndarray, shape (n_parcels,)
        FC dimensionality for each target parcel
    """
    
    n_parcels = len(fc_dict)
    dimensionalities = np.zeros(n_parcels)
    
    for target_roi_idx in range(n_parcels):
        if target_roi_idx in fc_dict:
            fc_matrix = fc_dict[target_roi_idx]
            dimensionalities[target_roi_idx] = get_svpr(fc_matrix)
        else:
            dimensionalities[target_roi_idx] = np.nan
    
    return dimensionalities


def compute_fc_dimensionality_all_subjects(subject_ids, fc_dir, output_dir):
    """
    Compute FC dimensionality for all subjects and save.
    
    Parameters
    ----------
    subject_ids : list of str
        Subject identifiers
    fc_dir : str
        Directory containing vertex-wise FC pickle files
    output_dir : str
        Output directory for dimensionality results
        
    Returns
    -------
    all_dimensionalities : ndarray, shape (n_subjects, n_parcels)
        FC dimensionality values for all subjects and parcels
    """
    
    n_subjects = len(subject_ids)
    n_parcels = 360
    
    all_dimensionalities = np.zeros((n_subjects, n_parcels))
    
    for sub_idx, subject_id in enumerate(subject_ids):
        print(f'Computing FC dimensionality for subject {subject_id} ({sub_idx + 1}/{n_subjects})')
        
        # Load vertex-wise FC
        fc_file = f'{fc_dir}sub{subject_id}_vertFC_CVoptimal_nPCs.pkl'
        with open(fc_file, 'rb') as f:
            fc_dict = pickle.load(f)
            fc_rsq_arr = pickle.load(f)  # R-squared values (not used here)
            optimal_nPC = pickle.load(f)  # Optimal nPCs (not used here)
        
        # Compute FC dimensionality for all regions
        dimensionalities = compute_fc_dimensionality_all_regions(fc_dict)
        all_dimensionalities[sub_idx, :] = dimensionalities
    
    # Save results
    output_file = f'{output_dir}allsub_FC_dimensionality.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_dimensionalities, f)
    
    print(f'\nSaved FC dimensionality results to: {output_file}')
    
    return all_dimensionalities


def load_fc_dimensionality(dimensionality_file):
    """
    Load FC dimensionality results from pickle file.
    
    Parameters
    ----------
    dimensionality_file : str
        Path to dimensionality pickle file
        
    Returns
    -------
    dimensionalities : ndarray, shape (n_subjects, n_parcels)
        FC dimensionality values
    """
    
    with open(dimensionality_file, 'rb') as f:
        dimensionalities = pickle.load(f)
    
    return dimensionalities
