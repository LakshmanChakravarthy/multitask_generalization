"""
Activity Flow Permutation Tests.

Generates null distributions by permuting connectivity matrices and rerunning
activity flow predictions. Computes RSMs, dimensionality, and transformation
distances from permuted predictions to test statistical significance.
"""

import numpy as np
import pickle
from tqdm import tqdm
import actflow_prediction
import actflow_metrics


def permute_fc_matrix(fc_matrix, permute_mode='full'):
    """
    Randomly permute functional connectivity matrix.
    
    Parameters
    ----------
    fc_matrix : ndarray, shape (n_target_vertices, n_source_vertices)
        Original FC matrix (targets = rows, sources = columns)
    permute_mode : str, default='full'
        Permutation strategy:
        - 'full': Shuffle all values (default)
        - 'columns': Shuffle within each column independently
          (preserves target vertex properties like in-degree)
        
    Returns
    -------
    fc_permuted : ndarray, shape (n_target_vertices, n_source_vertices)
        Permuted FC matrix
        
    Notes
    -----
    Column-wise permutation preserves:
    - Target vertex in-degree (sum across sources)
    - Target vertex in-degree distribution
    - Overall connectivity strength to each target
    
    This tests whether the specific arrangement of source→target connections
    matters, while controlling for target vertex connectivity properties.
    """
    
    fc_permuted = fc_matrix.copy()
    
    if permute_mode == 'full':
        # Flatten, permute, reshape (original behavior)
        fc_flat = fc_permuted.flatten()
        fc_permuted_flat = np.random.permutation(fc_flat)
        fc_permuted = fc_permuted_flat.reshape(fc_matrix.shape)
        
    elif permute_mode == 'columns':
        # Shuffle each column (source) independently
        # This preserves each row's (target's) sum and distribution
        for col_idx in range(fc_permuted.shape[1]):
            fc_permuted[:, col_idx] = np.random.permutation(fc_permuted[:, col_idx])
    
    else:
        raise ValueError(f"Invalid permute_mode: {permute_mode}. Must be 'full' or 'columns'")
    
    return fc_permuted


def predict_with_permuted_fc(source_betas, fc_matrix_original, n_permutations=100, 
                             permute_mode='full'):
    """
    Generate predictions using permuted connectivity.
    
    Parameters
    ----------
    source_betas : ndarray, shape (n_conditions, n_source_vertices)
        Source beta weights
    fc_matrix_original : ndarray, shape (n_target_vertices, n_source_vertices)
        Original FC matrix
    n_permutations : int, default=100
        Number of permutations
    permute_mode : str, default='full'
        Permutation strategy: 'full' or 'columns'
        
    Returns
    -------
    permuted_predictions : ndarray, shape (n_conditions, n_target_vertices, n_permutations)
        Predictions from each permutation
    """
    
    n_conditions = source_betas.shape[0]
    n_target_vertices = fc_matrix_original.shape[0]
    
    permuted_predictions = np.zeros((n_conditions, n_target_vertices, n_permutations))
    
    for perm_idx in range(n_permutations):
        # Permute FC
        fc_permuted = permute_fc_matrix(fc_matrix_original, permute_mode=permute_mode)
        
        # Activity flow prediction with permuted FC
        permuted_predictions[:, :, perm_idx] = actflow_prediction.predict_target_betas_actflow(
            source_betas, fc_permuted
        )
    
    return permuted_predictions


def compute_permuted_rsms(predicted_betas_permuted, rsm_computation_module):
    """
    Compute RSMs from permuted predictions.
    
    Parameters
    ----------
    predicted_betas_permuted : ndarray, shape (n_conditions, n_target_vertices, n_sessions, n_permutations)
        Permuted predictions for one region
    rsm_computation_module : module
        The rsm_computation module
        
    Returns
    -------
    permuted_rsms : ndarray, shape (n_permutations, n_conditions, n_conditions)
        RSMs from each permutation
    """
    
    n_permutations = predicted_betas_permuted.shape[3]
    n_conditions = predicted_betas_permuted.shape[0]
    
    permuted_rsms = np.zeros((n_permutations, n_conditions, n_conditions))
    
    for perm_idx in range(n_permutations):
        # Extract this permutation's predictions
        betas_sess0 = predicted_betas_permuted[:, :, 0, perm_idx]
        betas_sess1 = predicted_betas_permuted[:, :, 1, perm_idx]
        
        # Compute RSM
        permuted_rsms[perm_idx] = rsm_computation_module.compute_rsm_single_region(
            betas_sess0, betas_sess1
        )
    
    return permuted_rsms


def compute_permuted_transformation_distance(observed_rsms, permuted_rsms, 
                                             target_idx, parcel_fc):
    """
    Compute d_trans_hat for connectivity-permuted predictions.
    
    For each permutation: distance from permuted predicted target to observed sources.
    
    Parameters
    ----------
    observed_rsms : ndarray, shape (n_parcels, n_conditions, n_conditions)
        Observed RSMs
    permuted_rsms : ndarray, shape (n_permutations, n_conditions, n_conditions)
        Permuted predicted RSMs for this target
    target_idx : int
        Target parcel index
    parcel_fc : ndarray, shape (n_parcels, n_parcels)
        Parcel-level FC
        
    Returns
    -------
    d_trans_hat_permuted : ndarray, shape (n_permutations,)
        Predicted transformation distance for each permutation
    """
    
    n_permutations = permuted_rsms.shape[0]
    d_trans_hat_permuted = np.zeros(n_permutations)
    
    for perm_idx in range(n_permutations):
        # Compute d_trans_hat for this permutation
        d_trans_hat_permuted[perm_idx], _ = actflow_metrics.compute_predicted_transformation_distance(
            observed_rsms, permuted_rsms[perm_idx], target_idx, parcel_fc
        )
    
    return d_trans_hat_permuted


def permutation_test_single_subject(subject_id, task_betas, fc_dict, parcel_fc, 
                                    observed_rsms, glasser, glasser_parcels_dir,
                                    rsm_computation_module, n_permutations=100,
                                    permute_mode='full', output_dir=None):
    """
    Complete permutation test pipeline for one subject.
    
    For each target region:
    1. Permute connectivity n_permutations times
    2. Generate predictions with permuted connectivity
    3. Compute permuted RSMs
    4. Compute permuted dimensionality
    5. Compute permuted d_trans_hat
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    task_betas : ndarray, shape (2, n_conditions, n_vertices)
        Observed task betas
    fc_dict : dict
        Vertex-wise FC dictionary
    parcel_fc : ndarray, shape (360, 360)
        Parcel-level FC
    observed_rsms : ndarray, shape (360, n_conditions, n_conditions)
        Observed RSMs
    glasser : ndarray
        Parcellation labels
    glasser_parcels_dir : str
        Dilated parcel directory
    rsm_computation_module : module
        RSM computation module
    n_permutations : int, default=100
        Number of permutations
    permute_mode : str, default='full'
        Permutation strategy:
        - 'full': Shuffle all FC values
        - 'columns': Shuffle within columns (preserves target properties)
    output_dir : str, optional
        Output directory
        
    Returns
    -------
    results : dict
        Permutation test results
    """
    
    print(f"\n{'='*70}")
    print(f"Permutation Test for Subject {subject_id}")
    print(f"  Permutations: {n_permutations}")
    print(f"  Mode: {permute_mode}")
    print(f"{'='*70}")
    
    n_parcels = 360
    n_conditions = task_betas.shape[1]
    n_sessions = 2
    
    # Store results
    all_permuted_rsms = []
    all_permuted_dimensionality = np.zeros((n_parcels, n_permutations))
    all_permuted_d_trans_hat = np.zeros((n_parcels, n_permutations))
    
    for target_idx in tqdm(range(n_parcels), desc='Parcels'):
        
        # Get target and source vertices (same as in actflow_prediction)
        target_vert = np.where(glasser == target_idx + 1)[0]
        
        dil_file = f'{glasser_parcels_dir}GlasserParcel{target_idx+1}_dilated_10mm.dscalar.nii'
        import nibabel as nib
        dilated_mask = np.squeeze(nib.load(dil_file).get_fdata())
        target_vert_dilated = np.where(dilated_mask == 1)[0]
        
        connected_sources = np.where(parcel_fc[target_idx, :] != 0)[0]
        
        source_vert_list = []
        for source_idx in connected_sources:
            source_vert = np.where(glasser == source_idx + 1)[0]
            source_vert = source_vert[~np.isin(source_vert, target_vert_dilated)]
            source_vert_list.append(source_vert)
        
        source_vert = np.concatenate(source_vert_list)
        
        # Get original FC
        fc_matrix = fc_dict[target_idx]
        
        # Predict with permuted FC for both sessions
        permuted_betas = np.zeros((n_conditions, len(target_vert), n_sessions, n_permutations))
        
        for sess_idx in range(n_sessions):
            source_betas = task_betas[sess_idx, :, source_vert]
            
            permuted_betas[:, :, sess_idx, :] = predict_with_permuted_fc(
                source_betas, fc_matrix, n_permutations, permute_mode=permute_mode
            )
        
        # Compute permuted RSMs
        permuted_rsms = compute_permuted_rsms(permuted_betas, rsm_computation_module)
        all_permuted_rsms.append(permuted_rsms)
        
        # Compute permuted dimensionality
        for perm_idx in range(n_permutations):
            all_permuted_dimensionality[target_idx, perm_idx] = \
                rsm_computation_module.get_dimensionality(permuted_rsms[perm_idx])
        
        # Compute permuted d_trans_hat
        all_permuted_d_trans_hat[target_idx, :] = compute_permuted_transformation_distance(
            observed_rsms, permuted_rsms, target_idx, parcel_fc
        )
    
    results = {
        'permuted_rsms': all_permuted_rsms,  # List of (n_permutations, n_cond, n_cond) per parcel
        'permuted_dimensionality': all_permuted_dimensionality,  # (n_parcels, n_permutations)
        'permuted_d_trans_hat': all_permuted_d_trans_hat,  # (n_parcels, n_permutations)
        'n_permutations': n_permutations,
        'permute_mode': permute_mode
    }
    
    # Save
    if output_dir is not None:
        output_file = f'{output_dir}{subject_id}_actflow_permutation_{permute_mode}.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved permutation results to: {output_file}")
    
    print(f"{'='*70}\n")
    
    return results


def compute_permutation_pvalues(observed_values, permuted_distribution):
    """
    Compute p-values from permutation distribution.
    
    Two-tailed test: proportion of permutations more extreme than observed.
    
    Parameters
    ----------
    observed_values : ndarray, shape (n_parcels,)
        Observed metric values
    permuted_distribution : ndarray, shape (n_parcels, n_permutations)
        Permuted metric values
        
    Returns
    -------
    pvalues : ndarray, shape (n_parcels,)
        Two-tailed p-values
    """
    
    n_parcels = observed_values.shape[0]
    n_permutations = permuted_distribution.shape[1]
    
    pvalues = np.zeros(n_parcels)
    
    for parcel_idx in range(n_parcels):
        # Count how many permutations are more extreme
        obs = observed_values[parcel_idx]
        perm_dist = permuted_distribution[parcel_idx, :]
        
        # Two-tailed: count both tails
        n_extreme = np.sum(np.abs(perm_dist - np.mean(perm_dist)) >= 
                          np.abs(obs - np.mean(perm_dist)))
        
        pvalues[parcel_idx] = (n_extreme + 1) / (n_permutations + 1)  # +1 for continuity
    
    return pvalues
