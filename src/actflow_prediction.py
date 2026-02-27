"""
Activity Flow Modeling for Task Predictions.

Predicts target region task activations as weighted sums of source region activations,
weighted by functional connectivity. Can then compute predicted RSMs and dimensionality.

Data Flow:
-----------
1. Task GLM (task_glm.py) → Task betas per run/session
2. Load & group betas → (2 session groups, 96 conditions, n_vertices)
3. Vertex-wise FC (vertexwise_fc_pcr.py) → FC from sources to each target
4. Parcel-level FC (graphical_lasso_cv.py) → Defines which regions connect
5. Activity Flow Prediction → predicted_betas = source_betas × FC^T
6. Predicted RSMs (reuses rsm_computation.py) → Cross-validated RSMs
7. Predicted Dimensionality (reuses rsm_computation.py) → Participation ratio

This module integrates outputs from earlier pipeline stages and reuses existing
RSM/dimensionality computation functions to ensure consistency.
"""

import numpy as np
import pickle
import nibabel as nib
from tqdm import tqdm


def load_mdtb_task_betas(subject_id, beta_dir, task_subset_indices, space='vertex'):
    """
    Load MDTB task betas following the same approach as RSM computation.
    
    Loads from task GLM outputs, groups by sessions (a vs b), and selects subset.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    beta_dir : str
        Directory containing task beta HDF5 files (from task_glm.py output)
    task_subset_indices : ndarray
        Indices of task conditions to select (e.g., 96 active visual tasks)
    space : str, default='vertex'
        Data space ('vertex' for activity flow)
        
    Returns
    -------
    betas : ndarray, shape (2, n_conditions_subset, n_vertices)
        Task betas organized as [session_group, condition, vertex]
        - betas[0] = a-sessions (a1 + a2)
        - betas[1] = b-sessions (b1 + b2)
    """
    
    import h5py
    
    session_ids = ['a1', 'a2', 'b1', 'b2']
    n_runs = 8
    
    # Collect betas separately for a-sessions and b-sessions
    betas_a_all = []
    betas_b_all = []
    
    for session_id in session_ids:
        for run_id in range(1, n_runs + 1):
            filename = f'{beta_dir}{subject_id}_{session_id}_tfMRI_{space}_betaseries_bold{run_id}.h5'
            
            # Load betas
            with h5py.File(filename, 'r') as f:
                run_betas = f['betas'][:]  # Shape: (n_conditions, n_vertices)
            
            # Separate a-sessions from b-sessions
            if session_id in ['a1', 'a2']:
                betas_a_all.append(run_betas)
            else:  # b1, b2
                betas_b_all.append(run_betas)
    
    # Concatenate across runs within each session type
    betas_a = np.vstack(betas_a_all)  # Shape: (n_conditions_a, n_vertices)
    betas_b = np.vstack(betas_b_all)  # Shape: (n_conditions_b, n_vertices)
    
    # Select task subset (96 active visual tasks)
    betas_a_subset = betas_a[task_subset_indices, :]
    betas_b_subset = betas_b[task_subset_indices, :]
    
    # Stack into (2, n_conditions, n_vertices) format
    # [0] = a-sessions, [1] = b-sessions
    betas = np.stack([betas_a_subset, betas_b_subset], axis=0)
    
    return betas


def predict_target_betas_actflow(source_betas, fc_matrix):
    """
    Predict target region betas using activity flow mapping.
    
    Activity flow equation: predicted_target = source_betas × FC^T
    
    Parameters
    ----------
    source_betas : ndarray, shape (n_conditions, n_source_vertices)
        Beta weights from source regions (connected to target)
    fc_matrix : ndarray, shape (n_target_vertices, n_source_vertices)
        Functional connectivity from source to target vertices
        
    Returns
    -------
    predicted_betas : ndarray, shape (n_conditions, n_target_vertices)
        Predicted beta weights for target region
    """
    
    # Activity flow: multiply source activations by connectivity weights
    # Shape: (n_conditions, n_source) × (n_source, n_target) = (n_conditions, n_target)
    predicted_betas = np.dot(source_betas, fc_matrix.T)
    
    return predicted_betas


def compute_actflow_predictions_single_subject(subject_id, task_betas, fc_dict, 
                                                parcel_fc, glasser, glasser_parcels_dir,
                                                output_dir=None):
    """
    Compute activity flow predictions for all target regions in one subject.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    task_betas : ndarray, shape (n_sessions, n_conditions, n_vertices)
        Task beta weights (from task GLM)
    fc_dict : dict
        Vertex-wise FC dictionary (from vertexwise_fc_pcr.py)
        Keys = target region indices, Values = FC matrices
    parcel_fc : ndarray, shape (360, 360)
        Parcel-level FC skeleton (defines which regions are connected)
    glasser : ndarray, shape (n_vertices,)
        Glasser parcellation labels
    glasser_parcels_dir : str
        Directory with dilated parcel masks
    output_dir : str, optional
        Directory to save predictions
        
    Returns
    -------
    all_predictions : list of ndarray
        List of predicted betas for each target region
        Each element: shape (n_conditions, n_target_vertices, n_sessions)
    """
    
    n_parcels = 360
    n_sessions = task_betas.shape[0]
    n_conditions = task_betas.shape[1]
    
    all_predictions = []
    
    for target_idx in tqdm(range(n_parcels), desc=f'Subject {subject_id}'):
        
        # Get target vertices
        target_vert = np.where(glasser == target_idx + 1)[0]
        
        # Get dilated target (to exclude from sources)
        dil_file = f'{glasser_parcels_dir}GlasserParcel{target_idx+1}_dilated_10mm.dscalar.nii'
        dilated_mask = np.squeeze(nib.load(dil_file).get_fdata())
        target_vert_dilated = np.where(dilated_mask == 1)[0]
        
        # Get connected source regions (non-zero in parcel FC)
        connected_sources = np.where(parcel_fc[target_idx, :] != 0)[0]
        
        # Compile source vertices from all connected regions
        source_vert_list = []
        for source_idx in connected_sources:
            # Vertices in this source region
            source_vert = np.where(glasser == source_idx + 1)[0]
            # Exclude vertices in dilated target
            source_vert = source_vert[~np.isin(source_vert, target_vert_dilated)]
            source_vert_list.append(source_vert)
        
        source_vert = np.concatenate(source_vert_list)
        
        # Get FC for this target
        fc_matrix = fc_dict[target_idx]  # Shape: (n_target_vertices, n_source_vertices)
        
        # Predict for each session
        predicted_target = np.zeros((n_conditions, len(target_vert), n_sessions))
        
        for sess_idx in range(n_sessions):
            # Get source betas for this session
            source_betas = task_betas[sess_idx, :, source_vert]  # (n_conditions, n_source_vertices)
            
            # Activity flow prediction
            predicted_target[:, :, sess_idx] = predict_target_betas_actflow(source_betas, fc_matrix)
        
        all_predictions.append(predicted_target)
    
    # Optionally save
    if output_dir is not None:
        output_file = f'{output_dir}{subject_id}_actflow_predictions.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(all_predictions, f)
        print(f'Saved predictions to: {output_file}')
    
    return all_predictions


def compute_predicted_rsm_from_actflow(predicted_betas, rsm_computation_module):
    """
    Compute RSM from activity flow predicted betas.
    
    Uses the same RSM computation as for observed data (cross-validated between sessions).
    
    Parameters
    ----------
    predicted_betas : ndarray, shape (n_conditions, n_vertices, n_sessions)
        Predicted betas from activity flow for one region
    rsm_computation_module : module
        The rsm_computation module (to reuse compute_rsm_single_region function)
        
    Returns
    -------
    rsm : ndarray, shape (n_conditions, n_conditions)
        Cross-validated RSM from predicted betas
    """
    
    # Extract session 0 and session 1 betas
    betas_session0 = predicted_betas[:, :, 0]  # (n_conditions, n_vertices)
    betas_session1 = predicted_betas[:, :, 1]  # (n_conditions, n_vertices)
    
    # Use existing RSM computation (normalize + dot product for cosine similarity)
    rsm = rsm_computation_module.compute_rsm_single_region(betas_session0, betas_session1)
    
    return rsm


def compute_all_predicted_rsms(all_predictions, rsm_computation_module):
    """
    Compute RSMs for all regions from activity flow predictions.
    
    Parameters
    ----------
    all_predictions : list of ndarray
        Predicted betas for each target region
        Each element: shape (n_conditions, n_target_vertices, n_sessions)
    rsm_computation_module : module
        The rsm_computation module
        
    Returns
    -------
    predicted_rsms : ndarray, shape (n_parcels, n_conditions, n_conditions)
        Predicted RSMs for all parcels
    """
    
    n_parcels = len(all_predictions)
    n_conditions = all_predictions[0].shape[0]
    
    predicted_rsms = np.zeros((n_parcels, n_conditions, n_conditions))
    
    for parcel_idx in range(n_parcels):
        predicted_rsms[parcel_idx] = compute_predicted_rsm_from_actflow(
            all_predictions[parcel_idx], rsm_computation_module
        )
    
    return predicted_rsms


def compute_predicted_dimensionality(predicted_rsms, rsm_computation_module):
    """
    Compute dimensionality from predicted RSMs.
    
    Uses the same dimensionality function as for observed RSMs.
    
    Parameters
    ----------
    predicted_rsms : ndarray, shape (n_parcels, n_conditions, n_conditions)
        Predicted RSMs
    rsm_computation_module : module
        The rsm_computation module (to reuse get_dimensionality function)
        
    Returns
    -------
    dimensionalities : ndarray, shape (n_parcels,)
        Predicted dimensionality for each parcel
    """
    
    # Reuse the existing dimensionality computation
    dimensionalities = rsm_computation_module.compute_dimensionality_all_regions(predicted_rsms)
    
    return dimensionalities


def compute_double_crossvalidated_rsm(observed_betas, predicted_betas, parcel_idx, glasser,
                                      rsm_computation_module):
    """
    Compute double cross-validated RSMs: cross-validated between sessions AND obs/pred.
    
    Creates two RSMs:
    - RSM_1: Observed-Session0 × Predicted-Session1
    - RSM_2: Observed-Session1 × Predicted-Session0
    
    This ensures patterns are truly shared between observed and predicted, not just
    session-specific artifacts.
    
    Parameters
    ----------
    observed_betas : ndarray, shape (2, n_conditions, n_vertices)
        Observed task betas (from load_mdtb_task_betas)
    predicted_betas : ndarray, shape (n_conditions, n_target_vertices, 2)
        Predicted betas for this region (from activity flow)
    parcel_idx : int
        Index of the target parcel
    glasser : ndarray
        Glasser parcellation labels
    rsm_computation_module : module
        The rsm_computation module
        
    Returns
    -------
    rsm_1 : ndarray, shape (n_conditions, n_conditions)
        Observed-Session0 × Predicted-Session1
    rsm_2 : ndarray, shape (n_conditions, n_conditions)
        Observed-Session1 × Predicted-Session0
    """
    
    # Get vertices in this parcel
    parcel_vertices = np.where(glasser == parcel_idx + 1)[0]
    
    # Extract observed betas for this region
    obs_session0 = observed_betas[0, :, parcel_vertices]  # (n_conditions, n_vertices)
    obs_session1 = observed_betas[1, :, parcel_vertices]
    
    # Extract predicted betas for this region
    pred_session0 = predicted_betas[:, :, 0]  # (n_conditions, n_target_vertices)
    pred_session1 = predicted_betas[:, :, 1]
    
    # RSM_1: Observed-Session0 × Predicted-Session1
    rsm_1 = rsm_computation_module.compute_rsm_single_region(obs_session0, pred_session1)
    
    # RSM_2: Observed-Session1 × Predicted-Session0
    rsm_2 = rsm_computation_module.compute_rsm_single_region(obs_session1, pred_session0)
    
    return rsm_1, rsm_2


def compute_all_double_crossvalidated_rsms(observed_betas, predicted_betas_list, glasser,
                                           rsm_computation_module):
    """
    Compute double cross-validated RSMs for all regions.
    
    Parameters
    ----------
    observed_betas : ndarray, shape (2, n_conditions, n_vertices)
        Observed task betas
    predicted_betas_list : list of ndarray
        Predicted betas for each region
        Each element: shape (n_conditions, n_target_vertices, 2)
    glasser : ndarray
        Glasser parcellation labels
    rsm_computation_module : module
        The rsm_computation module
        
    Returns
    -------
    rsms_1 : ndarray, shape (n_parcels, n_conditions, n_conditions)
        Observed-Session0 × Predicted-Session1 for all parcels
    rsms_2 : ndarray, shape (n_parcels, n_conditions, n_conditions)
        Observed-Session1 × Predicted-Session0 for all parcels
    """
    
    n_parcels = len(predicted_betas_list)
    n_conditions = predicted_betas_list[0].shape[0]
    
    rsms_1 = np.zeros((n_parcels, n_conditions, n_conditions))
    rsms_2 = np.zeros((n_parcels, n_conditions, n_conditions))
    
    for parcel_idx in range(n_parcels):
        rsm_1, rsm_2 = compute_double_crossvalidated_rsm(
            observed_betas, predicted_betas_list[parcel_idx], parcel_idx,
            glasser, rsm_computation_module
        )
        rsms_1[parcel_idx] = rsm_1
        rsms_2[parcel_idx] = rsm_2
    
    return rsms_1, rsms_2


def compute_double_crossvalidated_dimensionality(rsms_1, rsms_2, rsm_computation_module):
    """
    Compute dimensionality from double cross-validated RSMs.
    
    Computes dimensionality separately for each RSM combination, then averages:
    - dim_1 = dimensionality(Obs-Sess0 × Pred-Sess1)
    - dim_2 = dimensionality(Obs-Sess1 × Pred-Sess0)
    - final_dim = (dim_1 + dim_2) / 2
    
    This averaging ensures robust dimensionality estimates that capture what's
    truly shared between observed and predicted patterns.
    
    Parameters
    ----------
    rsms_1 : ndarray, shape (n_parcels, n_conditions, n_conditions)
        Observed-Session0 × Predicted-Session1 RSMs
    rsms_2 : ndarray, shape (n_parcels, n_conditions, n_conditions)
        Observed-Session1 × Predicted-Session0 RSMs
    rsm_computation_module : module
        The rsm_computation module
        
    Returns
    -------
    dimensionality_avg : ndarray, shape (n_parcels,)
        Average dimensionality from both double-CV combinations
    dimensionality_1 : ndarray, shape (n_parcels,)
        Dimensionality from RSM_1 (for diagnostics)
    dimensionality_2 : ndarray, shape (n_parcels,)
        Dimensionality from RSM_2 (for diagnostics)
    """
    
    # Compute dimensionality for each combination
    dimensionality_1 = rsm_computation_module.compute_dimensionality_all_regions(rsms_1)
    dimensionality_2 = rsm_computation_module.compute_dimensionality_all_regions(rsms_2)
    
    # Average across the two combinations
    dimensionality_avg = (dimensionality_1 + dimensionality_2) / 2
    
    return dimensionality_avg, dimensionality_1, dimensionality_2


def process_subject_actflow_double_cv(subject_id, task_betas, fc_dict, parcel_fc, glasser,
                                       glasser_parcels_dir, rsm_computation_module,
                                       output_dir=None):
    """
    Complete activity flow pipeline with double cross-validation.
    
    Computes:
    1. Predicted betas (activity flow)
    2. Standard predicted RSMs (Pred-Sess0 × Pred-Sess1)
    3. Double-CV RSMs (Obs-Sess0 × Pred-Sess1 and Obs-Sess1 × Pred-Sess0)
    4. Standard predicted dimensionality
    5. Double-CV dimensionality (averaged)
    
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
    glasser : ndarray
        Parcellation labels
    glasser_parcels_dir : str
        Dilated parcel directory
    rsm_computation_module : module
        Import of rsm_computation module
    output_dir : str, optional
        Output directory
        
    Returns
    -------
    results : dict
        Dictionary with all results including double-CV RSMs and dimensionality
    """
    
    print(f"\n{'='*70}")
    print(f"Processing Activity Flow (Double CV) for Subject {subject_id}")
    print(f"{'='*70}")
    
    # Step 1: Predict betas
    print("\nStep 1: Computing activity flow predictions...")
    predicted_betas = compute_actflow_predictions_single_subject(
        subject_id, task_betas, fc_dict, parcel_fc, glasser, 
        glasser_parcels_dir, output_dir=None  # Don't save intermediate
    )
    
    # Step 2: Standard predicted RSMs
    print("\nStep 2: Computing standard predicted RSMs...")
    predicted_rsms = compute_all_predicted_rsms(predicted_betas, rsm_computation_module)
    print(f"Standard predicted RSMs shape: {predicted_rsms.shape}")
    
    # Step 3: Double cross-validated RSMs
    print("\nStep 3: Computing double cross-validated RSMs...")
    double_cv_rsms_1, double_cv_rsms_2 = compute_all_double_crossvalidated_rsms(
        task_betas, predicted_betas, glasser, rsm_computation_module
    )
    print(f"Double-CV RSM_1 shape: {double_cv_rsms_1.shape}")
    print(f"Double-CV RSM_2 shape: {double_cv_rsms_2.shape}")
    
    # Step 4: Standard predicted dimensionality
    print("\nStep 4: Computing standard predicted dimensionality...")
    predicted_dimensionality = compute_predicted_dimensionality(predicted_rsms, rsm_computation_module)
    print(f"Standard predicted dimensionality shape: {predicted_dimensionality.shape}")
    
    # Step 5: Double-CV dimensionality
    print("\nStep 5: Computing double cross-validated dimensionality...")
    double_cv_dim_avg, double_cv_dim_1, double_cv_dim_2 = compute_double_crossvalidated_dimensionality(
        double_cv_rsms_1, double_cv_rsms_2, rsm_computation_module
    )
    print(f"Double-CV dimensionality (averaged) shape: {double_cv_dim_avg.shape}")
    print(f"  Correlation between dim_1 and dim_2: {np.corrcoef(double_cv_dim_1, double_cv_dim_2)[0,1]:.3f}")
    
    # Save results
    if output_dir is not None:
        # Standard predictions
        with open(f'{output_dir}{subject_id}_actflow_predictions.pkl', 'wb') as f:
            pickle.dump(predicted_betas, f)
        
        with open(f'{output_dir}{subject_id}_predicted_RSM_actflow.pkl', 'wb') as f:
            pickle.dump(predicted_rsms, f)
        
        with open(f'{output_dir}{subject_id}_predicted_dimensionality_actflow.pkl', 'wb') as f:
            pickle.dump(predicted_dimensionality, f)
        
        # Double-CV RSMs (keep both separate)
        with open(f'{output_dir}{subject_id}_doubleCV_RSM_1_actflow.pkl', 'wb') as f:
            pickle.dump(double_cv_rsms_1, f)
        
        with open(f'{output_dir}{subject_id}_doubleCV_RSM_2_actflow.pkl', 'wb') as f:
            pickle.dump(double_cv_rsms_2, f)
        
        # Double-CV dimensionality (averaged + individual components)
        double_cv_dim_results = {
            'averaged': double_cv_dim_avg,
            'dim_1': double_cv_dim_1,
            'dim_2': double_cv_dim_2
        }
        with open(f'{output_dir}{subject_id}_doubleCV_dimensionality_actflow.pkl', 'wb') as f:
            pickle.dump(double_cv_dim_results, f)
        
        print(f"\nSaved all results to: {output_dir}")
    
    results = {
        'predicted_betas': predicted_betas,
        'predicted_rsms': predicted_rsms,
        'predicted_dimensionality': predicted_dimensionality,
        'double_cv_rsms_1': double_cv_rsms_1,
        'double_cv_rsms_2': double_cv_rsms_2,
        'double_cv_dimensionality_avg': double_cv_dim_avg,
        'double_cv_dimensionality_1': double_cv_dim_1,
        'double_cv_dimensionality_2': double_cv_dim_2
    }
    
    print(f"\n{'='*70}")
    print(f"Completed Activity Flow (Double CV) for Subject {subject_id}")
    print(f"{'='*70}\n")
    
    return results


def process_subject_actflow(subject_id, task_betas, fc_dict, parcel_fc, glasser,
                            glasser_parcels_dir, rsm_computation_module,
                            output_dir=None):
    """
    Complete activity flow pipeline for one subject.
    
    Computes: predicted betas → predicted RSMs → predicted dimensionality
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    task_betas : ndarray, shape (n_sessions, n_conditions, n_vertices)
        Observed task betas
    fc_dict : dict
        Vertex-wise FC dictionary
    parcel_fc : ndarray, shape (360, 360)
        Parcel-level FC
    glasser : ndarray
        Parcellation labels
    glasser_parcels_dir : str
        Dilated parcel directory
    rsm_computation_module : module
        Import of rsm_computation module
    output_dir : str, optional
        Output directory
        
    Returns
    -------
    results : dict
        Dictionary with 'predicted_betas', 'predicted_rsms', 'predicted_dimensionality'
    """
    
    print(f"\n{'='*70}")
    print(f"Processing Activity Flow for Subject {subject_id}")
    print(f"{'='*70}")
    
    # Step 1: Predict betas
    print("\nStep 1: Computing activity flow predictions...")
    predicted_betas = compute_actflow_predictions_single_subject(
        subject_id, task_betas, fc_dict, parcel_fc, glasser, 
        glasser_parcels_dir, output_dir
    )
    
    # Step 2: Compute predicted RSMs
    print("\nStep 2: Computing predicted RSMs...")
    predicted_rsms = compute_all_predicted_rsms(predicted_betas, rsm_computation_module)
    print(f"Predicted RSMs shape: {predicted_rsms.shape}")
    
    # Step 3: Compute predicted dimensionality
    print("\nStep 3: Computing predicted dimensionality...")
    predicted_dimensionality = compute_predicted_dimensionality(predicted_rsms, rsm_computation_module)
    print(f"Predicted dimensionality shape: {predicted_dimensionality.shape}")
    
    # Save results
    if output_dir is not None:
        # Save predicted RSMs
        rsm_file = f'{output_dir}{subject_id}_predicted_RSM_actflow.pkl'
        with open(rsm_file, 'wb') as f:
            pickle.dump(predicted_rsms, f)
        print(f"\nSaved predicted RSMs to: {rsm_file}")
        
        # Save predicted dimensionality
        dim_file = f'{output_dir}{subject_id}_predicted_dimensionality_actflow.pkl'
        with open(dim_file, 'wb') as f:
            pickle.dump(predicted_dimensionality, f)
        print(f"Saved predicted dimensionality to: {dim_file}")
    
    results = {
        'predicted_betas': predicted_betas,
        'predicted_rsms': predicted_rsms,
        'predicted_dimensionality': predicted_dimensionality
    }
    
    print(f"\n{'='*70}")
    print(f"Completed Activity Flow for Subject {subject_id}")
    print(f"{'='*70}\n")
    
    return results
