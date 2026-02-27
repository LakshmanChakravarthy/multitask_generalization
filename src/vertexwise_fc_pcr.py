"""
Vertex-wise FC estimation using PC regression with CV-optimized number of components.

This module estimates functional connectivity at vertex resolution, restricted to 
the parcel-level FC skeleton (from graphical lasso). Uses a two-round coarse-to-fine 
cross-validation approach to find the optimal number of PCs per subject.

Activity flow terminology:
- Target region/parcel: The region receiving input (prediction target)
- Source vertices: Vertices in connected regions that predict target activity
"""

import numpy as np
import pickle
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error


def compute_vertexwise_fc_with_cv(subj_id, sub_vert_data, sub_parcel_data, 
                                    parcel_fc, glasser, glasser_parcels_dir,
                                    n_parcels=360):
    """
    Compute vertex-wise FC for a subject using CV-optimized number of PCs.
    
    Parameters
    ----------
    subj_id : str
        Subject identifier
    sub_vert_data : ndarray, shape (n_vertices, n_timepoints)
        Vertex-level resting-state fMRI data
    sub_parcel_data : ndarray, shape (n_parcels, n_timepoints)
        Parcel-level resting-state fMRI data
    parcel_fc : ndarray, shape (n_parcels, n_parcels)
        Parcel-level FC skeleton (from graphical lasso)
    glasser : ndarray, shape (n_vertices,)
        Glasser parcellation labels
    glasser_parcels_dir : str
        Directory containing dilated parcel files
    n_parcels : int, default=360
        Number of parcels
        
    Returns
    -------
    vertFC_dict : dict
        Keys are target parcel indices (0-359), values are FC matrices of shape
        (n_source_vertices, n_target_vertices)
    vertFC_rsq_arr : ndarray, shape (n_parcels,)
        R-squared values for each target parcel
    optimal_nPC : int
        Optimal number of PCs determined by CV
    """
    
    # Step 1: Prepare X (source) and Y (target) matrices for all target parcels
    X_dict, Y_dict = _prepare_vertex_timeseries(
        sub_vert_data, parcel_fc, glasser, glasser_parcels_dir, n_parcels
    )
    
    # Step 2: Two-round CV to find optimal nPCs
    print("CV Round 1: Coarse search...")
    optimal_nPC_round1 = find_optimal_nPCs_coarse(X_dict, Y_dict, step=100)
    print(f"Round 1 optimal nPCs: {optimal_nPC_round1}")
    
    print("CV Round 2: Fine search...")
    optimal_nPC = find_optimal_nPCs_fine(
        X_dict, Y_dict, center=optimal_nPC_round1, window=50, step=10
    )
    print(f"Final optimal nPCs: {optimal_nPC}")
    
    # Step 3: Compute vertex-wise FC using optimal nPCs
    print("Computing vertex-wise FC with optimal nPCs...")
    vertFC_dict = {}
    vertFC_rsq_arr = np.zeros(n_parcels)
    
    for target_roi_idx in range(n_parcels):
        print(f'Target parcel: {target_roi_idx}')
        
        X = X_dict[target_roi_idx]
        Y = Y_dict[target_roi_idx]
        
        betasPCR, rsq = compute_pcr_fc(X, Y, n_components=optimal_nPC)
        
        vertFC_dict[target_roi_idx] = betasPCR
        vertFC_rsq_arr[target_roi_idx] = rsq
    
    return vertFC_dict, vertFC_rsq_arr, optimal_nPC


def _prepare_vertex_timeseries(sub_vert_data, parcel_fc, glasser, 
                                glasser_parcels_dir, n_parcels):
    """
    Prepare source (X) and target (Y) vertex timeseries for all target parcels.
    
    For each target parcel:
    - X contains timeseries of vertices from all connected source parcels
    - Y contains timeseries of vertices within the target parcel
    - Vertices within dilated target parcel are excluded from source vertices
    
    Parameters
    ----------
    sub_vert_data : ndarray, shape (n_vertices, n_timepoints)
        Vertex-level fMRI data
    parcel_fc : ndarray, shape (n_parcels, n_parcels)
        Parcel-level FC skeleton
    glasser : ndarray, shape (n_vertices,)
        Glasser parcellation labels
    glasser_parcels_dir : str
        Directory containing dilated parcel files
    n_parcels : int
        Number of parcels
        
    Returns
    -------
    X_dict : dict
        Keys are target parcel indices, values are source vertex timeseries
        of shape (n_source_vertices, n_timepoints)
    Y_dict : dict
        Keys are target parcel indices, values are target vertex timeseries
        of shape (n_target_vertices, n_timepoints)
    """
    
    X_dict = {}
    Y_dict = {}
    
    for target_roi_idx in range(n_parcels):
        print(f'Preparing data for target parcel: {target_roi_idx}')
        
        # Vertices in target parcel
        target_vert = np.where(glasser == target_roi_idx + 1)[0]
        
        # Vertices in dilated target parcel (10mm dilation)
        dil_dscalar_filename = (glasser_parcels_dir + 
                                f'GlasserParcel{target_roi_idx+1}_dilated_10mm.dscalar.nii')
        dilated_dscalar = np.squeeze(nib.load(dil_dscalar_filename).get_fdata())
        target_vert_dilated = np.where(dilated_dscalar == 1)[0]
        
        # Source parcels with non-zero connections to this target
        connected_source_parcels = np.where(parcel_fc[target_roi_idx, :] != 0)[0]
        
        # Compile source vertices from all connected parcels
        source_vert_list = []
        for source_parcel_idx in connected_source_parcels:
            # Vertices in this source parcel
            this_source_vert = np.where(glasser == source_parcel_idx + 1)[0]
            
            # Remove vertices that fall within dilated target parcel
            this_source_vert = this_source_vert[~np.isin(this_source_vert, target_vert_dilated)]
            
            source_vert_list.append(this_source_vert)
        
        source_vert = np.concatenate(source_vert_list)
        
        # Extract timeseries
        X_dict[target_roi_idx] = sub_vert_data[source_vert, :]  # source vertices
        Y_dict[target_roi_idx] = sub_vert_data[target_vert, :]  # target vertices
    
    return X_dict, Y_dict


def find_optimal_nPCs_coarse(X_dict, Y_dict, step=100):
    """
    First CV round: coarse search for optimal number of PCs.
    
    Parameters
    ----------
    X_dict : dict
        Source vertex timeseries for each target parcel
    Y_dict : dict
        Target vertex timeseries for each target parcel
    step : int, default=100
        Step size for component search
        
    Returns
    -------
    optimal_nPC : int
        Optimal number of PCs from coarse search
    """
    
    n_parcels = len(X_dict)
    mse_cv_all_targets = []
    
    for target_roi_idx in range(n_parcels):
        print(f'CV Round 1 - Target parcel: {target_roi_idx}')
        
        X = X_dict[target_roi_idx]
        Y = Y_dict[target_roi_idx]
        
        # Component range: 1 to min(X.shape) in steps
        n_components_min = 1
        n_components_max = np.min(X.shape)
        component_range = np.arange(n_components_min, n_components_max + 1, step)
        
        mse_cv = _cv_pcr_over_components(X, Y, component_range)
        mse_cv_all_targets.append(mse_cv)
    
    # Average MSE across all target parcels
    mse_cv_all_targets = np.array(mse_cv_all_targets)
    mse_cv_mean = np.mean(mse_cv_all_targets, axis=0)
    
    # Find optimal nPC
    optimal_nPC = component_range[np.argmin(mse_cv_mean)]
    
    return optimal_nPC


def find_optimal_nPCs_fine(X_dict, Y_dict, center, window=50, step=10):
    """
    Second CV round: fine search around the coarse optimum.
    
    Parameters
    ----------
    X_dict : dict
        Source vertex timeseries for each target parcel
    Y_dict : dict
        Target vertex timeseries for each target parcel
    center : int
        Center point for fine search (from coarse round)
    window : int, default=50
        Search window (±window around center)
    step : int, default=10
        Step size for component search
        
    Returns
    -------
    optimal_nPC : int
        Optimal number of PCs from fine search
    """
    
    n_parcels = len(X_dict)
    mse_cv_all_targets = []
    
    for target_roi_idx in range(n_parcels):
        print(f'CV Round 2 - Target parcel: {target_roi_idx}')
        
        X = X_dict[target_roi_idx]
        Y = Y_dict[target_roi_idx]
        
        # Component range: center ± window in steps
        n_components_min = max(1, center - window)
        n_components_max = center + window
        component_range = np.arange(n_components_min, n_components_max + 1, step)
        
        mse_cv = _cv_pcr_over_components(X, Y, component_range)
        mse_cv_all_targets.append(mse_cv)
    
    # Average MSE across all target parcels
    mse_cv_all_targets = np.array(mse_cv_all_targets)
    mse_cv_mean = np.mean(mse_cv_all_targets, axis=0)
    
    # Find optimal nPC
    optimal_nPC = component_range[np.argmin(mse_cv_mean)]
    
    return optimal_nPC


def _cv_pcr_over_components(X, Y, component_range):
    """
    Run cross-validated PCR over a range of component numbers.
    
    Parameters
    ----------
    X : ndarray, shape (n_vertices, n_timepoints)
        Source vertices timeseries
    Y : ndarray, shape (n_vertices, n_timepoints)
        Target vertices timeseries
    component_range : ndarray
        Array of component numbers to test
        
    Returns
    -------
    mse_cv_vals : ndarray
        Cross-validation MSE for each component number
    """
    
    # Demean along time dimension
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_demeaned = (X - X_mean).T  # Shape: (n_timepoints, n_vertices)
    
    Y_mean = np.mean(Y, axis=1, keepdims=True)
    Y_demeaned = (Y - Y_mean).T  # Shape: (n_timepoints, n_vertices)
    
    # Run PCA on X once
    pca = PCA()
    X_pca_full = pca.fit_transform(X_demeaned)
    
    # Test each component number
    mse_cv_vals = np.zeros(len(component_range))
    for idx, n_comp in enumerate(component_range):
        mse_cv_vals[idx] = _pcr_cv_single(X_pca_full, Y_demeaned, n_comp, cv=10)
    
    return mse_cv_vals


def _pcr_cv_single(X_pca_full, Y, n_components, cv=10):
    """
    Cross-validate PCR for a single number of components.
    
    Parameters
    ----------
    X_pca_full : ndarray, shape (n_timepoints, n_components_max)
        Full PCA-transformed X
    Y : ndarray, shape (n_timepoints, n_vertices)
        Target timeseries
    n_components : int
        Number of components to use
    cv : int, default=10
        Number of cross-validation folds
        
    Returns
    -------
    mse_cv : float
        Cross-validation mean squared error
    """
    
    # Select first n_components
    X_pca = X_pca_full[:, :n_components]
    
    # Create and fit linear regression
    regr = LinearRegression()
    regr.fit(X_pca, Y)
    
    # Cross-validation prediction
    Y_cv = cross_val_predict(regr, X_pca, Y, cv=cv)
    
    # Calculate MSE
    mse_cv = mean_squared_error(Y, Y_cv)
    
    return mse_cv


def compute_pcr_fc(X, Y, n_components):
    """
    Compute PC regression-based functional connectivity.
    
    Parameters
    ----------
    X : ndarray, shape (n_source_vertices, n_timepoints)
        Source vertices timeseries
    Y : ndarray, shape (n_target_vertices, n_timepoints)
        Target vertices timeseries
    n_components : int
        Number of PCs to use
        
    Returns
    -------
    betas_pcr : ndarray, shape (n_target_vertices, n_source_vertices)
        PCR-based FC weights
    rsq : float
        R-squared of the regression
    """
    
    # Demean along time dimension
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_demeaned = (X - X_mean).T  # Shape: (n_timepoints, n_vertices)
    
    Y_mean = np.mean(Y, axis=1, keepdims=True)
    Y_demeaned = (Y - Y_mean).T  # Shape: (n_timepoints, n_vertices)
    
    # PCA on X
    pca = PCA(n_components)
    X_reduced = pca.fit_transform(X_demeaned)
    
    # Regression of Y on reduced X
    reg = LinearRegression().fit(X_reduced, Y_demeaned)
    
    # Transform regression coefficients back to original space
    betas_pcr = pca.inverse_transform(reg.coef_)
    rsq = reg.score(X_reduced, Y_demeaned)
    
    return betas_pcr, rsq


def save_vertexwise_fc(vertFC_dict, vertFC_rsq_arr, optimal_nPC, 
                       subject_id, output_dir):
    """
    Save vertex-wise FC results to pickle file.
    
    Parameters
    ----------
    vertFC_dict : dict
        Vertex-wise FC matrices for each target parcel
    vertFC_rsq_arr : ndarray
        R-squared values for each target parcel
    optimal_nPC : int
        Optimal number of PCs used
    subject_id : str
        Subject identifier
    output_dir : str
        Output directory path
    """
    
    filename = f'sub{subject_id}_vertFC_CVoptimal_nPCs.pkl'
    filepath = output_dir + filename
    
    with open(filepath, 'wb') as f:
        pickle.dump(vertFC_dict, f)
        pickle.dump(vertFC_rsq_arr, f)
        pickle.dump(optimal_nPC, f)
    
    print(f"Saved vertex-wise FC to: {filepath}")


# Main execution function
def process_subject(subject_id, sub_vert_data, sub_parcel_data, parcel_fc,
                   glasser, glasser_parcels_dir, output_dir, n_parcels=360):
    """
    Complete pipeline for one subject: CV optimization + vertex-wise FC estimation.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    sub_vert_data : ndarray, shape (n_vertices, n_timepoints)
        Vertex-level resting-state fMRI data
    sub_parcel_data : ndarray, shape (n_parcels, n_timepoints)
        Parcel-level resting-state fMRI data (currently unused but kept for API)
    parcel_fc : ndarray, shape (n_parcels, n_parcels)
        Parcel-level FC skeleton (from graphical lasso)
    glasser : ndarray, shape (n_vertices,)
        Glasser parcellation labels
    glasser_parcels_dir : str
        Directory containing dilated parcel files
    output_dir : str
        Output directory for results
    n_parcels : int, default=360
        Number of parcels
        
    Returns
    -------
    None (saves results to file)
    """
    
    print(f"\nProcessing subject {subject_id}")
    print("=" * 60)
    
    # Compute vertex-wise FC with CV-optimized nPCs
    vertFC_dict, vertFC_rsq_arr, optimal_nPC = compute_vertexwise_fc_with_cv(
        subject_id, sub_vert_data, sub_parcel_data, parcel_fc,
        glasser, glasser_parcels_dir, n_parcels
    )
    
    # Save results
    save_vertexwise_fc(vertFC_dict, vertFC_rsq_arr, optimal_nPC,
                      subject_id, output_dir)
    
    print(f"\nCompleted subject {subject_id}")
    print(f"Optimal nPCs: {optimal_nPC}")
    print("=" * 60)
