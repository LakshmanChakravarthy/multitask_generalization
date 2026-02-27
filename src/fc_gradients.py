"""
Functional Connectivity (FC) Gradient Computation.

Computes FC gradients from parcel-level connectivity matrices using PCA.
The gradients represent the dominant axes of variation in connectivity patterns
across brain regions.
"""

import numpy as np
import pickle
from sklearn.decomposition import PCA


def threshold_top_percentile(fc_matrix, percentile=80):
    """
    Retain only top percentile of FC values, set rest to zero.
    
    Parameters
    ----------
    fc_matrix : ndarray, shape (n_parcels, n_parcels)
        Functional connectivity matrix
    percentile : float, default=80
        Percentile threshold (0-100). Values below this percentile are set to 0.
        
    Returns
    -------
    fc_thresholded : ndarray, shape (n_parcels, n_parcels)
        Thresholded FC matrix
    """
    
    # Compute threshold value (using absolute values to capture strong connections)
    threshold = np.percentile(np.abs(fc_matrix), percentile)
    
    # Threshold: keep values above threshold, set rest to zero
    fc_thresholded = fc_matrix.copy()
    fc_thresholded[np.abs(fc_matrix) < threshold] = 0
    
    return fc_thresholded


def compute_fc_gradients(fc_matrix, n_components=2, threshold_percentile=80):
    """
    Compute FC gradients using PCA on thresholded connectivity matrix.
    
    Parameters
    ----------
    fc_matrix : ndarray, shape (n_parcels, n_parcels)
        Functional connectivity matrix (typically averaged across subjects)
    n_components : int, default=2
        Number of principal components (gradients) to extract
    threshold_percentile : float, default=80
        Percentile threshold for retaining strong connections
        
    Returns
    -------
    gradients : ndarray, shape (n_parcels, n_components)
        PC loadings for each region. First column is gradient 1, second is gradient 2, etc.
    explained_variance : ndarray, shape (n_components,)
        Variance explained by each gradient
    pca : PCA object
        Fitted PCA object (for accessing all components if needed)
        
    Notes
    -----
    The gradients represent the dominant axes of variation in connectivity patterns.
    Regions with similar gradient values have similar connectivity profiles.
    """
    
    # Threshold to keep only top connections
    fc_thresholded = threshold_top_percentile(fc_matrix, threshold_percentile)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    gradients = pca.fit_transform(fc_thresholded)
    
    explained_variance = pca.explained_variance_ratio_
    
    return gradients, explained_variance, pca


def compute_fc_gradients_from_subjects(all_subjects_fc, n_components=2, 
                                        threshold_percentile=80):
    """
    Compute FC gradients from multi-subject connectivity matrices.
    
    Averages FC across subjects first, then computes gradients.
    
    Parameters
    ----------
    all_subjects_fc : ndarray, shape (n_subjects, n_parcels, n_parcels)
        FC matrices for all subjects
    n_components : int, default=2
        Number of gradients to extract
    threshold_percentile : float, default=80
        Percentile threshold for retaining strong connections
        
    Returns
    -------
    gradients : ndarray, shape (n_parcels, n_components)
        FC gradients (PC loadings)
    explained_variance : ndarray, shape (n_components,)
        Variance explained by each gradient
    mean_fc : ndarray, shape (n_parcels, n_parcels)
        Average FC matrix across subjects
    pca : PCA object
        Fitted PCA object
    """
    
    # Average FC across subjects
    mean_fc = np.mean(all_subjects_fc, axis=0)
    
    # Compute gradients
    gradients, explained_variance, pca = compute_fc_gradients(
        mean_fc, n_components=n_components, threshold_percentile=threshold_percentile
    )
    
    return gradients, explained_variance, mean_fc, pca


def load_and_compute_fc_gradients(fc_file, n_components=2, threshold_percentile=80):
    """
    Load parcel-level FC from file and compute gradients.
    
    Parameters
    ----------
    fc_file : str
        Path to pickle file containing FC matrices
        Expected format: ndarray of shape (n_subjects, n_parcels, n_parcels)
    n_components : int, default=2
        Number of gradients to extract
    threshold_percentile : float, default=80
        Percentile threshold for retaining strong connections
        
    Returns
    -------
    gradients : ndarray, shape (n_parcels, n_components)
        FC gradients
    explained_variance : ndarray, shape (n_components,)
        Variance explained by each gradient
    mean_fc : ndarray, shape (n_parcels, n_parcels)
        Average FC matrix
    """
    
    # Load FC data
    with open(fc_file, 'rb') as f:
        all_subjects_fc = pickle.load(f)
    
    print(f"Loaded FC data shape: {all_subjects_fc.shape}")
    
    # Compute gradients
    gradients, explained_variance, mean_fc, pca = compute_fc_gradients_from_subjects(
        all_subjects_fc, n_components=n_components, threshold_percentile=threshold_percentile
    )
    
    print(f"\nFC Gradients computed:")
    print(f"  Gradients shape: {gradients.shape}")
    print(f"  Explained variance: {explained_variance}")
    print(f"  Cumulative variance: {np.cumsum(explained_variance)}")
    
    return gradients, explained_variance, mean_fc


def save_fc_gradients(gradients, explained_variance, mean_fc, output_file):
    """
    Save FC gradients to file.
    
    Parameters
    ----------
    gradients : ndarray, shape (n_parcels, n_components)
        FC gradients
    explained_variance : ndarray, shape (n_components,)
        Variance explained by each gradient
    mean_fc : ndarray, shape (n_parcels, n_parcels)
        Average FC matrix
    output_file : str
        Output file path (will save as .npz)
    """
    
    np.savez(output_file,
             gradients=gradients,
             explained_variance=explained_variance,
             mean_fc=mean_fc)
    
    print(f"Saved FC gradients to: {output_file}")


def load_fc_gradients(gradient_file):
    """
    Load FC gradients from file.
    
    Parameters
    ----------
    gradient_file : str
        Path to .npz file containing gradients
        
    Returns
    -------
    gradients : ndarray, shape (n_parcels, n_components)
        FC gradients
    explained_variance : ndarray, shape (n_components,)
        Variance explained by each gradient
    mean_fc : ndarray, shape (n_parcels, n_parcels)
        Average FC matrix
    """
    
    data = np.load(gradient_file)
    
    gradients = data['gradients']
    explained_variance = data['explained_variance']
    mean_fc = data['mean_fc']
    
    return gradients, explained_variance, mean_fc
