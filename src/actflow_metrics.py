"""
Activity Flow Metrics: Transformation and Prediction Distances.

Computes distances between RSMs to evaluate:
1. How much connectivity transforms source information (transformation distance)
2. How well the model captures this transformation (predicted transformation distance)
3. How accurate predictions are (prediction distance)

Key insight: If predicted transformation distance > prediction distance, it's evidence
that connectivity does meaningful transformation rather than just averaging inputs.
"""

import numpy as np
import pickle


def cosine_similarity_rsms(rsm1, rsm2):
    """
    Compute cosine similarity between two RSMs.
    
    Parameters
    ----------
    rsm1 : ndarray, shape (n_conditions, n_conditions)
        First RSM
    rsm2 : ndarray, shape (n_conditions, n_conditions)
        Second RSM
        
    Returns
    -------
    similarity : float
        Cosine similarity between flattened RSMs
    """
    
    # Flatten RSMs
    rsm1_flat = rsm1.flatten()
    rsm2_flat = rsm2.flatten()
    
    # Normalize
    rsm1_norm = rsm1_flat / np.linalg.norm(rsm1_flat)
    rsm2_norm = rsm2_flat / np.linalg.norm(rsm2_flat)
    
    # Cosine similarity
    similarity = np.dot(rsm1_norm, rsm2_norm)
    
    return similarity


def compute_transformation_distance(observed_rsms, target_idx, parcel_fc):
    """
    Compute transformation distance: how different is target from its sources?
    
    d_trans = average cosine distance between observed source RSMs and observed target RSM
    
    Parameters
    ----------
    observed_rsms : ndarray, shape (n_parcels, n_conditions, n_conditions)
        Observed RSMs for all parcels
    target_idx : int
        Index of target parcel
    parcel_fc : ndarray, shape (n_parcels, n_parcels)
        Parcel-level FC (defines which sources connect to target)
        
    Returns
    -------
    d_trans : float
        Transformation distance (averaged across sources)
    source_distances : ndarray
        Individual distances for each source (for diagnostics)
    """
    
    # Get connected source indices
    source_indices = np.where(parcel_fc[target_idx, :] != 0)[0]
    
    # Target RSM
    target_rsm = observed_rsms[target_idx]
    
    # Compute distance to each source
    source_distances = np.zeros(len(source_indices))
    
    for i, source_idx in enumerate(source_indices):
        source_rsm = observed_rsms[source_idx]
        
        # Cosine similarity
        cos_sim = cosine_similarity_rsms(source_rsm, target_rsm)
        
        # Cosine distance = 1 - cosine similarity
        source_distances[i] = 1 - cos_sim
    
    # Average across sources
    d_trans = np.mean(source_distances)
    
    return d_trans, source_distances


def compute_predicted_transformation_distance(observed_rsms, predicted_rsm, 
                                               target_idx, parcel_fc):
    """
    Compute predicted transformation distance: how different is predicted target from sources?
    
    d_trans_hat = average cosine distance between observed source RSMs and predicted target RSM
    
    Parameters
    ----------
    observed_rsms : ndarray, shape (n_parcels, n_conditions, n_conditions)
        Observed RSMs for all parcels
    predicted_rsm : ndarray, shape (n_conditions, n_conditions)
        Predicted RSM for target parcel
    target_idx : int
        Index of target parcel
    parcel_fc : ndarray, shape (n_parcels, n_parcels)
        Parcel-level FC
        
    Returns
    -------
    d_trans_hat : float
        Predicted transformation distance (averaged across sources)
    source_distances : ndarray
        Individual distances for each source
    """
    
    # Get connected source indices
    source_indices = np.where(parcel_fc[target_idx, :] != 0)[0]
    
    # Compute distance from predicted target to each source
    source_distances = np.zeros(len(source_indices))
    
    for i, source_idx in enumerate(source_indices):
        source_rsm = observed_rsms[source_idx]
        
        # Cosine similarity
        cos_sim = cosine_similarity_rsms(source_rsm, predicted_rsm)
        
        # Cosine distance
        source_distances[i] = 1 - cos_sim
    
    # Average across sources
    d_trans_hat = np.mean(source_distances)
    
    return d_trans_hat, source_distances


def compute_prediction_distance(observed_rsm, predicted_rsm):
    """
    Compute prediction distance: how different is prediction from observation?
    
    d_pred = cosine distance between predicted target RSM and observed target RSM
    
    Parameters
    ----------
    observed_rsm : ndarray, shape (n_conditions, n_conditions)
        Observed RSM for target parcel
    predicted_rsm : ndarray, shape (n_conditions, n_conditions)
        Predicted RSM for target parcel
        
    Returns
    -------
    d_pred : float
        Prediction distance
    """
    
    # Cosine similarity
    cos_sim = cosine_similarity_rsms(observed_rsm, predicted_rsm)
    
    # Cosine distance
    d_pred = 1 - cos_sim
    
    return d_pred


def compute_all_distances_single_subject(observed_rsms, predicted_rsms, parcel_fc):
    """
    Compute all three distance metrics for all parcels in one subject.
    
    Parameters
    ----------
    observed_rsms : ndarray, shape (n_parcels, n_conditions, n_conditions)
        Observed RSMs
    predicted_rsms : ndarray, shape (n_parcels, n_conditions, n_conditions)
        Predicted RSMs (from activity flow)
    parcel_fc : ndarray, shape (n_parcels, n_parcels)
        Parcel-level FC
        
    Returns
    -------
    results : dict
        Dictionary with:
        - 'd_trans': Transformation distance for each parcel
        - 'd_trans_hat': Predicted transformation distance for each parcel
        - 'd_pred': Prediction distance for each parcel
    """
    
    n_parcels = observed_rsms.shape[0]
    
    d_trans = np.zeros(n_parcels)
    d_trans_hat = np.zeros(n_parcels)
    d_pred = np.zeros(n_parcels)
    
    for target_idx in range(n_parcels):
        
        # Transformation distance (observed target from observed sources)
        d_trans[target_idx], _ = compute_transformation_distance(
            observed_rsms, target_idx, parcel_fc
        )
        
        # Predicted transformation distance (predicted target from observed sources)
        d_trans_hat[target_idx], _ = compute_predicted_transformation_distance(
            observed_rsms, predicted_rsms[target_idx], target_idx, parcel_fc
        )
        
        # Prediction distance (predicted target from observed target)
        d_pred[target_idx] = compute_prediction_distance(
            observed_rsms[target_idx], predicted_rsms[target_idx]
        )
    
    results = {
        'd_trans': d_trans,
        'd_trans_hat': d_trans_hat,
        'd_pred': d_pred
    }
    
    return results


def compute_transformation_evidence(d_trans_hat, d_pred):
    """
    Compute evidence for meaningful transformation.
    
    If d_trans_hat > d_pred: Evidence that connectivity transforms information
    (predicted target is closer to observed target than to sources)
    
    Parameters
    ----------
    d_trans_hat : ndarray, shape (n_parcels,)
        Predicted transformation distances
    d_pred : ndarray, shape (n_parcels,)
        Prediction distances
        
    Returns
    -------
    transformation_evidence : ndarray, shape (n_parcels,)
        Difference: d_trans_hat - d_pred
        Positive values = evidence for transformation
    """
    
    transformation_evidence = d_trans_hat - d_pred
    
    return transformation_evidence


def process_subject_distances(subject_id, observed_rsms, predicted_rsms, parcel_fc,
                              output_dir=None):
    """
    Complete distance computation pipeline for one subject.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    observed_rsms : ndarray, shape (n_parcels, n_conditions, n_conditions)
        Observed RSMs
    predicted_rsms : ndarray, shape (n_parcels, n_conditions, n_conditions)
        Predicted RSMs
    parcel_fc : ndarray, shape (n_parcels, n_parcels)
        Parcel-level FC
    output_dir : str, optional
        Output directory
        
    Returns
    -------
    results : dict
        All distance metrics and transformation evidence
    """
    
    print(f"\n{'='*70}")
    print(f"Computing Distance Metrics for Subject {subject_id}")
    print(f"{'='*70}")
    
    # Compute all distances
    print("\nComputing transformation and prediction distances...")
    distances = compute_all_distances_single_subject(observed_rsms, predicted_rsms, parcel_fc)
    
    # Compute transformation evidence
    print("Computing transformation evidence...")
    transformation_evidence = compute_transformation_evidence(
        distances['d_trans_hat'], distances['d_pred']
    )
    
    # Summary statistics
    print(f"\nDistance Metrics Summary:")
    print(f"  d_trans (obs target from obs sources):")
    print(f"    Mean: {np.mean(distances['d_trans']):.3f}")
    print(f"    Range: [{np.min(distances['d_trans']):.3f}, {np.max(distances['d_trans']):.3f}]")
    
    print(f"  d_trans_hat (pred target from obs sources):")
    print(f"    Mean: {np.mean(distances['d_trans_hat']):.3f}")
    print(f"    Range: [{np.min(distances['d_trans_hat']):.3f}, {np.max(distances['d_trans_hat']):.3f}]")
    
    print(f"  d_pred (pred target from obs target):")
    print(f"    Mean: {np.mean(distances['d_pred']):.3f}")
    print(f"    Range: [{np.min(distances['d_pred']):.3f}, {np.max(distances['d_pred']):.3f}]")
    
    print(f"\nTransformation Evidence (d_trans_hat - d_pred):")
    print(f"  Mean: {np.mean(transformation_evidence):.3f}")
    print(f"  Parcels with positive evidence: {np.sum(transformation_evidence > 0)}/360")
    
    # Add to results
    results = distances.copy()
    results['transformation_evidence'] = transformation_evidence
    
    # Save
    if output_dir is not None:
        output_file = f'{output_dir}{subject_id}_actflow_distances.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved distance metrics to: {output_file}")
    
    print(f"{'='*70}\n")
    
    return results


def load_distances(distance_file):
    """
    Load distance metrics from file.
    
    Parameters
    ----------
    distance_file : str
        Path to distance pickle file
        
    Returns
    -------
    results : dict
        Distance metrics
    """
    
    with open(distance_file, 'rb') as f:
        results = pickle.load(f)
    
    return results
