"""
Data loading utilities for resting-state fMRI data.

This module provides functions to load subject data in different spaces
(vertex-level and parcel-level).
"""

import numpy as np
import nibabel as nib


def load_rsfmri(subject_id, space='vertex', data_dir=None):
    """
    Load resting-state fMRI data for a subject.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., '02', '03', etc.)
    space : str, default='vertex'
        Data space to load. Options: 'vertex', 'parcellated'
    data_dir : str, optional
        Base directory containing the data. If None, uses default project directory.
        
    Returns
    -------
    data : ndarray
        Loaded fMRI data
        - If space='vertex': shape (n_vertices, n_timepoints)
        - If space='parcellated': shape (n_parcels, n_timepoints)
        
    Raises
    ------
    ValueError
        If space is not 'vertex' or 'parcellated'
    """
    
    if space not in ['vertex', 'parcellated']:
        raise ValueError(f"Invalid space '{space}'. Must be 'vertex' or 'parcellated'")
    
    # Set default data directory if not provided
    if data_dir is None:
        data_dir = '/home/ln275/f_mc1689_1/multitask_generalization/data/'
    
    # Construct file path based on space
    if space == 'vertex':
        # Load vertex-level data (surface space)
        filepath = f"{data_dir}/sub-{subject_id}/func/sub-{subject_id}_task-rest_space-fsLR32k_bold.dtseries.nii"
        data = nib.load(filepath).get_fdata().T  # Transpose to (vertices, time)
        
    elif space == 'parcellated':
        # Load parcellated data (Glasser parcels)
        filepath = f"{data_dir}/sub-{subject_id}/func/sub-{subject_id}_task-rest_space-Glasser360_bold.ptseries.nii"
        data = nib.load(filepath).get_fdata().T  # Transpose to (parcels, time)
    
    return data


def load_glasser_parcellation(glasser_file=None):
    """
    Load Glasser parcellation labels.
    
    Parameters
    ----------
    glasser_file : str, optional
        Path to Glasser parcellation file. If None, uses default location.
        
    Returns
    -------
    glasser : ndarray, shape (n_vertices,)
        Glasser parcellation labels (1-360, 0 for non-cortical)
    """
    
    if glasser_file is None:
        glasser_file = ('/home/ln275/f_mc1689_1/multitask_generalization/docs/files/'
                       'Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii')
    
    glasser = np.squeeze(nib.load(glasser_file).get_fdata())
    
    return glasser


def load_subject_parcel_fc(subject_id, fc_dir=None):
    """
    Load parcel-level FC for a single subject from the saved pickle file.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    fc_dir : str, optional
        Directory containing FC files. If None, uses default location.
        
    Returns
    -------
    parcel_fc : ndarray, shape (n_parcels, n_parcels)
        Parcel-level functional connectivity matrix
    """
    
    import pickle
    
    if fc_dir is None:
        fc_dir = '/home/ln275/f_mc1689_1/multitask_generalization/data/derivatives/FC_new/'
    
    # Assuming individual subject FC is stored separately
    # Adjust this based on your actual file structure
    with open(f'{fc_dir}/sub{subject_id}_parcel_fc.pkl', 'rb') as f:
        parcel_fc = pickle.load(f)
    
    return parcel_fc


def load_all_subjects_parcel_fc(fc_dir=None):
    """
    Load parcel-level FC for all subjects from the saved pickle file.
    
    Parameters
    ----------
    fc_dir : str, optional
        Directory containing FC files. If None, uses default location.
        
    Returns
    -------
    all_sub_parcel_fc : ndarray, shape (n_subjects, n_parcels, n_parcels)
        Parcel-level functional connectivity matrices for all subjects
    """
    
    import pickle
    
    if fc_dir is None:
        fc_dir = '/home/ln275/f_mc1689_1/multitask_generalization/data/derivatives/FC_new/'
    
    with open(f'{fc_dir}/allSubFC_parCorr.pkl', 'rb') as f:
        all_sub_parcel_fc = pickle.load(f)
    
    return all_sub_parcel_fc
