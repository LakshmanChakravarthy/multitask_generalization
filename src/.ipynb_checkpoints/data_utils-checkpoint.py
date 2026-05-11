"""
Data loading utilities for resting-state fMRI data.

This module provides functions to load subject data in different spaces
(vertex-level and parcel-level).
"""

import numpy as np
import nibabel as nib
import h5py

projdir = '/home/ln275/f_mc1689_1/multitask_generalization/'
preprocessed_data = projdir + 'data/derivatives/postprocessing/'

def loadrsfMRI(subj,space='parcellated',atlas='glasser',retain_mean=False):
    """
    Load in resting-state residuals
    """
    runs = ['bold9','bold10']
    if space == 'vertex':
        atlas_str = ''
    elif space == 'parcellated':
        if atlas=='glasser':
            atlas_str = ''
        elif atlas=='schaefer':
            atlas_str = '_' + atlas 
        elif atlas=='gordon':
            atlas_str = '_' + atlas 

    data = []
    
    if retain_mean:
        suffix = '_mean_retained'
    else:
        suffix = ''
    
    for run in runs:
        try:
            h5f = h5py.File(preprocessed_data + 'rest_'+space+'_data' + suffix + '/' +  subj + '_b2_rsfMRI' + atlas_str + '_' + space + '_qunex_' + run + suffix + '.h5','r')
            
            ts = h5f['residuals'][:].T
            data.extend(ts)
            h5f.close()
        except:
            print('Subject', subj, '| run', run, ' does not exist... skipping')

    try:
        data = np.asarray(data).T
    except:
        print('\tError')

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
