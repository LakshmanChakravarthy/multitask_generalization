"""
Task GLM estimation using ridge regression with cross-validation.

Estimates beta weights for each task condition using ridge regression on fMRI data,
with nuisance regression for motion and physiological confounds.
"""

import numpy as np
import pickle
import h5py
import pandas as pd
from scipy import signal
from nipy.modalities.fmri.hemodynamic_models import spm_hrf
import nibabel as nib


def compute_task_betas_single_run(subject_id, session_id, run_id, space='parcellated',
                                   zscore=True, data_dir=None, output_dir=None, 
                                   task_info_file=None):
    """
    Compute task condition betas for a single run using ridge regression.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., '02', '03')
    session_id : str
        Session identifier (e.g., 'a1', 'a2', 'b1', 'b2')
    run_id : int
        Run number (1-8)
    space : str, default='parcellated'
        Data space: 'parcellated' or 'vertex'
    zscore : bool, default=True
        Whether to z-score data and regressors before regression
    data_dir : str, optional
        Base QuNex data directory
    output_dir : str, optional
        Output directory for results
    task_info_file : str, optional
        Path to task condition info pickle file
        
    Returns
    -------
    None (saves results to HDF5 file)
    """
    
    # Set default paths
    if data_dir is None:
        data_dir = '/home/ln275/f_mc1689_1/MDTB/qunex_mdtb/'
    if output_dir is None:
        output_dir = '/home/ln275/f_mc1689_1/MDTB/derivatives/postprocessing/betasByTaskCondition/'
    if task_info_file is None:
        task_info_file = '/home/ln275/f_mc1689_1/MDTB/data/derivatives/allSubTaskConditionInfo.pkl'
    
    # Create session string
    sess = f'{subject_id}_{session_id}'
    
    # Load task timing information
    with open(task_info_file, 'rb') as f:
        allSubInfoSlices = pickle.load(f)
    
    # Get subject/session/run indices
    subIDs = ['02','03','04','06','08','09','10','12','14','15','18','20','22','25','27','29','31','17','19','21','24','26','28','30']
    sessionIDs = ['a1','a2','b1','b2']
    
    subIdx = subIDs.index(subject_id)
    sessIdx = sessionIDs.index(session_id)
    runIdx = run_id - 1  # Convert to 0-indexed
    
    print(f'Processing: subject {subject_id}, session {session_id}, run {run_id}')
    
    # Load fMRI data
    run_name = f'bold{run_id}'
    if space == 'parcellated':
        rundata = load_parcellated_data(sess, run_name, data_dir)
    elif space == 'vertex':
        rundata = load_vertex_data(sess, run_name, data_dir)
    else:
        raise ValueError(f"Invalid space '{space}'. Must be 'parcellated' or 'vertex'")
    
    num_timepoints = rundata.shape[0]
    
    # Skip initial frames
    frames_to_skip = 5
    tMask = np.ones((num_timepoints,), dtype=bool)
    tMask[:frames_to_skip] = False
    rundata = rundata[tMask, :]
    
    # Detrend
    rundata = signal.detrend(rundata, axis=0, type='linear')
    
    # Load nuisance regressors
    nuisregs = load_nuisance_regressors(sess, run_name, num_timepoints, data_dir)
    nuisregs = nuisregs[tMask, :]
    
    # Load task timing and create task regressors
    tasktiming = load_task_timing_betas(allSubInfoSlices, subIdx, sessIdx, runIdx, num_timepoints)
    task_regs = tasktiming['taskRegressors'][tMask, :]
    regression_index = np.array(tasktiming['stimIndex'])
    
    # Combine task and nuisance regressors
    allRegressors = np.hstack((task_regs, nuisregs))
    print(f'Design matrix shape: {allRegressors.shape}')
    
    # Z-score data and regressors if requested
    if zscore:
        from scipy import stats
        rundata = stats.zscore(rundata, axis=0)
        allRegressors = stats.zscore(allRegressors, axis=0)
        print('Data and regressors z-scored')
    
    # Ridge regression with cross-validation to find optimal alpha
    print('Running ridge regression with CV for alpha selection...')
    betas, best_alpha = ridge_regression_cv(allRegressors, rundata, n_folds=4)
    
    # Extract only task-related betas (exclude nuisance regressors)
    betas = betas[:len(regression_index), :]
    print(f'Task betas shape: {betas.shape}')
    print(f'Best alpha: {best_alpha}')
    
    # Compute residuals (optional, for quality control)
    y_pred = np.dot(allRegressors, np.vstack([betas, np.zeros((nuisregs.shape[1], betas.shape[1]))]))
    residual_ts = rundata - y_pred
    
    # Save results
    output_filename = f'{output_dir}{sess}_tfMRI_{space}_betaseries_bold{run_id}'
    
    # Save task condition index
    np.savetxt(f'{output_filename}_taskIndex.csv', regression_index, delimiter=',', fmt='%s')
    
    # Save betas and residuals
    with h5py.File(f'{output_filename}.h5', 'a') as h5f:
        for name in ['betas', 'residuals']:
            if name in h5f:
                del h5f[name]
        h5f.create_dataset('betas', data=betas)
        h5f.create_dataset('residuals', data=residual_ts)
    
    print(f'Results saved to: {output_filename}.h5')


def load_task_timing_betas(allSubInfoSlices, subIdx, sessIdx, runIdx, num_timepoints):
    """
    Load task timing information and create HRF-convolved task regressors.
    
    Parameters
    ----------
    allSubInfoSlices : list
        Nested list containing task condition info for all subjects/sessions/runs
    subIdx : int
        Subject index
    sessIdx : int
        Session index (0=a1, 1=a2, 2=b1, 3=b2)
    runIdx : int
        Run index (0-7)
    num_timepoints : int
        Number of timepoints in the run
        
    Returns
    -------
    output : dict
        Dictionary with keys:
        - 'taskRegressors': HRF-convolved design matrix (time x conditions)
        - 'taskDesignMat': Binary design matrix before convolution
        - 'stimIndex': List of task condition names
    """
    
    tr_length = 1.0
    resample_rate = 10  # Upsample for better HRF convolution
    
    # Get task info for this subject/session/run
    stimdf = allSubInfoSlices[subIdx][sessIdx][runIdx]
    
    # Different instruction screen naming between session types
    if sessIdx in [0, 1]:  # Sessions a1, a2
        index_loc = np.where((stimdf.taskCondname != "Ins_rest") & (stimdf.taskCondname != 0))[0]
        conditions = [c for c in np.unique(stimdf.taskCondname.values) 
                     if c not in ['Ins_rest', 'rest']]
    else:  # Sessions b1, b2
        index_loc = np.where((stimdf.taskCondname != "Ins_rest2") & (stimdf.taskCondname != 0))[0]
        conditions = [c for c in np.unique(stimdf.taskCondname.values) 
                     if c not in ['Ins_rest2', 'rest2']]
    
    n_betas = len(conditions)
    
    # Create upsampled design matrix
    stim_mat = np.zeros((num_timepoints * resample_rate, n_betas))
    stim_index = []
    
    for col_count, cond in enumerate(conditions):
        cond_rows = np.where(stimdf.taskCondname == cond)[0]
        for i in cond_rows:
            trstart = int(np.floor(stimdf.onset[i] * resample_rate))
            duration = stimdf.trialDur[i] * resample_rate
            trend = int(np.ceil(trstart + duration))
            stim_mat[trstart:trend, col_count] = 1.0
        
        stim_index.append(cond)
    
    # Convolve with HRF
    taskStims_HRF = np.zeros((num_timepoints, n_betas))
    spm_hrfTS = spm_hrf(tr_length, oversampling=resample_rate)
    
    for stim in range(stim_mat.shape[1]):
        # Convolve
        tmpconvolve = np.convolve(stim_mat[:, stim], spm_hrfTS)
        # Downsample
        tmpconvolveDS = signal.decimate(tmpconvolve, resample_rate)
        # Truncate to original size
        taskStims_HRF[:, stim] = tmpconvolveDS[:num_timepoints]
    
    output = {
        'taskRegressors': taskStims_HRF,
        'taskDesignMat': stim_mat,
        'stimIndex': stim_index
    }
    
    return output


def load_nuisance_regressors(sess, run, num_timepoints, data_dir):
    """
    Load nuisance regressors for motion and physiological confounds (QuNex model).
    
    Includes:
    - 6 motion parameters (translation + rotation) + derivatives
    - Quadratic terms for motion parameters and derivatives (24 motion regressors total)
    - White matter signal + derivative
    - Ventricle signal + derivative
    - Quadratic terms for physiological signals (8 physiological regressors total)
    
    Total: 32 regressors (NO global signal regression)
    
    Parameters
    ----------
    sess : str
        Session string (e.g., '02_a1')
    run : str
        Run identifier (e.g., 'bold1')
    num_timepoints : int
        Number of timepoints (for validation)
    data_dir : str
        Base QuNex data directory
        
    Returns
    -------
    nuisanceRegressors : ndarray, shape (num_timepoints, 32)
        Nuisance regressor matrix
    """
    
    nuisdir = f'{data_dir}sessions/{sess}/hcp/{sess}/MNINonLinear/Results/{run}/'
    
    # Load physiological signals
    data = pd.read_csv(f'{nuisdir}{run}_Atlas_hp2000_clean_vn.txt', sep='\t')
    
    # Ventricle signal + derivative
    ventricles_signal = data.CSF.values[:-2]
    ventricles_signal_deriv = np.zeros(ventricles_signal.shape)
    ventricles_signal_deriv[1:] = ventricles_signal[1:] - ventricles_signal[:-1]
    ventricles_signal_deriv[0] = np.mean(ventricles_signal_deriv[1:])
    
    # White matter signal + derivative
    wm_signal = data.WM.values[:-2]
    wm_signal_deriv = np.zeros(wm_signal.shape)
    wm_signal_deriv[1:] = wm_signal[1:] - wm_signal[:-1]
    wm_signal_deriv[0] = np.mean(wm_signal_deriv[1:])
    
    # Load motion parameters
    motiondat = pd.read_csv(f'{nuisdir}{run}_mov.dat', sep='\s+')
    motionparams = np.zeros((len(motiondat), 6))
    motionparams[:, 0] = motiondat['dx(mm)'].values
    motionparams[:, 1] = motiondat['dy(mm)'].values
    motionparams[:, 2] = motiondat['dz(mm)'].values
    motionparams[:, 3] = motiondat['X(deg)'].values
    motionparams[:, 4] = motiondat['Y(deg)'].values
    motionparams[:, 5] = motiondat['Z(deg)'].values
    
    # Motion parameter derivatives
    motionparams_deriv = np.zeros((len(motiondat), 6))
    motionparams_deriv[1:, :] = motionparams[1:, :] - motionparams[:-1, :]
    motionparams_deriv[0, :] = np.mean(motionparams[1:, :], axis=0)
    
    # Quadratic terms
    motionparams_quadratics = motionparams ** 2
    motionparams_deriv_quadratics = motionparams_deriv ** 2
    
    # Physiological parameters (excluding global signal)
    physiological_params = np.vstack((wm_signal, wm_signal_deriv, 
                                     ventricles_signal, ventricles_signal_deriv)).T
    physiological_params_quadratics = physiological_params ** 2
    
    # Combine all nuisance regressors
    nuisanceRegressors = np.hstack((motionparams, motionparams_quadratics,
                                    motionparams_deriv, motionparams_deriv_quadratics,
                                    physiological_params, physiological_params_quadratics))
    
    return nuisanceRegressors


def load_parcellated_data(sess, run, data_dir, atlas='glasser'):
    """
    Load parcellated fMRI data.
    
    Parameters
    ----------
    sess : str
        Session string (e.g., '02_a1')
    run : str
        Run identifier (e.g., 'bold1')
    data_dir : str
        Base QuNex data directory
    atlas : str, default='glasser'
        Atlas name
        
    Returns
    -------
    data : ndarray, shape (timepoints, parcels)
        Parcellated fMRI data
    """
    
    datafile = f'{data_dir}sessions/{sess}/images/functional/{run}_Atlas.LR.Parcels.32k_fs_LR.ptseries.nii'
    data = nib.load(datafile).get_fdata()
    return data


def load_vertex_data(sess, run, data_dir):
    """
    Load vertex-level fMRI data.
    
    Parameters
    ----------
    sess : str
        Session string (e.g., '02_a1')
    run : str
        Run identifier (e.g., 'bold1')
    data_dir : str
        Base QuNex data directory
        
    Returns
    -------
    data : ndarray, shape (timepoints, vertices)
        Vertex-level fMRI data
    """
    
    datafile = f'{data_dir}sessions/{sess}/images/functional/{run}_Atlas.dtseries.nii'
    data = nib.load(datafile).get_fdata()
    return data


def ridge_regression_cv(X, y, n_folds=4, alphas=None):
    """
    Ridge regression with k-fold cross-validation for alpha selection.
    
    Uses the Huth lab's optimized ridge implementation.
    
    Parameters
    ----------
    X : ndarray, shape (timepoints, n_regressors)
        Design matrix (task + nuisance regressors)
    y : ndarray, shape (timepoints, n_voxels/vertices)
        fMRI data
    n_folds : int, default=4
        Number of CV folds
    alphas : array_like, optional
        Ridge parameters to test. If None, uses log-spaced values.
        
    Returns
    -------
    betas : ndarray, shape (n_regressors, n_voxels/vertices)
        Ridge regression weights using optimal alpha
    best_alpha : float or ndarray
        Optimal alpha value(s)
    """
    
    # Import ridge functions
    import sys
    sys.path.insert(0, '/home/ln275/f_mc1689_1/MDTB/huth_ridge')
    from huth_ridge import ridge, ridge_corr
    
    if alphas is None:
        alphas = np.logspace(0, 3, 20)
    
    # Split data into folds
    split_size = int(X.shape[0] / n_folds)
    allFoldRcorrs = []
    
    for i in range(n_folds):
        test_idx = range(i * split_size, (i + 1) * split_size)
        train_idx = np.delete(range(X.shape[0]), test_idx)
        
        Rstim = X[train_idx, :]
        Pstim = X[test_idx, :]
        Rresp = y[train_idx, :]
        Presp = y[test_idx, :]
        
        Rcorrs = ridge_corr(Rstim, Pstim, Rresp, Presp, alphas, 
                           normalpha=False, corrmin=0.2, singcutoff=1e-10, use_corr=True)
        
        allFoldRcorrs.append(Rcorrs)
    
    # Find best alpha
    allFoldRcorrs = np.array(allFoldRcorrs)
    avgFoldRcorr = np.mean(allFoldRcorrs, axis=0)
    best_alpha = alphas[np.argmax(avgFoldRcorr, axis=0)]
    
    # Compute final weights using optimal alpha
    betas = ridge(X, y, alpha=best_alpha)
    
    return betas, best_alpha


# Main processing function for batch processing
def process_all_subjects(subject_ids, session_ids, n_runs=8, space='parcellated', 
                        zscore=True, **kwargs):
    """
    Process all subjects/sessions/runs.
    
    Parameters
    ----------
    subject_ids : list of str
        Subject identifiers
    session_ids : list of str
        Session identifiers (e.g., ['a1', 'a2', 'b1', 'b2'])
    n_runs : int, default=8
        Number of runs per session
    space : str, default='parcellated'
        Data space
    zscore : bool, default=True
        Whether to z-score data and regressors
    **kwargs : dict
        Additional arguments passed to compute_task_betas_single_run
    """
    
    for subject_id in subject_ids:
        for session_id in session_ids:
            for run_id in range(1, n_runs + 1):
                try:
                    compute_task_betas_single_run(subject_id, session_id, run_id, 
                                                 space=space, zscore=zscore, **kwargs)
                except FileNotFoundError:
                    print(f'Files not found for {subject_id}_{session_id} run {run_id}, skipping...')
                    continue
