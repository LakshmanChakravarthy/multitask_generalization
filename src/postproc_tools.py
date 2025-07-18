# Takuya Ito

# Post-processing nuisance regression using Ciric et al. 2017 inspired best-practices
# Takes output from qunex (up to hcp5 + extracting nuisance signals)

import numpy as np
import os
import glob
from nipy.modalities.fmri.hemodynamic_models import spm_hrf
import multiprocessing as mp
import h5py
import scipy.stats as stats
from scipy import signal
import nibabel as nib
import scipy
import pandas as pd
import time
import warnings
warnings.simplefilter('ignore', np.ComplexWarning)
from sklearn.linear_model import LinearRegression
import glob
import pickle


## Define GLOBAL variables (variables accessible to all functions
# Define base data directory
#datadir = '/gpfs/loomis/project/n3/Studies/MurrayLab/taku/mdtb_data/qunex_mdtb/'
projdir = '/home/ln275/f_mc1689_1/MDTB/'
datadir = '/home/ln275/f_mc1689_1/MDTB/qunex_mdtb/'
# Define number of frames to skip
framesToSkip = 5
# Define the *output* directory for nuisance regressors
nuis_reg_dir = datadir + 'nuisanceRegressors/'
# Define the *output* directory for preprocessed data
#

# # Read in task condition info
# stroopENNdir = projdir + 'derivatives/stroop_ENN/'
# with open(stroopENNdir + 'allSubTaskConditionInfo.pkl', 'rb') as f:
#     allSubInfoSlices = pickle.load(f)
#########################################

def loadTaskTimingCanonical(sess, run, num_timepoints, nRegsFIR=20):
    """
    Loads task timings for each run separately
    Does this for each trial type separately (45 conditions)
    """
    trLength = 1.0
    subj = sess[:2] # first 2 characters form the subject ID
    sess_id = sess[-2:] # last 2 characters form the session
    tasktime_dir = datadir + 'sessions/' + sess + '/bids/func/'
    stimfile = glob.glob(tasktime_dir + 'sub-' + subj + '_ses-' + sess_id + '*' + str(run) + '_events.tsv')[0]
    stimdf = pd.read_csv(stimfile,sep='\t') 
    conditions = np.unique(stimdf.trial_type.values)
    conditions = list(conditions)
    conditions.remove('Instruct') # Remove this - not a condition (and no timing information)
    # Note that the event files don't have a distinction between 0-back nd 2-back conditions for both object recognition and verbal recognition tasks
    # conditions.remove('Rest')
    tasks = np.unique(stimdf.taskName.values)

    stim_mat = np.zeros((num_timepoints,len(conditions)))
    stim_index = []

    stimcount = 0
    for cond in conditions:
        conddf = stimdf.loc[stimdf.trial_type==cond]
        for ind in conddf.index:
            trstart = int(conddf.startTRreal[ind])
            duration = conddf.duration[ind]
            trend = int(trstart + duration)
            stim_mat[trstart:trend,stimcount] = 1.0

        stim_index.append(cond)
        stimcount += 1 # go to next condition


    ## 
    if taskModel=='FIR':
        ## TODO
        # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)

        ## First set up FIR design matrix
        stim_index = []
        taskStims_FIR = [] 
        for stim in range(stim_mat.shape[1]):
            taskStims_FIR.append([])
            time_ind = np.where(stim_mat[:,stim]==1)[0]
            blocks = _group_consecutives(time_ind) # Get blocks (i.e., sets of consecutive TRs)
            # Identify the longest block - set FIR duration to longest block
            maxRegsForBlocks = 0
            for block in blocks:
                if len(block) > maxRegsForBlocks: maxRegsForBlocks = len(block)
            taskStims_FIR[stim] = np.zeros((stim_mat.shape[0],maxRegsForBlocks+nRegsFIR)) # Task timing for this condition is TR x length of block + FIR lag
            stim_index.extend(np.repeat(stim,maxRegsForBlocks+nRegsFIR))
        stim_index = np.asarray(stim_index)

        ## Now fill in FIR design matrix
        # Make sure to cut-off FIR models for each run separately
        trcount = 0

        for run in range(nRunsPerTask):
            trstart = trcount
            trend = trstart + nTRsPerRun
                
            for stim in range(stim_mat.shape[1]):
                time_ind = np.where(stim_mat[:,stim]==1)[0]
                blocks = _group_consecutives(time_ind) # Get blocks (i.e., sets of consecutive TRs)
                for block in blocks:
                    reg = 0
                    for tr in block:
                        # Set impulses for this run/task only
                        if trstart < tr < trend:
                            taskStims_FIR[stim][tr,reg] = 1
                            reg += 1

                        if not trstart < tr < trend: continue # If TR not in this run, skip this block

                    # If TR is not in this run, skip this block
                    if not trstart < tr < trend: continue

                    # Set lag due to HRF
                    for lag in range(1,nRegsFIR+1):
                        # Set impulses for this run/task only
                        if trstart < tr+lag < trend:
                            taskStims_FIR[stim][tr+lag,reg] = 1
                            reg += 1
            trcount += nTRsPerRun
        

        taskStims_FIR2 = np.zeros((stim_mat.shape[0],1))
        task_index = []
        for stim in range(stim_mat.shape[1]):
            task_index.extend(np.repeat(stim,taskStims_FIR[stim].shape[1]))
            taskStims_FIR2 = np.hstack((taskStims_FIR2,taskStims_FIR[stim]))

        taskStims_FIR2 = np.delete(taskStims_FIR2,0,axis=1)

        #taskRegressors = np.asarray(taskStims_FIR)
        taskRegressors = taskStims_FIR2
    
        # To prevent SVD does not converge error, make sure there are no columns with 0s
        zero_cols = np.where(np.sum(taskRegressors,axis=0)==0)[0]
        taskRegressors = np.delete(taskRegressors, zero_cols, axis=1)
        stim_index = np.delete(stim_index, zero_cols)

    elif taskModel=='canonical':
        ## 
        # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
        taskStims_HRF = np.zeros(stim_mat.shape)
        spm_hrfTS = spm_hrf(trLength,oversampling=1)
       

        for stim in range(stim_mat.shape[1]):

            # Perform convolution
            tmpconvolve = np.convolve(stim_mat[:,stim],spm_hrfTS)
            taskStims_HRF[:,stim] = tmpconvolve[:num_timepoints]


        taskRegressors = taskStims_HRF.copy()
    

    # Create temporal mask (skipping which frames?)
    output = {}
    # Commented out since we demean each run prior to loading data anyway
    output['taskRegressors'] = taskRegressors
    output['taskDesignMat'] = stim_mat
    output['stimIndex'] = stim_index

    return output

def loadTaskTimingBetaSeries(sess, run, num_timepoints):
    """
    Loads task timings for each run separately
    """
    trLength = 1.0
    subj = sess[:2] # first 2 characters form the subject ID
    sess_id = sess[-2:] # last 2 characters form the session
    tasktime_dir = datadir + 'sessions/' + sess + '/bids/func/'
    stimfile = glob.glob(tasktime_dir + 'sub-' + subj + '_ses-' + sess_id + '*' + str(run) + '_events.tsv')[0]
    stimdf = pd.read_csv(stimfile,sep='\t') 

    # number of betas correspond to all task conditions "excluding" the instruction screen condition for each task
    n_betas = np.sum(stimdf.trial_type!="Instruct")
    index_loc = np.where(stimdf.trial_type!="Instruct")[0]
    
    stim_mat = np.zeros((num_timepoints,n_betas))
    stim_index = []

    col_count = 0 
    for i in index_loc:
        trstart = int(np.floor(stimdf.startTRreal[i]))
        duration = stimdf.duration[i]
        trend = int(np.ceil(trstart + duration))
        stim_mat[trstart:trend,col_count] = 1.0

        stim_index.append(stimdf.trial_type[i])
        col_count += 1 # go to next matrix_col

    conditions = np.unique(stimdf.trial_type.values)
    conditions = list(conditions)
    conditions.remove('Instruct') # Remove this - not a condition (and no timing information)

    ## 
    # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
    taskStims_HRF = np.zeros(stim_mat.shape)
    spm_hrfTS = spm_hrf(trLength,oversampling=1)
   

    for stim in range(stim_mat.shape[1]):

        # Perform convolution
        tmpconvolve = np.convolve(stim_mat[:,stim],spm_hrfTS)
        taskStims_HRF[:,stim] = tmpconvolve[:num_timepoints]


    taskRegressors = taskStims_HRF.copy()
    

    # Create temporal mask (skipping which frames?)
    output = {}
    # Commented out since we demean each run prior to loading data anyway
    output['taskRegressors'] = taskRegressors
    output['taskDesignMat'] = stim_mat
    output['stimIndex'] = stim_index

    return output

def loadBetasByTaskCondition(subIdx, sessIdx, runIdx, num_timepoints):

    trLength = 1.0
    resample_rate = 10
    
    stimdf = allSubInfoSlices[subIdx][sessIdx][runIdx]
    
    if sessIdx == 0 or sessIdx == 1:
        index_loc = np.where((stimdf.taskCondname!="Ins_rest") & (stimdf.taskCondname!=0))[0]
        
        conditions = np.unique(stimdf.taskCondname.values)
        conditions = list(conditions)
        conditions.remove('Ins_rest')
        conditions.remove('rest')
        
    elif sessIdx == 2 or sessIdx == 3:
        index_loc = np.where((stimdf.taskCondname!="Ins_rest2") & (stimdf.taskCondname!=0))[0]
        
        conditions = np.unique(stimdf.taskCondname.values)
        conditions = list(conditions)
        conditions.remove('Ins_rest2')
        conditions.remove('rest2')
    
    n_betas = len(conditions)
    
    stim_mat = np.zeros((num_timepoints*resample_rate,n_betas))
    stim_index = []

    col_count = 0 
    for cond in conditions:
        cond_rows = np.where(stimdf.taskCondname==cond)[0]
        for i in cond_rows:
            trstart = int(np.floor(stimdf.onset[i]*resample_rate))
            duration = stimdf.trialDur[i]*resample_rate
            trend = int(np.ceil(trstart + duration))
            stim_mat[trstart:trend,col_count] = 1.0

        stim_index.append(cond)
        col_count += 1 # go to next matrix_col
        
    ## 
    # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
    # Upsampling SPM HRF with resample rate
    taskStims_HRF = np.zeros((num_timepoints,n_betas))
    spm_hrfTS = spm_hrf(trLength,oversampling=resample_rate)


    for stim in range(stim_mat.shape[1]):

        # Perform convolution
        tmpconvolve = np.convolve(stim_mat[:,stim],spm_hrfTS)
        # Downsample
        tmpconvolveDS = signal.decimate(tmpconvolve,resample_rate)
        # Truncate to original size
        taskStims_HRF[:,stim] = tmpconvolveDS[:num_timepoints]


    taskRegressors = taskStims_HRF.copy()

    # Create temporal mask (skipping which frames?)
    output = {}
    # Commented out since we demean each run prior to loading data anyway
    output['taskRegressors'] = taskRegressors
    output['taskDesignMat'] = stim_mat
    output['stimIndex'] = stim_index

    return output
    

def loadTaskTimingBetaSeriesInsIncluded(sess, run, num_timepoints):
    """
    Loads task timings for each run separately
    """
    trLength = 1.0
    resample_rate = 10

    subj = sess[:2] # first 2 characters form the subject ID
    sess_id = sess[-2:] # last 2 characters form the session
    tasktime_dir = datadir + 'sessions/' + sess + '/bids/func/'
    stimfile = glob.glob(tasktime_dir + 'sub-' + subj + '_ses-' + sess_id + '*' + str(run) +                            '_events.tsv')[0]
    stimdf = pd.read_csv(stimfile,sep='\t') 
    
    
    
    # Differentiating each instruction screen by the following task name
    if sess_id[0] == 'a':
        idx = stimdf[stimdf['taskName']=='instruct'].index
        # number of betas correspond to all task conditions "including" the instruction screen                   condition for each task
        # using taskName instead of trial_type as identifier, as there is only one instruction screen             for multiple task conditions
        # Using rest condition as implicit baseline, hence not modelling it (see Methods in King et al.           2019)
        index_loc = np.where((stimdf.taskName!="Ins_rest") & (stimdf.taskName!="rest"))[0]
        stimdf.taskName[idx] = 'Ins_'+stimdf.taskName[np.array(idx)+1].copy()
        
        conditions = np.unique(stimdf.taskName.values)
        conditions = list(conditions)
        conditions.remove('Ins_rest')
        conditions.remove('rest')
    
    elif sess_id[0] == 'b':
        idx = stimdf[stimdf['taskName']=='instruct2'].index
        index_loc = np.where((stimdf.taskName!="Ins_rest2") & (stimdf.taskName!="rest2"))[0]
        stimdf.taskName[idx] = 'Ins_'+stimdf.taskName[np.array(idx)+1].copy()
    
        conditions = np.unique(stimdf.taskName.values)
        conditions = list(conditions)
        conditions.remove('Ins_rest2')
        conditions.remove('rest2')

    n_betas = len(index_loc)
    
    stim_mat = np.zeros((num_timepoints*resample_rate,n_betas))
    stim_index = []

    col_count = 0 
    for i in index_loc:
        trstart = int(np.floor(stimdf.onset[i]*resample_rate))
        duration = stimdf.trialDur[i]*resample_rate
        trend = int(np.ceil(trstart + duration))
        stim_mat[trstart:trend,col_count] = 1.0

        stim_index.append(stimdf.taskName[i])
        col_count += 1 # go to next matrix_col

    ## 
    # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
    # Upsampling SPM HRF with resample rate
    taskStims_HRF = np.zeros((num_timepoints,n_betas))
    spm_hrfTS = spm_hrf(trLength,oversampling=resample_rate)


    for stim in range(stim_mat.shape[1]):

        # Perform convolution
        tmpconvolve = np.convolve(stim_mat[:,stim],spm_hrfTS)
        # Downsample
        tmpconvolveDS = signal.decimate(tmpconvolve,resample_rate)
        # Truncate to original size
        taskStims_HRF[:,stim] = tmpconvolveDS[:num_timepoints]


    taskRegressors = taskStims_HRF.copy()

    # Create temporal mask (skipping which frames?)
    output = {}
    # Commented out since we demean each run prior to loading data anyway
    output['taskRegressors'] = taskRegressors
    output['taskDesignMat'] = stim_mat
    output['stimIndex'] = stim_index

    return output

def loadTaskTimingTaskSustainedInsIncluded(sess, run, num_timepoints):
    """
    Loads task timings for each run separately
    with sustained regressor for each task. Instructions included
    as a single regressor (of no interest)
    """
    
    trLength = 1.0
    resample_rate = 10
    subj = sess[:2] # first 2 characters form the subject ID
    sess_id = sess[-2:] # last 2 characters form the session
    tasktime_dir = datadir + 'sessions/' + sess + '/bids/func/'
    stimfile = glob.glob(tasktime_dir + 'sub-' + subj + '_ses-' + sess_id + '*' + str(run) + '_events.tsv')[0]
    stimdf = pd.read_csv(stimfile,sep='\t') 

    # Splitting verbGeneration and verbGeneration2 (in b1 and b2) task into conditions to get estimate for WordRead condition

    stimdf['newTaskName'] = stimdf['taskName']
    for i in stimdf.index:
        if stimdf['taskName'][i] == 'verbGeneration' or stimdf['taskName'][i] == 'verbGeneration2' or stimdf['taskName'][i] == 'arithmetic' or stimdf['taskName'][i] == 'emotionProcess':
            stimdf['newTaskName'][i] = stimdf['trial_type'][i]

    # Using rest as implicit baseline
    tasks = np.unique(stimdf.newTaskName.values)
    tasks = list(tasks)
    if sess_id[0] == 'a':
        tasks.remove('rest')
    elif sess_id[0] == 'b':
        tasks.remove('rest2')

    stim_mat = np.zeros((num_timepoints*resample_rate,len(tasks)))
    stim_index = []

    stimcount = 0
    for task in tasks:
        if task == 'instruct' or task == 'instruct2':
            taskdf = stimdf.loc[stimdf.newTaskName==task]
            for ind in taskdf.index:
                start = int(np.floor(stimdf.onset[ind]*resample_rate))
                duration = taskdf.duration[ind]*resample_rate
                trial_end = int(np.ceil(start+duration))
                stim_mat[start:trial_end,stimcount] = 1.0
        else:
            taskdf = stimdf.loc[stimdf.newTaskName==task]
            start = int(np.floor(stimdf.onset[taskdf.index[0]]*resample_rate))
            last_trial_duration = taskdf.duration[taskdf.index[-1]]*resample_rate
            block_end = int(np.ceil(stimdf.onset[taskdf.index[-1]]*resample_rate +                                              last_trial_duration))
            stim_mat[start:block_end,stimcount] = 1.0

        stim_index.append(task)
        stimcount += 1 # go to next condition

    # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
    taskStims_HRF = np.zeros((num_timepoints,len(tasks)))
    spm_hrfTS = spm_hrf(trLength,oversampling=resample_rate)


    for stim in range(stim_mat.shape[1]):

        # Perform convolution
        tmpconvolve = np.convolve(stim_mat[:,stim],spm_hrfTS)
        # Downsample
        tmpconvolveDS = signal.decimate(tmpconvolve,resample_rate)
        # Truncate to original size
        taskStims_HRF[:,stim] = tmpconvolveDS[:num_timepoints]


    taskRegressors = taskStims_HRF.copy()


    # Create temporal mask (skipping which frames?)
    output = {}
    # Commented out since we demean each run prior to loading data anyway
    output['taskRegressors'] = taskRegressors
    output['taskDesignMat'] = stim_mat
    output['stimIndex'] = stim_index

    return output

def loadTaskTimingFIR(sess, num_timepoints, nRegsFIR=20):
    """
    Loads task timings for all runs for a given subject session 

    Parameters:

    subj 
        subject ID (as a string, e.g., '02')
    num_timepoints
        an array containing number of time points in each run
    """

    blocklength = 30 # length of a task block
    trLength = 1.0
    stimdf = pd.DataFrame()
    subj = sess[:2] # first 2 characters form the subject ID
    sess_id = sess[-2:] # last 2 characters form the session
    for run in range(1,9):
        tasktime_dir = datadir + 'sessions/' + sess + '/bids/func/'
        stimfile = glob.glob(tasktime_dir + 'sub-' + subj + '_ses-' + sess_id + '*' + str(run) + '_events.tsv')[0]
        stimdf = stimdf.append(pd.read_csv(stimfile,sep='\t'))

    #### Now find all task block onsets by identifying 
    df_taskonsets = {}
    df_taskonsets['task'] = []
    df_taskonsets['onset'] = []
    df_taskonsets['run'] = []
    # Identify all instruction screens (i.e., the beginning of task blocks)
    stimdf = stimdf.reset_index() # Get new index
    instruction_ind = stimdf.loc[stimdf.trial_type=='Instruct'].index
    # Now add 1 to instruction index, since that's when the task starts
    task_ind = instruction_ind + 1
    for ind in task_ind:
        df_taskonsets['task'].append(stimdf.loc[ind].taskName)
        df_taskonsets['onset'].append(stimdf.loc[ind].startTRreal)
        df_taskonsets['run'].append(stimdf.loc[ind].boldRun)
    df_taskonsets = pd.DataFrame(df_taskonsets)

    #### Now for each task iterate and create FIR matrix
    fir_mat = []
    task_ind = []
    tr_labeling = np.empty((np.sum(num_timepoints),),dtype=object) # Variable to TRs during a specific task
    for task in np.unique(df_taskonsets.task.values):
        task_fir_arr = np.zeros((np.sum(num_timepoints),blocklength + nRegsFIR))
        taskdf = df_taskonsets.loc[df_taskonsets.task==task]
        for run in np.unique(taskdf.run.values):
            rundf = taskdf.loc[taskdf.run==run]
            # count the TR at which each new run starts
            # for example, if run == 1, np.sum(num_timepoints[:,run-1] equals 0
            runstart_tr = np.sum(num_timepoints[:run-1]) 
            runend_tr = np.sum(num_timepoints[:run])
            for i in rundf.index:
                starttr = rundf.loc[i].onset + runstart_tr
                starttr = int(starttr)
                for j in range(blocklength + nRegsFIR):
                    # Make sure FIR regressor doesn't bleed to the next run
                    if starttr+j<runend_tr:
                        task_fir_arr[int(starttr + j), int(j)] = 1

                #### Now label TRs without the FIR lag (so no overlapping labels)
                for j in range(blocklength):
                    if starttr+j<runend_tr:
                        tr_labeling[int(starttr+j)] = task

        fir_mat.extend(task_fir_arr.T)
        task_ind.extend(np.repeat(task,blocklength+nRegsFIR))

    # Transpose to time x features/regressors
    fir_mat = np.asarray(fir_mat).T
                
    # Create temporal mask (skipping which frames?)
    output = {}
    # Commented out since we demean each run prior to loading data anyway
    output['taskRegressors'] = fir_mat 
    output['stimIndex'] = task_ind
    output['task_time_labels'] = tr_labeling

    return output

def loadTaskTimingBetaSeriesWholeBlock(sess, run, num_timepoints):
    """
    Added: 1/24/22 -- for signalNoiseFC project

    Loads the task timings for each task block (excluding the instruction period)
    This a beta series model, but each block is it's own beta
    Does not isolate individual trials/conditions within a task

    Parameters:

    subj 
        subject ID (as a string, e.g., '02')
    run
        run number
    num_timepoints
        an array containing number of time points in each run
    """

    blocklength = 30 # length of a task block
    trLength = 1.0
    stimdf = pd.DataFrame()
    subj = sess[:2] # first 2 characters form the subject ID
    sess_id = sess[-2:] # last 2 characters form the session
    tasktime_dir = datadir + 'sessions/' + sess + '/bids/func/'
    stimfile = glob.glob(tasktime_dir + 'sub-' + subj + '_ses-' + sess_id + '*' + str(run) + '_events.tsv')[0]
    stimdf = stimdf.append(pd.read_csv(stimfile,sep='\t'))

    #### Now find all task block onsets by identifying 
    df_taskonsets = {}
    df_taskonsets['task'] = []
    df_taskonsets['onset'] = []
    # Identify all instruction screens (i.e., the beginning of task blocks)
    stimdf = stimdf.reset_index() # Get new index
    instruction_ind = stimdf.loc[stimdf.trial_type=='Instruct'].index
    # Now add 1 to instruction index, since that's when the task starts
    task_ind = instruction_ind + 1
    for ind in task_ind:
        df_taskonsets['task'].append(stimdf.loc[ind].taskName)
        df_taskonsets['onset'].append(stimdf.loc[ind].startTRreal)
    df_taskonsets = pd.DataFrame(df_taskonsets)


    #### Name all tasks
    tasks = np.unique(df_taskonsets.task.values)

    stim_mat = np.zeros((num_timepoints,len(df_taskonsets)))
    stim_index = []

    stimcount = 0
    for task in tasks:
        taskdf = df_taskonsets.loc[df_taskonsets.task==task]
        for ind in taskdf.index:
            trstart = int(df_taskonsets.onset[ind])
            trend = int(trstart + blocklength)
            stim_mat[trstart:trend,stimcount] = 1.0

            stim_index.append(task)
            stimcount += 1 # go to next condition

                
    #### HRF Convolution
    # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
    taskStims_HRF = np.zeros(stim_mat.shape)
    spm_hrfTS = spm_hrf(trLength,oversampling=1)
   

    for stim in range(stim_mat.shape[1]):

        # Perform convolution
        tmpconvolve = np.convolve(stim_mat[:,stim],spm_hrfTS)
        taskStims_HRF[:,stim] = tmpconvolve[:num_timepoints]


    taskRegressors = taskStims_HRF.copy()

    # Create temporal mask (skipping which frames?)
    output = {}
    # Commented out since we demean each run prior to loading data anyway
    output['taskRegressors'] = taskRegressors 
    output['stimIndex'] = stim_index
    output['taskDesignMat'] = stim_mat

    return output

def loadNuisanceRegressors(sess, run, num_timepoints, model='qunex', spikeReg=False, zscore=False):
    """
    This function runs nuisance regression on the Glasser Parcels (360) on a single sessects run
    Will only regress out noise parameters given the model choice (see below for model options)
    Input parameters:
        sess    : sess number as a string
        run     : task run
        model   : model choices for linear regression. Models include:
                    1. 24pXaCompCorXVolterra [default]
                        Variant from Ciric et al. 2017. 
                        Includes (64 regressors total):
                            - Movement parameters (6 directions; x, y, z displacement, and 3 rotations) and their derivatives, and their quadratics (24 regressors)
                            - aCompCor (5 white matter and 5 ventricle components) and their derivatives, and their quadratics (40 regressors)
                    2. 18p (the legacy default)
                        Includes (18 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - Global signal and its derivative (2 regressors)
                            - White matter signal and its derivative (2 regressors)
                            - Ventricles signal and its derivative (2 regressors)
                    3. 16pNoGSR (the legacy default, without GSR)
                        Includes (16 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - White matter signal and its derivative (2 regressors)
                            - Ventricles signal and its derivative (2 regressors)
                    4. 12pXaCompCor (Typical motion regression, but using CompCor (noGSR))
                        Includes (32 regressors total):
                            - Movement parameters (6 directions) and their derivatives (12 regressors)
                            - aCompCor (5 white matter and 5 ventricle components) and their derivatives (no quadratics; 20 regressors)
                    5. 36p (State-of-the-art, according to Ciric et al. 2017)
                        Includes (36 regressors total - same as legacy, but with quadratics):
                            - Movement parameters (6 directions) and their derivatives and quadratics (24 regressors)
                            - Global signal and its derivative and both quadratics (4 regressors)
                            - White matter signal and its derivative and both quadratics (4 regressors)
                            - Ventricles signal and its derivative (4 regressors)
                    6. qunex (similar to 16p no gsr, but a variant) -- uses qunex output time series 
                        Includes (32 regressors total):
                            - Movement parameters (6 directions), their derivatives, and all quadratics (24 regressors)
                            - White matter, white matter derivatives, and their quadratics (4 regressors)
                            - Ventricles, ventricle derivatives, and their quadratics (4 regressors)
        spikeReg : spike regression (Satterthwaite et al. 2013) [True/False]
                        Note, inclusion of this will add additional set of regressors, which is custom for each session/run
        zscore   : Normalize data (across time) prior to fitting regression
    """

    # Load nuisance regressors for this data
    if model=='qunex':
        print('qunex is chosen')
        nuisdir = datadir + 'sessions/' + sess + '/images/functional/movement/'
        # Load physiological signals
        data = pd.read_csv(nuisdir + run + '.nuisance',sep='\s+')
        ventricles_signal = data.V.values[:-2]
        ventricles_signal_deriv = np.zeros(ventricles_signal.shape)
        ventricles_signal_deriv[1:] = ventricles_signal[1:] - ventricles_signal[:-1] 
        ventricles_signal_deriv[0] = np.mean(ventricles_signal_deriv[1:])
        #
        
        wm_signal = data.WM.values[:-2]
        wm_signal_deriv = np.zeros(wm_signal.shape)
        wm_signal_deriv[1:] = wm_signal[1:] - wm_signal[:-1] 
        wm_signal_deriv[0] = np.mean(wm_signal_deriv[1:])
        #
        global_signal = data.WB.values[:-2]
        global_signal_deriv = np.zeros(global_signal.shape)
        global_signal_deriv[1:] = global_signal[1:] - global_signal[:-1] 
        global_signal_deriv[0] = np.mean(global_signal_deriv[1:])
        #
        motiondat = pd.read_csv(nuisdir + run + '_mov.dat',sep='\s+')
        motionparams = np.zeros((len(motiondat),6)) # time x num params
        motionparams[:,0] = motiondat['dx(mm)'].values
        motionparams[:,1] = motiondat['dy(mm)'].values
        motionparams[:,2] = motiondat['dz(mm)'].values
        motionparams[:,3] = motiondat['X(deg)'].values
        motionparams[:,4] = motiondat['Y(deg)'].values
        motionparams[:,5] = motiondat['Z(deg)'].values
        motionparams_deriv = np.zeros((len(motiondat),6)) # time x num params
        motionparams_deriv[1:,:] = motionparams[1:,:] - motionparams[:-1,:] 
        motionparams_deriv[0,:] = np.mean(motionparams[1:,:],axis=0)
        ## Include quadratics
        motionparams_quadratics = motionparams**2
        motionparams_deriv_quadratics = motionparams_deriv**2

        ## EXCLUDE GLOBAL SIGNAL - my philosophical preference
        physiological_params = np.vstack((wm_signal,wm_signal_deriv,ventricles_signal,ventricles_signal_deriv)).T
        physiological_params_quadratics = physiological_params**2
        nuisanceRegressors = np.hstack((motionparams,motionparams_quadratics,motionparams_deriv,motionparams_deriv_quadratics,physiological_params,physiological_params_quadratics))

    else:
        'qunex is not chosen'
        # load all nuisance regressors for all other regression models
        h5f = h5py.File(nuis_reg_dir + sess + '_nuisanceRegressors.h5','r') 
        # Motion parameters + derivatives
        motion_parameters = h5f[run]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
        # Global signal + derivatives
        global_signal = h5f[run]['global_signal'][:].copy()
        global_signal_deriv = h5f[run]['global_signal_deriv'][:].copy()
        # white matter signal + derivatives
        wm_signal = h5f[run]['wm_signal'][:].copy()
        wm_signal_deriv = h5f[run]['wm_signal_deriv'][:].copy()
        # ventricle signal + derivatives
        ventricle_signal = h5f[run]['ventricle_signal'][:].copy()
        ventricle_signal_deriv = h5f[run]['ventricle_signal_deriv'][:].copy()
        # Create nuisance regressors design matrix
        tmp = np.vstack((global_signal,global_signal_deriv,wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))

    if model=='24pXaCompCorXVolterra':
        # Motion parameters + derivatives
        motion_parameters = h5f[run]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
        # WM aCompCor + derivatives
        aCompCor_WM = h5f[run]['aCompCor_WM'][:].copy()
        aCompCor_WM_deriv = h5f[run]['aCompCor_WM_deriv'][:].copy()
        # Ventricles aCompCor + derivatives
        aCompCor_ventricles = h5f[run]['aCompCor_ventricles'][:].copy()
        aCompCor_ventricles_deriv = h5f[run]['aCompCor_ventricles_deriv'][:].copy()
        # Create nuisance regressors design matrix
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, aCompCor_WM, aCompCor_WM_deriv, aCompCor_ventricles, aCompCor_ventricles_deriv))
        quadraticRegressors = nuisanceRegressors**2
        nuisanceRegressors = np.hstack((nuisanceRegressors,quadraticRegressors))
    
    elif model=='18p':
        # Motion parameters + derivatives
        motion_parameters = h5f[run]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
        # Global signal + derivatives
        global_signal = h5f[run]['global_signal'][:].copy()
        global_signal_deriv = h5f[run]['global_signal_deriv'][:].copy()
        # white matter signal + derivatives
        wm_signal = h5f[run]['wm_signal'][:].copy()
        wm_signal_deriv = h5f[run]['wm_signal_deriv'][:].copy()
        # ventricle signal + derivatives
        ventricle_signal = h5f[run]['ventricle_signal'][:].copy()
        ventricle_signal_deriv = h5f[run]['ventricle_signal_deriv'][:].copy()
        # Create nuisance regressors design matrix
        tmp = np.vstack((global_signal,global_signal_deriv,wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))

    elif model=='16pNoGSR':
        # Motion parameters + derivatives
        motion_parameters = h5f[run]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
        # white matter signal + derivatives
        wm_signal = h5f[run]['wm_signal'][:].copy()
        wm_signal_deriv = h5f[run]['wm_signal_deriv'][:].copy()
        # ventricle signal + derivatives
        ventricle_signal = h5f[run]['ventricle_signal'][:].copy()
        ventricle_signal_deriv = h5f[run]['ventricle_signal_deriv'][:].copy()
        # Create nuisance regressors design matrix
        tmp = np.vstack((wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))
    
    elif model=='12pXaCompCor':
        # Motion parameters + derivatives
        motion_parameters = h5f[run]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
        # WM aCompCor + derivatives
        aCompCor_WM = h5f[run]['aCompCor_WM'][:].copy()
        aCompCor_WM_deriv = h5f[run]['aCompCor_WM_deriv'][:].copy()
        # Ventricles aCompCor + derivatives
        aCompCor_ventricles = h5f[run]['aCompCor_ventricles'][:].copy()
        aCompCor_ventricles_deriv = h5f[run]['aCompCor_ventricles_deriv'][:].copy()
        # Create nuisance regressors design matrix
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, aCompCor_WM, aCompCor_WM_deriv, aCompCor_ventricles, aCompCor_ventricles_deriv))
    
    elif model=='36p':
        # Motion parameters + derivatives
        motion_parameters = h5f[run]['motionParams'][:].copy()
        motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
        # Global signal + derivatives
        global_signal = h5f[run]['global_signal'][:].copy()
        global_signal_deriv = h5f[run]['global_signal_deriv'][:].copy()
        # white matter signal + derivatives
        wm_signal = h5f[run]['wm_signal'][:].copy()
        wm_signal_deriv = h5f[run]['wm_signal_deriv'][:].copy()
        # ventricle signal + derivatives
        ventricle_signal = h5f[run]['ventricle_signal'][:].copy()
        ventricle_signal_deriv = h5f[run]['ventricle_signal_deriv'][:].copy()
        # Create nuisance regressors design matrix
        tmp = np.vstack((global_signal,global_signal_deriv,wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
        nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))
        quadraticRegressors = nuisanceRegressors**2
        nuisanceRegressors = np.hstack((nuisanceRegressors,quadraticRegressors))


    if spikeReg:
        # Obtain motion spikes
        try:
            motion_spikes = h5f[run]['motionSpikes'][:].copy()
            nuisanceRegressors = np.hstack((nuisanceRegressors,motion_spikes))
        except:
            print('Spike regression option was chosen... but no motion spikes for sess', sess, '| run', run, '!')
        # Update the model name - to keep track of different model types for output naming
        model = model + '_spikeReg' 

    if zscore:
        model = model + '_zscore'

    #if model!='qunex':
    #    h5f.close()
    
    return nuisanceRegressors

def loadRawParcellatedData(sess,run,datadir='/home/ln275/f_mc1689_1/MDTB/qunex_mdtb/sessions/',atlas='glasser'):
    """
    Load in parcellated data for given session and run
    """
    if atlas=='glasser':
        datafile = datadir + sess + '/images/functional/' + run + '_Atlas.LR.Parcels.32k_fs_LR.ptseries.nii'
    elif atlas=='schaefer':
        datafile = datadir + sess + '/images/functional/' + run + '_SchaeferAtlas.LR.Parcels.32k_fs_LR.ptseries.nii'
    elif atlas=='gordon':
        datafile = datadir + sess + '/images/functional/' + run + '_GordonAtlas.LR.Parcels.32k_fs_LR.ptseries.nii'
    data = nib.load(datafile).get_data()
    return data

def loadRawVertexData(sess,run,datadir='/home/ln275/f_mc1689_1/MDTB/qunex_mdtb/sessions/'):
    """
    Load in surface vertex data for given session and run
    """
    datafile = datadir + sess + '/images/functional/' + run + '_Atlas.dtseries.nii'
    data = nib.load(datafile).get_data()
    return data
