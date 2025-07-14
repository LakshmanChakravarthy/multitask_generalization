# utils

import os
import numpy as np
import scipy.stats as stats
import pickle
import nibabel as nib
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from brainsmash.mapgen.base import Base

# Paths
projdir = '/home/ln275/f_mc1689_1/multitask_generalization/'
subProjDir = projdir + 'data/derivatives/RSM_ActFlow/'
helpfiles_dir = projdir + 'docs/experimentfiles/'
fcdir = projdir + 'data/derivatives/FC_new/'
surfaceDeriv_dir = projdir + 'data/derivatives/surface/'


# Params
subIDs=['02','03','06','08','10','12','14','18','20',
        '22','24','25','26','27','28','29','30','31']

onlyRestSubIdx = [ 0, 1, 3, 4, 6, 7, 8,10,11,
                  12,20,13,21,14,22,15,23,16] # Use when dealing with full subjects' data (n=24)

nSub = len(subIDs)
nParcels = 360
nNetwork = 12
nBrainSys = 3
nVertices = 59412
nSessions = 2
nTask = 16
nTaskCond = 96 # excluding passive tasks and interval timing (auditory task)

TaskCondIdx_subset = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,36,37,38,39,40,
                      41,42,43,44,45,46,49,50,51,52,53,54,55,56,60,61,62,63,64,65,
                      66,67,68,69,70,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,
                      87,88,89,91,92,93,94,95,96,97,98,99,100,101,102,104,105,106,
                      107,108,109,110,111,112,113,114,115,120,121,122,123,124,125])

TaskIdx = np.array([1,1,2,2,3,3,3,3,3,3,3,3,7,7,4,4,5,5,7,7,8,8,8,2,2,9,9,9,9,11,11,
                    11,11,11,11,11,11,14,14,14,14,14,14,15,15,15,15,15,17,17,17,17,17,
                    17,17,17,17,17,17,17,18,18,18,18,18,18,19,19,19,19,19,19,19,19,19,
                    19,19,19,20,20,20,20,20,20,20,20,20,20,20,20,24,24,24,24,24,24])

tasknames = ['IAPS-Affective','Biological Motion','CPRO','IAPS-Emotional','Go-NoGo','Arithmetic','Word Prediction',
            'Theory of Mind','Stroop','Mental Rotation','Motor Sequence','Object n-back','Verbal n-back','Response Alt.',
             'Spatial Map','Visual Search']

# Help files
glasserfilename = helpfiles_dir + 'Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser = np.squeeze(nib.load(glasserfilename).get_fdata())

networkdef = np.loadtxt(helpfiles_dir + 'cortex_parcel_network_assignments.txt')
networkNames = ['VIS1','VIS2','SMN','CON','DAN','LAN','FPN','AUD','DMN','PMULTI','VMM','ORA']

# Brain system parsing

sensorynets = [1,2]
associationnets = [4,5,6,7,8,9,10,11,12]
motornets =[3]

roi_id = np.zeros((nParcels,))
for netw in range(1,nNetwork+1):
    thisnetROIs = np.where(networkdef==netw)[0]
    for roi in thisnetROIs:
        if netw in sensorynets:
            roi_id[roi] = 1

        elif netw in associationnets:
            roi_id[roi] = 2

        elif netw in motornets:
            roi_id[roi] = 3

sensory_roi_id = np.where(roi_id==1)[0]
association_roi_id = np.where(roi_id==2)[0]
motor_roi_id = np.where(roi_id==3)[0]

# File load

with open(fcdir + 'allSubFC_parCorr.pkl', 'rb') as f:
    allSubParCorr = pickle.load(f)

def get_task_parsed_RSM_idx():
    
    same_task = TaskIdx[:, None] == TaskIdx[None, :]

    same_task_idx = np.where(same_task)
    diff_task_idx = np.where(~same_task)

    # Dictionary to store indices for each label combination
    
    unique_tasks = np.unique(TaskIdx)
    
    taskcond_indices_of_task_pair = {}
    for taskA in unique_tasks:
        for taskB in unique_tasks:
            
            # Create a boolean mask for this label combination
            mask = (TaskIdx[:, None] == taskA) & (TaskIdx[None, :] == taskB)

            # Store the indices for this label combination
            taskcond_indices_of_task_pair[(taskA, taskB)] = np.where(mask)
            
    return diff_task_idx, taskcond_indices_of_task_pair
 
def get_cross_task_sim_pos_neg(RSM_data):
    
    '''
        RSM_data shape: (nParcels,nTaskCond,nTaskCond)
    '''
    
    diff_task_idx, _ = get_task_parsed_RSM_idx()

    cross_task_sim_pos = np.zeros(nParcels)
    cross_task_sim_neg = np.zeros(nParcels)
    for reg_idx in range(nParcels):

        diff_task_RSM_values = RSM_data[reg_idx][diff_task_idx]
        cross_task_sim_pos[reg_idx] = np.mean(diff_task_RSM_values[diff_task_RSM_values>0])
        cross_task_sim_neg[reg_idx] = np.mean(diff_task_RSM_values[diff_task_RSM_values<0])
        
    return cross_task_sim_pos,cross_task_sim_neg
    
def get_BS_averaged_data(data):
    '''
        data shape: (360,)
        return bs_avg shape: (3,)
    '''
    bs_avg = np.zeros(nBrainSys)
    for bsIdx in range(nBrainSys):
        bs_avg[bsIdx] = np.mean(data[np.where(roi_id==bsIdx+1)[0]])
        
    return bs_avg


def get_gradients():
    
    gradients = np.load(fcdir + 'meansub_FCgrad_3PCs_0.8thr.npy')
    
    return gradients
    
def get_brainsmash_perm(data,n_surrogates):
    
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    geo_dist_rh = np.loadtxt(helpfiles_dir + 'RightParcelGeodesicDistmat.txt')
    geo_dist_lh = np.loadtxt(helpfiles_dir + 'LeftParcelGeodesicDistmat.txt')
    base = Base(data[:180],geo_dist_lh)
    surrogates_lh = base(n=n_surrogates)
    base = Base(data[180:],geo_dist_rh)
    surrogates_rh = base(n=n_surrogates)
    surrogates_data = np.hstack((surrogates_lh,surrogates_rh))

    return surrogates_data
    
def get_brainsmash_correlation(x,y,n_perm=1000):
    
    obs_r,_ = stats.pearsonr(x,y)
    
    surrogate_x = get_brainsmash_perm(x,n_perm)
    
    surr_r = np.zeros(n_perm)
    for surr_idx in range(n_perm):
        surr_x = surrogate_x[surr_idx,:]
        surr_r[surr_idx],_ = stats.pearsonr(surr_x,y) 
    
    if obs_r > 0:
        pval = (np.sum(surr_r > obs_r))/n_perm
    else:
        pval = (np.sum(surr_r < obs_r))/n_perm
        
    return obs_r, pval
 
def get_residual(X,y):
    
    model = LinearRegression()
    model.fit(X, y)
    coef = np.array(model.coef_)

    y_pred = np.dot(coef,X.T)
    resid = y - y_pred.T
    
    return coef,resid
    
def get_quad_fit(x,y,n_perm=1000):

    '''
     Shape:
     Y: (nParcels,)
     X: (nParcels,)
    '''

    X = sm.add_constant(np.column_stack((x, x**2)))
    model = sm.OLS(y, X).fit()
    coef = model.params
    n_coef = len(coef)
    
    null_coef = np.zeros((n_perm,n_coef))
    surrogate_x = get_brainsmash_perm(x,n_perm) # shape:(1000,360)
    
    for surr_idx in range(n_perm):
        surr_x = surrogate_x[surr_idx,:]
        X = sm.add_constant(np.column_stack((surr_x, surr_x**2)))
        model = sm.OLS(y, X).fit()
        null_coef[surr_idx,:] = model.params
        
    pval = np.zeros(n_coef)
    for coef_idx in range(n_coef):
        if coef[coef_idx] > 0:
            pval[coef_idx] = (np.sum(null_coef[:,coef_idx] > coef[coef_idx]))/n_perm
        else:
            pval[coef_idx] = (np.sum(null_coef[:,coef_idx] < coef[coef_idx]))/n_perm
    
    return coef,pval


def get_dim_diff():
    
    '''
     returns dim_diff: target dimensionality - average source dimensionality
     positive values mean expansion, negative values mean compression 
    '''
    
    allsub_obs_dimensionality = get_obs_dimensionality()
    
    dim_diff = np.zeros((nSub,nParcels))
    mean_connreg_dim = np.zeros((nSub,nParcels))

    for subIdx in range(nSub):

        for targetroi_idx in range(nParcels):

            # get connected regions indices
            connected_regionsIdx = np.where(allSubParCorr[subIdx,targetroi_idx,:]!=0)[0]

            # get mean dimensionality of connected regions
            mean_connreg_dim[subIdx,targetroi_idx] = np.mean(allsub_obs_dimensionality[subIdx,connected_regionsIdx])
            dim_diff[subIdx,targetroi_idx] = allsub_obs_dimensionality[subIdx,targetroi_idx] - mean_connreg_dim[subIdx,targetroi_idx]
            
    return dim_diff

def get_region_size():
    
    region_size = np.zeros(nParcels)
    for roiIdx in range(nParcels):
        region_size[roiIdx] = len(np.where(glasser==roiIdx+1)[0])
        
    return region_size

def get_obs_RSM():
    
    with open(subProjDir + 'allsub_taskCond_RSM_activeVisual.pkl', 'rb') as f:
        allsub_taskCondRSM = pickle.load(f)
    
    select_sub_taskCondRSM = allsub_taskCondRSM[onlyRestSubIdx,:,:,:] # rest data's available only for 18 sub
    
    return select_sub_taskCondRSM

def get_pred_RSM():
    
    with open(subProjDir + 'allsub_taskCond_RSM_activeVisual_actflowpred_PCR_CVoptimal.pkl', 'rb') as f:
        allsub_predRSM = pickle.load(f)
    
    return allsub_predRSM

def get_pred_RSM_connperm():
    
    with open(subProjDir + 'allsub_actflow_pred_RSM_connperm.pkl', 'rb') as f:
        RSM_connperm = pickle.load(f)
        
    return RSM_connperm

def get_RSM_SNR():
    
    select_sub_taskCondRSM = get_obs_RSM()

    # Computing mean of RSM diagonal as SNR metric:
    allsub_RSM_SNR = np.zeros((nSub,nParcels))

    for subIdx in range(nSub):
        for regIdx in range(nParcels):
            allsub_RSM_SNR[subIdx,regIdx] = np.trace(select_sub_taskCondRSM[subIdx,regIdx,:,:])/nTaskCond

    meansub_RSM_SNR = np.mean(allsub_RSM_SNR,axis=0)
    
    return meansub_RSM_SNR

def get_parcelwise_intervertex_distance():
    
    with open(surfaceDeriv_dir + 'intervertex_dist_LR.pkl', 'rb') as f:
        intervertex_dist_LR = pickle.load(f)
    
    intervertex_dist_LR_parcelmean = np.zeros(nParcels)
    for roiIdx in range(nParcels):
        intervertex_dist_LR_parcelmean[roiIdx] = np.nanmean(intervertex_dist_LR[np.where(glasser==roiIdx+1)[0]])
        
    return intervertex_dist_LR_parcelmean

def get_betas_comparison_r():
    
    with open(subProjDir + 'allsub_meantaskcond_meansess_TaskBetas_comparison_r_rsq.pkl', 'rb') as f:
        meantaskcond_meansess_TaskBetas_comparison_r_arr = pickle.load(f)
    
    return meantaskcond_meansess_TaskBetas_comparison_r_arr

def get_betas_comparison_r_connperm():
    
    with open(subProjDir + 'allsub_meantaskcond_meansess_TaskBetas_comparison_r_connperm.pkl', 'rb') as f:
        r_arr_connperm_betas = pickle.load(f)
    
    return r_arr_connperm_betas

def get_RSM_comparison_cos_sim():
    
    with open(subProjDir + 'allsub_obs_vs_actflowpredRSM_compar.pkl', 'rb') as f:
        RSM_obs_vs_pred_cosSim = pickle.load(f)
        
    return RSM_obs_vs_pred_cosSim

def get_RSM_comparison_cos_sim_connperm():
    
    with open(subProjDir + 'allsub_meantaskcond_meansess_RSM_comparison_cosSim_connperm.pkl', 'rb') as f:
        cosSim_arr_connperm_RSM = pickle.load(f)
        
    return cosSim_arr_connperm_RSM

def get_obs_dimensionality():
    
    with open(subProjDir + 'allsub_dimensionality_activeVisual_observed_18sub.pkl', 'rb') as f:
        allsub_obs_dimensionality = pickle.load(f)
        
    return allsub_obs_dimensionality

def get_pred_dimensionality():
    
    with open(subProjDir + 'allsub_dimensionality_activeVisual_actflowpred_PCR_CVoptimal.pkl', 'rb') as f:
        allsub_dimensionality = pickle.load(f)
        
    return allsub_dimensionality

def get_pred_dimensionality_connperm():
    
    with open(subProjDir + 'allsub_dim_connperm.pkl', 'rb') as f:
        dim_connperm = pickle.load(f)
        
    return dim_connperm

def get_doubleCV_dimensionality():
    
    with open(subProjDir + 'allsub_doublecrossvalidated_dim_activeVisual_PCR_CVoptimal.pkl', 'rb') as f:
        allsub_obs_pred_CV_dimensionality = pickle.load(f)
        
    allsub_obs_pred_CV_dimensionality = np.mean(allsub_obs_pred_CV_dimensionality,axis=2) # avg across sess1->2 and 2->1
        
    return allsub_obs_pred_CV_dimensionality

def get_doubleCV_dimensionality_surr(n_surrogates = 100):
    
    file_path = subProjDir + 'allsub_doubleCV_dim_brainsmash_surr.pkl'
    
    if os.path.exists(file_path):
        
        with open(file_path, 'rb') as f:
            doubleCV_dim_surr = pickle.load(f)
    else:

        doubleCV_dim_surr = np.zeros((nSub,nParcels,n_surrogates))
        doubleCV_dim = get_doubleCV_dimensionality()
        
        for subIdx in range(nSub):
            print('subIdx:',subIdx)

            data = doubleCV_dim[subIdx,:]
            doubleCV_dim_surr[subIdx,:,:] = get_brainsmash_perm(data,n_surrogates).T
            
        with open(file_path, 'wb') as f:
            pickle.dump(doubleCV_dim_surr,f)
            
    return doubleCV_dim_surr

def get_FC_dimensionality():
    
    with open(subProjDir + 'allsub_alltargetSVPR.pkl', 'rb') as f:
        allsub_SVPR = pickle.load(f)
        
    return allsub_SVPR

def get_source_to_obs_target_dist():
    
    file_path = subProjDir + 'allsub_obs_sources_vs_target_cosSim.pkl'
    
    if os.path.exists(file_path):
    
        with open(file_path, 'rb') as f:
            RSM_obs_sources_vs_target_cosSim = pickle.load(f)
            
    else:
        
        print('fill this')
        
    source_to_obs_target_dist = 1 - RSM_obs_sources_vs_target_cosSim
        
    return source_to_obs_target_dist

def get_source_to_pred_target_dist():
    
    file_path = subProjDir + 'allsub_actflowpredRSM_sourcecompar.pkl'
    
    if os.path.exists(file_path):

        with open(file_path, 'rb') as f:
            RSM_obs_vs_pred_cosSim_sourcecompar = pickle.load(f)
            
    else:
        print('fill this')
        
    source_to_pred_target_dist = 1 - RSM_obs_vs_pred_cosSim_sourcecompar
    
    return source_to_pred_target_dist

def get_source_to_pred_target_dist_connperm():

    file_path = subProjDir + 'allsub_actflowpredRSM_sourcecompar_connperm.pkl'
    
    if os.path.exists(file_path):
    
        with open(file_path, 'rb') as f:
            RSM_obs_vs_pred_cosSim_sourcecompar_connperm = pickle.load(f)
            
    else:
        print('fill this')
        
    source_to_pred_target_dist_connperm = 1 - RSM_obs_vs_pred_cosSim_sourcecompar_connperm
    
    return source_to_pred_target_dist_connperm

def generate_group_null_from_perm(perm_data, n_groupnull = 1000, nPerm = 100):
    
    '''
     Example perm data shape: (nSub,nParcels,nPerm) or (nSub,nParcels,nTaskCond,nTaskCond,nPerm)
     Generating group null by taking mean across subjects' individual permuted values
    '''
    
    if perm_data.shape == (nSub,nParcels,nPerm): # usual
        
        groupnull = np.zeros((nParcels,n_groupnull))

        for groupnullIdx in range(n_groupnull):

            this_groupiter_arr = []

            for subIdx in range(nSub):

                this_sub_perm_data = perm_data[subIdx,:,np.random.randint(0, nPerm)]
                this_groupiter_arr.append(this_sub_perm_data)
                
            groupnull[:,groupnullIdx] = np.mean(np.array(this_groupiter_arr),axis=0)

    elif perm_data.shape == (nSub,nParcels,nTaskCond,nTaskCond,nPerm): # for RSM
        
        groupnull = np.zeros((nParcels,nTaskCond,nTaskCond,n_groupnull))

        for groupnullIdx in range(n_groupnull):

            this_groupiter_arr = []

            for subIdx in range(nSub):

                this_sub_perm_data = perm_data[subIdx,:,:,:,np.random.randint(0, nPerm)]
                this_groupiter_arr.append(this_sub_perm_data)
                
            groupnull[:,:,:,groupnullIdx] = np.mean(np.array(this_groupiter_arr),axis=0)

    return groupnull

def get_pval(observed_value,null_distribution,direction,n_null=1000):
    
    '''
        observed_value shape: (1,)
        null_distributions shape: (n_null,)
        
        direction:
            one-sided-higher: obs > null
            one-sided-lower: obs < null
            two-sided: obs different from null on both tails
    '''
    
    if direction == 'one-sided-higher':
        pval = (np.sum(null_distribution > observed_value))/n_null

    elif direction == 'one-sided-lower':
        pval = (np.sum(null_distribution < observed_value))/n_null

    elif direction == 'two-sided':
        pval = (np.sum(np.abs(null_distribution - np.mean(null_distribution)) >= 
                        np.abs(observed_value - np.mean(null_distribution))))/n_null
        
    return pval

def get_pval_parcelwise(observed_values,null_distributions,direction,n_null=1000):
    
    '''
        observed_values shape: (nParcels,)
        null_distributions shape: (nParcels,n_null)
        direction:
            one-sided-higher: obs > null
            one-sided-lower: obs < null
            two-sided: obs different from null on both tails
    '''

    pval = np.zeros(nParcels)
    for reg_idx in range(nParcels):
            
        observed_value = observed_values[reg_idx]
        null_distribution = null_distributions[reg_idx,:]

        pval[reg_idx] = get_pval(observed_value,null_distribution,direction,n_null=n_null)

    return pval

def compute_null_correlations(obs_data,null_data,n_null=1000):
    
    '''
        obs_data shape: (nParcels,)
        null_data shape: (nParcels,n_null)
    '''
    
    null_corr = np.zeros(n_null)
    for null_idx in range(n_null):
        null_corr[null_idx],_ = stats.pearsonr(null_data[:,null_idx], obs_data)
        
    return null_corr
