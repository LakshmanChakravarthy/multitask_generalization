# Relevant Actflow analyses for multitask project
# Lakshman NC March 14 2024

import numpy as np
import pickle
import nibabel as nib
from sklearn import metrics as skm
from tqdm import tqdm


# Paths
projdir = '/home/ln275/f_mc1689_1/multitask_generalization/'
fcdir = projdir + 'derivatives/FC_new/'
fcoutdir = fcdir+ 'vertexwiseFC/'
helpfiles_dir = projdir + 'docs/files/'
glasser_parcels_dir = helpfiles_dir + 'dilated_glasser_parcel_dscalar_files/'
subProjDir = projdir + 'derivatives/RSM_ActFlow/'


# Params and helpfiles
subIDs=['02','03','06','08','10','12','14','18','20',
        '22','24','25','26','27','28','29','30','31']

# Use when dealing with full subjects' data (n=24)
onlyRestSubIdx = [ 0, 1, 3, 4, 6, 7, 8,10,11,
                  12,20,13,21,14,22,15,23,16] 
nSub = len(subIDs)
nParcels = 360
nVertices = 59412
nSessions = 2

glasserfilename = helpfiles_dir + 'Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser = np.squeeze(nib.load(glasserfilename).get_fdata())
networkdef = np.loadtxt(helpfiles_dir + 'cortex_parcel_network_assignments.txt')
networkNames = ['VIS1','VIS2','SMN','CON','DAN','LAN','FPN','AUD','DMN','PMULTI','VMM','ORA']

# Reading in parcel-level FC:
with open(fcdir + 'allSubFC_parCorr.pkl', 'rb') as f:
    allSubParCorr = pickle.load(f)

# # Get source vertices for each target
# with open(fcdir + 'allSub_sourcevert_sourcelabels_bytarget.pkl', 'rb') as f:
#     allsub_sourcevert_sourcelabels = pickle.load(f)

## Task data

# Task conditions subset (active visual task conditions)
nTaskCond_select = 96 # excluding passive tasks and interval timing (auditory task)

TaskCondIdx_subset = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,36,37,38,39,40,
                      41,42,43,44,45,46,49,50,51,52,53,54,55,56,60,61,62,63,64,65,
                      66,67,68,69,70,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,
                      87,88,89,91,92,93,94,95,96,97,98,99,100,101,102,104,105,106,
                      107,108,109,110,111,112,113,114,115,120,121,122,123,124,125])

# # Reading in task betas
# with open(subProjDir + 'allSubTaskCondBetas.pkl', 'rb') as f:
#     allSubTaskBetas = pickle.load(f)
    
# # Reading in Actflow pred betas
# with open(subProjDir + 'allsub_actflow_pred_betas_PCR_CVoptimal.pkl', 'rb') as f:
#     alltarget_allsub_pred_betas_list_CVoptimal = pickle.load(f)
        
        
# read RSMs:
with open(subProjDir + 'allsub_taskCond_RSM_activeVisual.pkl', 'rb') as f:
    allsub_taskCondRSM = pickle.load(f)
    
select_sub_taskCondRSM = allsub_taskCondRSM[onlyRestSubIdx,:,:,:] # rest data's available only for 18 sub
   
# Reading in the permuted RSMs

with open(subProjDir + 'allsub_actflow_pred_RSM_connperm.pkl', 'rb') as f:
        RSM_connperm = pickle.load(f)
    
def get_actflow_pred_betas(subIdx, cvFC=True):

    '''
    cvFC = True for PCR FC with crossvalidated optimal number of PCs. If not, nPCs = 500
    '''
    
    # This subject's parcel-level FC
    thisSubParcelFC = allSubParCorr[subIdx,:,:]

    # This subject's vertex-wise FC

    if cvFC:
        print('crossvalidated PCR FC is being used ...')
        with open(fcdir + 'sub'+ subIDs[subIdx] +'_vertFC_CVoptimal_nPCs.pkl', 'rb') as f:
            vertFC_dict = pickle.load(f)
            vertFC_rsq_arr = pickle.load(f)  
    else:
        print('PCR FC WITH 500 components is being used ...')
        with open(fcdir + 'sub'+ subIDs[subIdx] +'_vertFC_500PCs.pkl', 'rb') as f:
            vertFC_dict = pickle.load(f)
            vertFC_rsq_arr = pickle.load(f)


    # This subject's task betas
    thisSubTaskCondBetas = allSubTaskBetas[onlyRestSubIdx[subIdx],:,TaskCondIdx_subset,:]

    # Running actflow to get predicted betas

    alltargetroi_alltaskcond_r = np.zeros((nTaskCond_select,nParcels))
    pred_target_betas_arr = []


    for targetroi_idx in range(nParcels):

        print('subIdx:',subIdx,'TargetIdx:',targetroi_idx)

        target_vert = np.where(glasser == targetroi_idx+1)[0]

        # vertices in target parcel (dilated)
        dil_dscalarfilename = glasser_parcels_dir + 'GlasserParcel'+str(targetroi_idx+1)+'_dilated_10mm.dscalar.nii'
        dilated_dscalar = np.squeeze(nib.load(dil_dscalarfilename).get_fdata())
        target_vert_dil = np.where(dilated_dscalar == 1)[0]

        nonzeroconn_target_indices = np.where(thisSubParcelFC[targetroi_idx,:]!=0)[0]    

        pred_targetROI_betas = np.zeros((nTaskCond_select,target_vert.shape[0],nSessions))

        for sess_idx in range(nSessions):

            this_targetROI_betas = thisSubTaskCondBetas[:,sess_idx,target_vert]

            # compile target vertices from all connected targets
            sourceroi_vert = []
            for sourceIdx in range(len(nonzeroconn_target_indices)):

                # vertices in source parcels, one at a time
                thisparcel_source_vert = np.where(glasser == nonzeroconn_target_indices[sourceIdx]+1)[0]

                # remove vertices that fall within dilated target parcel
                thisparcel_source_vert = thisparcel_source_vert[~np.isin(thisparcel_source_vert,target_vert_dil)]

                sourceroi_vert.append(thisparcel_source_vert)

            sourceroi_vert = np.concatenate(sourceroi_vert)

            this_sourceROI_betas = thisSubTaskCondBetas[:,sess_idx,sourceroi_vert]

            this_target_FC = vertFC_dict.get(targetroi_idx)
            
            # print('activity matrix shape:',this_sourceROI_betas.shape)
            # print('FC matrix shape:',this_target_FC.T.shape)

            pred_targetROI_betas[:,:,sess_idx] = np.dot(this_sourceROI_betas,this_target_FC.T)

        pred_target_betas_arr.append(pred_targetROI_betas)

    with open(subProjDir + subIDs[subIdx]+'_actflow_pred_betas_PCR_CVoptimal.pkl', 'wb') as f:
            pickle.dump(pred_target_betas_arr,f)
            
def compute_actflow_pred_RSM(subIdx,cvFC=True):
    
    allRegionRSM = []
    
    if cvFC:
        pred_betas_list = alltarget_allsub_pred_betas_list_CVoptimal
    else:
        pred_betas_list = alltarget_allsub_pred_betas_list_500
    
    for roi in range(1,nParcels+1):

        print('SubIdx:',subIdx,'roi:',roi)

        roiIdx = np.where(glasser==roi)[0]

        # compute cross-task similarity
        cos_sim_mat = np.zeros((nTaskCond_select,nTaskCond_select))
        for sess1Task in range(nTaskCond_select):
            for sess2Task in range(nTaskCond_select):

                vector1 = pred_betas_list[roi-1][subIdx][sess1Task,:,0].reshape(1,-1)
                vector2 = pred_betas_list[roi-1][subIdx][sess2Task,:,1].reshape(1,-1)

                cos_sim_mat[sess1Task,sess2Task] = skm.pairwise.cosine_similarity(vector1,vector2)[0][0]

        allRegionRSM.append(cos_sim_mat)

    with open(subProjDir + subIDs[subIdx]+'_taskCond_RSM_activeVisual_actflowpred_PCR_CVoptimal.pkl', 'wb') as f:
        pickle.dump(allRegionRSM,f)
        
def compute_doublecrossvalidated_RSM(subIdx,cvFC=True):
    
    '''
    One instance of RSM each by combining session1 and session2 of observed and actflow predicted betas
    '''
    allRegionRSM_1 = []
    allRegionRSM_2 = []
    
    if cvFC:
        pred_betas_list = alltarget_allsub_pred_betas_list_CVoptimal
    else:
        pred_betas_list = alltarget_allsub_pred_betas_list_500
        
    obs_betas = allSubTaskBetas[onlyRestSubIdx,:,:,:][:,:,TaskCondIdx_subset,:][:,:,:,:nVertices] # shape: (18,2,96,59412) 
    
    for roi in range(1,nParcels+1):

        print('SubIdx:',subIdx,'roi:',roi)

        roiIdx = np.where(glasser==roi)[0]

        # compute cross-task similarity
        cos_sim_mat_1 = np.zeros((nTaskCond_select,nTaskCond_select))
        cos_sim_mat_2 = np.zeros((nTaskCond_select,nTaskCond_select))
        
        for sess1Task in range(nTaskCond_select):
            for sess2Task in range(nTaskCond_select):

                vector1 = pred_betas_list[roi-1][subIdx][sess1Task,:,0].reshape(1,-1)
                vector2 = obs_betas[subIdx,1,sess2Task,roiIdx].reshape(1,-1)
                
                vector3 = pred_betas_list[roi-1][subIdx][sess1Task,:,1].reshape(1,-1)
                vector4 = obs_betas[subIdx,0,sess2Task,roiIdx].reshape(1,-1)

                cos_sim_mat_1[sess1Task,sess2Task] = skm.pairwise.cosine_similarity(vector1,vector2)[0][0]
                cos_sim_mat_2[sess1Task,sess2Task] = skm.pairwise.cosine_similarity(vector3,vector4)[0][0]

        allRegionRSM_1.append(cos_sim_mat_1)
        allRegionRSM_2.append(cos_sim_mat_2)

    with open(subProjDir + subIDs[subIdx]+'_doublecrossvalidated_RSM_activeVisual_PCR_CVoptimal.pkl', 'wb') as f:
        pickle.dump(allRegionRSM_1,f)
        pickle.dump(allRegionRSM_2,f)
        
        
def getSourceTargetRSMDissimilarity(subIdx):
    
    alltarget_cos_sim_mat = []
    
    with tqdm(total=nParcels, desc="Progress") as pbar:
    
        for target_roi_idx in range(nParcels):

            #print('subIdx:',subIdx,' targetIdx:',target_roi_idx)

            # get connected regions indices
            connected_regionsIdx = np.where(allSubParCorr[subIdx,target_roi_idx,:]!=0)[0]

            targetRSM = select_sub_taskCondRSM[subIdx,target_roi_idx,:,:].flatten().reshape(-1,1)

            cos_sim_mat = np.zeros(len(connected_regionsIdx))

            for i,conn_roiIdx in enumerate(connected_regionsIdx):

                sourceRSM_i = select_sub_taskCondRSM[subIdx,conn_roiIdx,:,:].flatten().reshape(-1,1)

                cos_sim_mat[i] = skm.pairwise.cosine_similarity(targetRSM,sourceRSM_i)[0][0]

            alltarget_cos_sim_mat.append(cos_sim_mat)
            
            pbar.update(1)
        
    with open(subProjDir + subIDs[subIdx]+'_SourceTargetRSMDissimilarity_activeVisual.pkl', 'wb') as f:
        pickle.dump(alltarget_cos_sim_mat,f)

def getTargetWiseSVPR(subIdx):

    with open(fcdir + 'sub'+ subIDs[subIdx] +'_vertFC_CVoptimal_nPCs.pkl', 'rb') as f2:
        vertFC_CVnPC_dict = pickle.load(f2)
        vertFC_CVnPC_rsq_arr = pickle.load(f2)
        vertFC_CVnPC_nPCs = pickle.load(f2)

    this_sub_svpr = np.zeros(nParcels)
    for targetroi_idx in range(nParcels):

        print('subIdx:',subIdx,' targetroi_idx:',targetroi_idx)

        this_target_FC = vertFC_CVnPC_dict.get(targetroi_idx)

        this_sub_svpr[targetroi_idx] = getSVPR(this_target_FC)
        
    with open(subProjDir + subIDs[subIdx]+'_alltargetSVPR.pkl', 'wb') as f:
        pickle.dump(this_sub_svpr,f)
        
def getSVPR(data):
    """
    singular value participation ratio
    """
    U,S,V_T = np.linalg.svd(data)
    
    dimensionality_nom = 0
    dimensionality_denom = 0
    for sv in S:
        dimensionality_nom += np.real(sv)
        dimensionality_denom += np.real(sv)**2

    dimensionality = dimensionality_nom**2/dimensionality_denom

    return dimensionality

def getTargetWise_FCEigenDecomp(subIdx):

    with open(fcdir + 'sub'+ subIDs[subIdx] +'_vertFC_CVoptimal_nPCs.pkl', 'rb') as f2:
        vertFC_CVnPC_dict = pickle.load(f2)
        vertFC_CVnPC_rsq_arr = pickle.load(f2)
        vertFC_CVnPC_nPCs = pickle.load(f2)

    this_sub_fc_eigendecomp = np.zeros(nParcels) # target space only
    for targetroi_idx in range(nParcels):

        print('subIdx:',subIdx)

        this_target_FC = vertFC_CVnPC_dict.get(targetroi_idx)
        #print('this_target_FC shape:', this_target_FC.shape)
        
        this_sub_fc_eigendecomp[targetroi_idx] = getFCEigenDecomp(this_target_FC)
        
    with open(subProjDir + subIDs[subIdx]+'_alltarget_FCEigenDecomp.pkl', 'wb') as f:
        pickle.dump(this_sub_fc_eigendecomp,f)
        
        
def getsubwise_thr_conv(subIdx):

    with open(fcdir + 'sub'+ subIDs[subIdx] +'_vertFC_CVoptimal_nPCs.pkl', 'rb') as f2:
        vertFC_CVnPC_dict = pickle.load(f2)
        vertFC_CVnPC_rsq_arr = pickle.load(f2)
        vertFC_CVnPC_nPCs = pickle.load(f2)

    alltarget_thr_conv_diff = []
    alltarget_thr_conv_ratio = []

    with tqdm(total=nParcels, desc="Progress") as pbar:
    
        for targetroi_idx in range(nParcels):

            this_target_FC = vertFC_CVnPC_dict.get(targetroi_idx)
            pref_density,asymp_conv,allthr_thr_conv_diff,allthr_thr_conv_ratio = get_thr_conv(this_target_FC)

            alltarget_thr_conv_diff.append(allthr_thr_conv_diff)
            alltarget_thr_conv_ratio.append(allthr_thr_conv_ratio)
            
            pbar.update(1)
            
    with open(subProjDir + subIDs[subIdx]+'_thrconv_diff_and_ratio.pkl', 'wb') as f:
        pickle.dump(alltarget_thr_conv_diff,f)
        pickle.dump(alltarget_thr_conv_ratio,f)
        
def get_thr_conv(this_target_FC,start = 5,end = 100, stepsize = 5, cutoff = -0.1):

    thr_value_arr = np.arange(start,end,stepsize)

    allthr_thr_conv_diff = np.zeros(len(thr_value_arr))
    allthr_thr_conv_ratio = np.zeros(len(thr_value_arr))

    for thrIdx,thr in enumerate(thr_value_arr):

        threshold = np.percentile(np.abs(this_target_FC), thr)
        this_target_FC_thr = np.where(np.abs(this_target_FC) >= threshold, this_target_FC, 0)

        in_deg_dist,out_deg_dist = get_deg(this_target_FC_thr,deg_type='abs')

        mean_indeg_nonzero = np.mean(in_deg_dist[in_deg_dist != 0])
        mean_outdeg_nonzero = np.mean(out_deg_dist[out_deg_dist != 0])

        allthr_thr_conv_diff[thrIdx] = mean_indeg_nonzero - mean_outdeg_nonzero
        allthr_thr_conv_ratio[thrIdx] = mean_indeg_nonzero / mean_outdeg_nonzero

        
    # Looking at asymptotic convergence value for ratio-based measure
    kinkIdx = np.where(np.diff(allthr_thr_conv_ratio) > cutoff)[0][-1]
    
    pref_density = thr_value_arr[kinkIdx]
    asymp_conv = allthr_thr_conv_ratio[kinkIdx]

    return pref_density,asymp_conv,allthr_thr_conv_diff,allthr_thr_conv_ratio

def get_deg(this_target_FC,deg_type='abs'):
    
    if deg_type=='abs':
        this_target_FC_trans = np.abs(this_target_FC) # trans meaning transformed
        
    elif deg_type=='non_neg':
        this_target_FC_trans = this_target_FC.copy()
        neg_idx = np.where(this_target_FC_trans<0)
        this_target_FC_trans[neg_idx] = 0
        
    elif deg_type=='zeromin':
        this_target_FC_trans = this_target_FC.copy()
        this_target_FC_trans = this_target_FC_trans - np.min(this_target_FC)
    
    in_deg_dist = np.sum(this_target_FC_trans,axis=1)
    out_deg_dist = np.sum(this_target_FC_trans,axis=0)
        
    return in_deg_dist,out_deg_dist
        
def getFCEigenDecomp(data):
    """
    Eigen decomposition of FC matrix's inner and outer products, but normalized using corrcoef
    """
#     # Source space dimensionality
#     print('Computing source space dimensionality ...')
#     sourcespace = np.corrcoef(data.T)
#     print('sourcespace shape:',sourcespace.shape)
    
#     sourcespacedim = getDimensionality(sourcespace)
    
    #print('Computing target space dimensionality ...')
    targetspace = np.corrcoef(data)
    #print('targetspace shape:',targetspace.shape)
    targetspacedim = getDimensionality(targetspace)

    return targetspacedim

def getDimensionality(data):
    """
    data needs to be a square, symmetric matrix
    """
    corrmat = data
    
    #print('Computing eigen decomposition')
    eigenvalues, eigenvectors = np.linalg.eig(corrmat)
    
    #print('Computing participation ratio')
    dimensionality_nom = 0
    dimensionality_denom = 0
    for eig in eigenvalues:
        dimensionality_nom += np.real(eig)
        dimensionality_denom += np.real(eig)**2

    dimensionality = dimensionality_nom**2/dimensionality_denom

    return dimensionality

def getSourceSimilarity(subIdx):
    
    all_target_sourcesimmat = []
    
    with tqdm(total=nParcels, desc="Progress") as pbar:
    
        for targetroi_idx in range(nParcels):

            # print('targetIdx:',targetroi_idx)

            connected_regionsIdx = np.where(allSubParCorr[subIdx,targetroi_idx,:]!=0)[0]

            nSources = len(connected_regionsIdx)

            source_sim_mat = np.zeros((nSources,nSources))

            for sourceIdx_1 in range(nSources):
                for sourceIdx_2 in range(sourceIdx_1,nSources):

                    vector1 = select_sub_taskCondRSM[subIdx][connected_regionsIdx[sourceIdx_1]].flatten().reshape(1,-1)
                    vector2 = select_sub_taskCondRSM[subIdx][connected_regionsIdx[sourceIdx_2]].flatten().reshape(1,-1)

                    source_sim_mat[sourceIdx_1,sourceIdx_2] = skm.pairwise.cosine_similarity(vector1,vector2)[0][0]
            
            all_target_sourcesimmat.append(source_sim_mat)
            
            pbar.update(1)
            
    with open(subProjDir + subIDs[subIdx]+'_all_target_sourcesimmat.pkl', 'wb') as f:
        pickle.dump(all_target_sourcesimmat,f)
        
        
def get_actflow_pred_betas_connperm(subIdx,nperm=100):

    '''
    Permute connectivity 100 times and rerunning actflow for betas
    '''

    # This subject's vertex-wise FC
    print('crossvalidated PCR FC is being used ...')
    with open(fcdir + 'sub'+ subIDs[subIdx] +'_vertFC_CVoptimal_nPCs.pkl', 'rb') as f:
        vertFC_dict = pickle.load(f)
        vertFC_rsq_arr = pickle.load(f)


    # This subject's task betas
    thisSubTaskCondBetas = allSubTaskBetas[onlyRestSubIdx[subIdx],:,TaskCondIdx_subset,:]

    # Running actflow to get predicted betas

    alltargetroi_alltaskcond_r = np.zeros((nTaskCond_select,nParcels))
    pred_target_betas_connperm_arr = []

    for targetroi_idx in range(nParcels):

        print('subIdx:',subIdx,'targetroi_idx:',targetroi_idx)
        
        target_vert = np.where(glasser == targetroi_idx+1)[0]

        sourceroi_vert = allsub_sourcevert_sourcelabels[subIdx][targetroi_idx][0] # 0 is sourcevert, 1 is sourceroilabels

        pred_targetROI_betas_connperm = np.zeros((nTaskCond_select,target_vert.shape[0],nSessions,nperm))

        for sess_idx in range(nSessions):

            this_targetROI_betas = thisSubTaskCondBetas[:,sess_idx,target_vert]

            this_sourceROI_betas = thisSubTaskCondBetas[:,sess_idx,sourceroi_vert]

            this_target_FC = vertFC_dict.get(targetroi_idx)

            with tqdm(total=nperm, desc="Progress") as pbar:

                for connpermIdx in range(nperm):

                    this_target_FC_flat = this_target_FC.flatten()
                    perm_temp = np.random.permutation(this_target_FC_flat)
                    this_target_FC_perm = perm_temp.reshape(this_target_FC.shape)

                    pred_targetROI_betas_connperm[:,:,sess_idx,connpermIdx] = np.dot(this_sourceROI_betas,this_target_FC_perm.T)

                    pbar.update(1)

        pred_target_betas_connperm_arr.append(pred_targetROI_betas_connperm)

            

    with open(subProjDir + subIDs[subIdx]+'_actflow_pred_betas_connperm.pkl', 'wb') as f:
            pickle.dump(pred_target_betas_connperm_arr,f)
        
        
def get_transformation_distance_connperm(subIdx,nPerm=100):
    
    print('SubIdx:',subIdx)
    
    RSM_obs_vs_predConnPerm_cosSim_sourcecompar = np.zeros((nParcels,nPerm))
    
    with tqdm(total=nParcels, desc="Progress") as pbar:
    
        for roiIdx in range(nParcels):

            source_indices = np.where(allSubParCorr[subIdx,roiIdx,:]!=0)[0]

            with_allsources_cosSim = np.zeros((len(source_indices),nPerm))
            for sourceIdx in range(len(source_indices)):

                obsRSM_flat = select_sub_taskCondRSM[subIdx,source_indices[sourceIdx],:,:].flatten() # obs
                
                for permIdx in range(nPerm):
                    predRSM_flat = RSM_connperm[subIdx,roiIdx,:,:,permIdx].flatten() # pred

                    # Compute cosine similarity
                    obsRSM_flat_normed = obsRSM_flat / np.linalg.norm(obsRSM_flat)
                    predRSM_flat_normed = predRSM_flat / np.linalg.norm(predRSM_flat)

                    cosSim = np.dot(obsRSM_flat_normed, predRSM_flat_normed.T)

                    with_allsources_cosSim[sourceIdx,permIdx] = cosSim

                    meansource_cosSim = np.mean(with_allsources_cosSim,axis=0)

            RSM_obs_vs_predConnPerm_cosSim_sourcecompar[roiIdx,:] = meansource_cosSim
            
            pbar.update(1)
            
    with open(subProjDir + subIDs[subIdx]+'_actflowpredRSM_sourcecompar_connperm.pkl', 'wb') as f:
            pickle.dump(RSM_obs_vs_predConnPerm_cosSim_sourcecompar,f)
            
def get_transformation_distance_randomTrans(subIdx,nPerm=100):
    
    print('SubIdx:',subIdx)

    RSM_obs_vs_randomTrans_cosSim_sourcecompar = np.zeros((nParcels,nPerm))

    with tqdm(total=nParcels, desc="Progress") as pbar:

        for roiIdx in range(nParcels):

            source_indices = np.where(allSubParCorr[subIdx,roiIdx,:]!=0)[0]

            with_allsources_cosSim = np.zeros((len(source_indices),nPerm))
            for sourceIdx in range(len(source_indices)):

                obsSourceRSM = select_sub_taskCondRSM[subIdx,source_indices[sourceIdx],:,:] # obs source
                obsSourceRSM_flat = obsSourceRSM.flatten()
                obsSourceRSM_flat_normed = obsSourceRSM_flat / np.linalg.norm(obsSourceRSM_flat)

                for permIdx in range(nPerm):

                    random_weightMat = np.random.rand(nTaskCond_select,nTaskCond_select) - 0.5

                    random_trans_RSM = np.dot(obsSourceRSM,random_weightMat.T)

                    # Compute cosine similarity
                    random_trans_RSM_flat = random_trans_RSM.flatten()
                    random_trans_RSM_normed = random_trans_RSM_flat / np.linalg.norm(random_trans_RSM_flat)

                    cosSim = np.dot(obsSourceRSM_flat_normed, random_trans_RSM_normed.T)

                    with_allsources_cosSim[sourceIdx,permIdx] = cosSim

                    meansource_cosSim = np.mean(with_allsources_cosSim,axis=0)

            RSM_obs_vs_randomTrans_cosSim_sourcecompar[roiIdx,:] = meansource_cosSim

            pbar.update(1)
            
    with open(subProjDir + subIDs[subIdx]+'_actflowpredRSM_sourcecompar_randomTrans.pkl', 'wb') as f:
            pickle.dump(RSM_obs_vs_randomTrans_cosSim_sourcecompar,f)