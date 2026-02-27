import numpy as np
import pickle
import nibabel as nib
from sklearn import metrics as skm
from tqdm import tqdm
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run activity flow predictions with permuted connectivity')
    parser.add_argument('--sub-idx', type=int, required=True, help='Subject index to process')
    parser.add_argument('--nperm', type=int, default=100, help='Number of permutations to run')
    parser.add_argument('--proj-dir', type=str, default='/home/ln275/f_mc1689_1/multitask_generalization/',
                        help='Project directory path')
    args = parser.parse_args()
    
    # Set up directories based on the project directory
    projdir = args.proj_dir
    fcdir = projdir + 'data/derivatives/FC_new/'
    helpfiles_dir = projdir + 'docs/experimentfiles/'
    subProjDir = projdir + 'data/derivatives/RSM_ActFlow/'
    
    # Task conditions subset (active visual task conditions)
    nTaskCond_select = 96 # excluding passive tasks and interval timing (auditory task)
    
    TaskCondIdx_subset = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,36,37,38,39,40,
                          41,42,43,44,45,46,49,50,51,52,53,54,55,56,60,61,62,63,64,65,
                          66,67,68,69,70,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,
                          87,88,89,91,92,93,94,95,96,97,98,99,100,101,102,104,105,106,
                          107,108,109,110,111,112,113,114,115,120,121,122,123,124,125])
    
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
    
    # Reading in task betas
    with open(subProjDir + 'allSubTaskCondBetas.pkl', 'rb') as f:
        allSubTaskBetas = pickle.load(f)
        
    # Get source vertices for each target
    with open(fcdir + 'allSub_sourcevert_sourcelabels_bytarget.pkl', 'rb') as f:
        allsub_sourcevert_sourcelabels = pickle.load(f)
    
    # Call the function with the specified subject index
    get_actflow_pred_betas_connperm(args.sub_idx, args.nperm, 
                                    subIDs, onlyRestSubIdx, glasser, 
                                    nParcels, nTaskCond_select, nSessions, 
                                    allSubTaskBetas, allsub_sourcevert_sourcelabels, 
                                    TaskCondIdx_subset, fcdir, subProjDir)
    
    print(f"Completed processing for subject {subIDs[args.sub_idx]}")

def get_actflow_pred_betas_connperm(subIdx, nperm, 
                                   subIDs, onlyRestSubIdx, glasser, 
                                   nParcels, nTaskCond_select, nSessions, 
                                   allSubTaskBetas, allsub_sourcevert_sourcelabels, 
                                   TaskCondIdx_subset, fcdir, subProjDir):
    '''
    Permute connectivity 100 times and rerunning actflow for betas
    '''

    # This subject's vertex-wise FC
    print('crossvalidated PCR FC is being used ...')
    with open(fcdir + 'sub'+ subIDs[subIdx] +'_vertFC_CVoptimal_nPCs.pkl', 'rb') as f:
        vertFC_dict = pickle.load(f)


    # This subject's task betas
    thisSubTaskCondBetas = allSubTaskBetas[onlyRestSubIdx[subIdx],:,TaskCondIdx_subset,:]

    # Running actflow to get predicted betas

    alltargetroi_alltaskcond_r = np.zeros((nTaskCond_select,nParcels))
    pred_target_betas_connperm_arr = []

    for targetroi_idx in range(nParcels):

        print('subIdx:',subIdx,'targetroi_idx:',targetroi_idx)
        
        target_vert = np.where(glasser == targetroi_idx+1)[0]

        sourceroi_vert = allsub_sourcevert_sourcelabels[subIdx][targetroi_idx][0]
        sourceroilabels = allsub_sourcevert_sourcelabels[subIdx][targetroi_idx][1]# 0 is sourcevert, 1 is sourceroilabels

        pred_targetROI_betas_connperm = np.zeros((nTaskCond_select,target_vert.shape[0],nSessions,nperm))

        for sess_idx in range(nSessions):

            this_targetROI_betas = thisSubTaskCondBetas[:,sess_idx,target_vert]

            this_sourceROI_betas = thisSubTaskCondBetas[:,sess_idx,sourceroi_vert]

            this_target_FC = vertFC_dict.get(targetroi_idx)

            with tqdm(total=nperm, desc="Progress") as pbar:

                for connpermIdx in range(nperm):
                    
                    # Get FC permuted at source level, but within each source (no mixing between vertices of different sources)
                    
                    this_target_FC_perm = np.zeros(this_target_FC.shape)

                    # Get unique source labels
                    unique_sources = np.unique(sourceroilabels)

                    # For each source, shuffle connectivity among its vertices
                    for source_id in unique_sources:
                        # Find indices for this bin
                        source_indices = np.where(sourceroilabels == source_id)[0]

                        # Skip if there's only one element in this bin
                        if len(source_indices) <= 1:
                            continue

                        # Get permutation of indices for this bin
                        perm_indices = np.random.permutation(len(source_indices))

                        # Apply permutation to this bin's values
                        this_target_FC_perm[:, source_indices] = this_target_FC[:, source_indices[perm_indices]]
                    
                    
                    pred_targetROI_betas_connperm[:,:,sess_idx,connpermIdx] = np.dot(this_sourceROI_betas,this_target_FC_perm.T)

                    pbar.update(1)

        pred_target_betas_connperm_arr.append(pred_targetROI_betas_connperm)

    with open(subProjDir + subIDs[subIdx]+'_actflow_pred_betas_connperm.pkl', 'wb') as f:
            pickle.dump(pred_target_betas_connperm_arr,f)

if __name__ == "__main__":
    main()
