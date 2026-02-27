# This code measures sparsity in the functional connectivity (FC) data corresponding to each region, in each subject. 
# Sparsity is defined as the proportion of values which are under a certain threshold (thr).

import pickle
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../src')
import utils

projdir = '/home/ln275/f_mc1689_1/multitask_generalization/'
fcdir = projdir + 'data/derivatives/FC_new/'

subIDs=['02','03','06','08','10','12','14','18','20',
        '22','24','25','26','27','28','29','30','31']

nSub = len(subIDs)
nParcels = 360

# Define threshold values
thresholds = [0.0001, 0.00005, 0.00001]
nThresholds = len(thresholds)

# Initialize array to store sparsity values: (nSub, nParcels, nThresholds)
sparsity_all = np.zeros((nSub, nParcels, nThresholds))

# Loop over subjects
for subIdx in tqdm(range(nSub), desc='Processing subjects'):
    # Load FC data for this subject
    with open(fcdir + 'sub'+ subIDs[subIdx] +'_vertFC_CVoptimal_nPCs.pkl', 'rb') as f:
        vertFC_CVnPC_dict = pickle.load(f)
    
    # Loop over regions
    for roi_idx in range(nParcels):
        # Get FC matrix for this region
        fc_matrix = vertFC_CVnPC_dict[roi_idx]
        full_size = fc_matrix.shape[0] * fc_matrix.shape[1]
        
        # Loop over thresholds
        for thr_idx, thr in enumerate(thresholds):
            # Count values below threshold
            small_val = np.where(np.abs(fc_matrix) < thr)[0].shape[0]
            prop = small_val / full_size
            
            # Store result
            sparsity_all[subIdx, roi_idx, thr_idx] = prop

# Save results
with open(fcdir + 'allsub_sparsity_count.pkl', 'wb') as f:
    pickle.dump(sparsity_all, f)

print(f"\nSparsity analysis complete!")
print(f"Output shape: {sparsity_all.shape}")
print(f"Saved to: {fcdir}allsub_sparsity_count.pkl")

# # Access FC dimensionality of all subjects and regions: shape: (nSub,nParcels)
# allsub_conn_dim = utils.get_FC_dimensionality()