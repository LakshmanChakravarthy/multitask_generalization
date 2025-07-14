# Lakshman NC
#
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
import glob
import argparse
np.set_printoptions(suppress=True)
import postproc_tools as pptools
import sys  
sys.path.insert(0, '/home/ln275/f_mc1689_1/MDTB/huth_ridge')
from huth_ridge import ridge, ridge_corr, bootstrap_ridge_modified

## Define GLOBAL variables (variables accessible to all functions
# Define base data directory
# datadir = '/gpfs/loomis/project/n3/Studies/MurrayLab/taku/multiTaskVAE/qunexMultiTaskVAE/'
datadir = '/home/ln275/f_mc1689_1/MDTB/qunex_mdtb/'
# Define number of frames to skip
framesToSkip = 5
# Define the *output* directory for nuisance regressors
nuis_reg_dir = datadir + 'nuisanceRegressors/'
# Create directory if it doesn't exist
if not os.path.exists(nuis_reg_dir): os.makedirs(nuis_reg_dir)
# Define the *output* directory for preprocessed data
#

# All subjects
sessions=['02_a1','02_a2','02_b1','02_b2','03_a1','03_a2','03_b1','03_b2','04_a1','04_a2','04_b1','04_b2','06_a1','06_a2','06_b1','06_b2','09_a1','09_a2','09_b1','09_b2','12_a1','12_a2','12_b1','12_b2','15_a1','15_a2','15_b1','15_b2','18_a1','18_a2','18_b1','18_b2','20_a1','20_a2','20_b1','20_b2','22_a1','22_a2','22_b1','22_b2','25_a1','25_a2','25_b1','25_b2','27_a1','27_a2','27_b1','27_b2','29_a1','29_a2','29_b1','29_b2','31_a1','31_a2','31_b1','31_b2','02_a1','02_a2','02_b1','02_b2','04_a1','04_a2','04_b1','04_b2','08_a1','08_a2','08_b1','08_b2','10_a1','10_a2','10_b1','10_b2','14_a1','14_a2','14_b1','14_b2','17_a1','17_a2','17_b1','17_b2','19_a1','19_a2','19_b1','19_b2','21_a1','21_a2','21_b1','21_b2','24_a1','24_a2','24_b1','24_b2','26_a1','26_a2','26_b1','26_b2','28_a1','28_a2','28_b1','28_b2','30_a1','30_a2','30_b1','30_b2']

subIDs=['02','03','04','06','08','09','10','12','14','15','18','20','22','25','27','29','31','17','19','21','24','26','28','30']

nruns = 8
#subIDs = ['14']
sessionIDs=['_a1','_a2','_b1','_b2']
outputdir = datadir + '../derivatives/postprocessing/betasByTaskCondition/'


parser = argparse.ArgumentParser('./main.py', description='Run postprocessing')
parser.add_argument('--space', type=str, default="parcellated", help="'parcellated' or 'vertex'")
parser.add_argument('--model', type=str, default="qunex", help="For now, qunex is the only implemented option")
parser.add_argument('--zscore', action='store_true', help='zscore data before regression')
parser.add_argument('--spikereg', action='store_true', help='implement spike regression')
parser.add_argument('--output_suffix', type=str, default="", help="output suffix")
parser.add_argument('--verbose', action='store_true', help='verbose')


def run(args):
    """
    Perform task GLM in conjunction with nuisance regression
    """
    args 
    space = args.space
    model = args.model
    zscore = args.zscore
    spikereg = args.spikereg
    output_suffix = args.output_suffix
    verbose = args.verbose
    
    # Iterate through each subject
    runs = range(1,nruns+1)
    
    
    #runs = [8]
    #sess_suffix = ['a1']

    for subIdx in range(0,4):
        
        subj = subIDs[subIdx]

        sess_suffix = ['a1','a2','b1','b2']
        
        # Note all Rest fMRI runs are in ${sub}_b2
        # Note rest runs are bold9 and bold10.nii.gz
        # Iterate through each run
        for sessIdx,sess_id in enumerate(sess_suffix):
            sess = subj + '_' + sess_id
            for run in runs:
                print ('Running task regression on session', sess, '| run', run)
                print ('\tModel:', model, '| beta series | spikereg:', spikereg, '| zscore:', zscore)
                # Run nuisance regression for this subject's run, using a helper function defined below
                # Data will be output in 'outputdir', defined above
                outputfilename = outputdir + sess + '_tfMRI_' + space + '_betaseries_' + model + '_bold' + str(run) + output_suffix

                run_id = 'bold' + str(run)
                try: 
                    if space=='parcellated':
                        rundata = pptools.loadRawParcellatedData(sess,run_id)
                    elif space=='vertex':
                        rundata = pptools.loadRawVertexData(sess,run_id)
                except:
                    print('Looks like no files')
                    continue

                num_timepoints = rundata.shape[0]

                tMask = np.ones((num_timepoints,))
                tMask[:framesToSkip] = 0
                tMask = np.asarray(tMask,dtype=bool)

                # Skip frames
                rundata = rundata[tMask,:]
                print('rundata_shape:',rundata.shape)
                
    #            # Demean each run
    #            rundata = signal.detrend(rundata,axis=0,type='constant')
                # Detrend each run
                rundata = signal.detrend(rundata,axis=0,type='linear')
                
                # Load in nuisance regressors
                nuisregs = pptools.loadNuisanceRegressors(sess,run_id,num_timepoints,model=model)
                # Skip nuis regs frames
                nuisregs = nuisregs[tMask,:]

                data = rundata
                nROIs = data.shape[1]

                #  Load in task timing file (including instructions)
                runIdx = run-1
                tasktiming = pptools.loadBetasByTaskCondition(subIdx, sessIdx, runIdx, num_timepoints)
                task_regs = tasktiming['taskRegressors'][tMask,:]
                regression_index = np.asarray(tasktiming['stimIndex'])
                

                allRegressors = np.hstack((task_regs,nuisregs))
                print('allRegressors shape:',allRegressors.shape)

                if zscore:
                    data = stats.zscore(data,axis=0)
                    allRegressors = stats.zscore(allRegressors,axis=0)
                   
                ########################################################
                # Ridge regression using alexhuth code
                # Hyperparameter selection using crossvalidation
                
                nfold = 4
                split_size = int(allRegressors.shape[0]/nfold)
                alphas = np.logspace(0,3,20)

                allFoldRcorrs =[]
                for i in range(nfold):
                    test_idx = range(i*split_size,(i+1)*split_size)
                    train_idx = np.delete(range(allRegressors.shape[0]),test_idx)

                    Rstim = allRegressors[train_idx,:]
                    Pstim = allRegressors[test_idx,:]
                    Rresp = data[train_idx,:]
                    Presp = data[test_idx,:]

                    Rcorrs = ridge_corr(Rstim, Pstim, Rresp, Presp, alphas, normalpha=False,                                                   corrmin=0.2, singcutoff=1e-10, use_corr=True)

                    allFoldRcorrs.append(Rcorrs)

                allFoldRcorrs = np.array(allFoldRcorrs)
                avgFoldRcorr = np.mean(allFoldRcorrs,axis=0)
                best_alpha = alphas[np.argmax(avgFoldRcorr,axis=0)]

                # weights based on best alpha
                betas = ridge(allRegressors, data, alpha=best_alpha)

                #####################################################
                
                print('Betas shape:',betas.shape)
                #print('Best alpha:',bestalpha)
                y_pred = np.dot(allRegressors,betas)
                residual_ts = data - y_pred

                # For task data, only include task-related regressors
                betas = betas[:len(regression_index),:]
                print(betas.shape)
                
                if verbose: print('\tOutputting processed data:', outputfilename)

                # Save out index file that indicates each beta coef with a different task condition
                np.savetxt(outputfilename + '_taskIndex.csv', regression_index,delimiter=',',fmt ='% s')
                
                h5f = h5py.File(outputfilename + '.h5','a')
                outname1 = 'residuals'
                outname2 = 'betas'
                try:
                    h5f.create_dataset(outname1,data=residual_ts)
                    h5f.create_dataset(outname2,data=betas)
                except:
                    del h5f[outname1], h5f[outname2]
                    h5f.create_dataset(outname1,data=residual_ts)
                    h5f.create_dataset(outname2,data=betas)
                h5f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
