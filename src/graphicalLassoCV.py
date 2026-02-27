import os
import numpy as np
from scipy import stats
from scipy import linalg
from gglasso.problem import glasso_problem
from sklearn.covariance import log_likelihood,empirical_covariance

#-----
    
def graphicalLassoCV(data,L1s,kFolds=10,optMethod='loglikelihood',saveFiles=0,outDir=''): 
    '''
    INPUT:
        data : a dataset with dimension [nDatapoints x nNodes]
        L1s : list of L1 (lambda1) hyperparameter values to test- scales the lasso penalty term
        kFolds : number of cross-validation folds to use during hyperparameter selection
        optMethod : method for choosing the optimal hyperparameter ('loglikelihood' or 'R2')
        saveFiles : whether to save intermediate and output variables to .npy files (0 = no, 1 = save R^2 and negative loglikelihood results, 2 = save connectivity and precision matrices too
        outDir : if saveFiles>=1, the directory in which to save the files
    OUTPUT:
        parCorr : graphical lasso (L1-regularized partial correlation) connectivity matrix, using optimal L1 value from list of L1s
        cvResults : dictionary with 'bestParam' for the optimal L1 value from input list of L1s and 'loglikelihood' or 'R2' containing scores for every cross validation fold and L1 value
    '''
    
    if not ((optMethod == 'loglikelihood') or (optMethod == 'R2')):
        raise ValueError(f'optMethod "{optMethod}" does not match available methods. Available options are "loglikelihood" and "R2".')
    
    L1s = np.array(L1s)
    
    # Divide timepoints into folds
    nTRs = data.shape[0]
    kFoldsTRs = np.full((kFolds,nTRs),False)     
    k = 0
    for t in range(nTRs):
        kFoldsTRs[k,t] = True
        k += 1
        if k >= kFolds:
            k = 0
    
    # Define arrays to hold performance metrics
    scores = np.zeros((len(L1s),kFolds))
    
    # If saving intermediate files
    if saveFiles >= 1:
        
        # Where to save files
        if outDir == '':
            outDir = os.getcwd()
            print(f'Directory for output files not provided, saving to current directory:\n{outDir}')
        if not os.path.exists(outDir):
            os.mkdir(outDir)
            
        # Loop through L1s
        for l,L1 in enumerate(L1s):
            outfileCVScores = f'{outDir}/L1-{L1}_{optMethod}.npy'
            
            # If performance metrics were already saved for this L1, load them and move on
            if os.path.exists(outfileCVScores):
                scores[l,:] = np.load(outfileCVScores)
                continue
            
            # Loop through folds
            for k in range(kFolds):
                outfileParCorr = f'{outDir}/L1-{L1}_kFolds-{k+1}of{kFolds}_partialCorr.npy'
                outfilePrec = f'{outDir}/L1-{L1}_kFolds-{k+1}of{kFolds}_precison.npy'
                
                # Check if partial corr and precision matrices were already created for this fold
                if os.path.exists(outfileParCorr) and os.path.exists(outfilePrec):
                    parCorr = np.load(outfileParCorr)
                    prec = np.load(outfilePrec)
                else:
                    # Estimate the regularized partial correlation and precision (intermediate) matrices
                    parCorr,prec = graphicalLasso(data[~kFoldsTRs[k],:],L1)
                    
                    # Save partial corr and precision matrices if saveFiles = 2
                    if saveFiles == 2:
                        np.save(outfileParCorr,parCorr)
                        np.save(outfilePrec,prec)
                
                if optMethod == 'loglikelihood':
                    # Calculate negative loglikelihood
                    empCov_test = np.cov(stats.zscore(data[kFoldsTRs[k],:],axis=0),rowvar=False)
                    scores[l,k] = -log_likelihood(empCov_test,prec)
                
                elif optMethod == 'R2':
                    # Calculate R^2
                    scores[l,k],r = activityPrediction(data[kFoldsTRs[k],:],parCorr)
                
            # Save performance metrics for this L1
            np.save(outfileCVScores,scores[l,:])
    
    # If not saving intermediate files
    else:
        # Loop through L1s
        for l,L1 in enumerate(L1s):
            # Loop through folds
            for k in range(kFolds):
                # Estimate the regularized partial correlation and precision (intermediate) matrices
                parCorr,prec = graphicalLasso(data[~kFoldsTRs[k],:],L1)
                
                if optMethod == 'loglikelihood':
                    # Calculate negative loglikelihood
                    empCov_test = np.cov(stats.zscore(data[kFoldsTRs[k],:],axis=0),rowvar=False)
                    scores[l,k] = -log_likelihood(empCov_test,prec)
                
                elif optMethod == 'R2':
                    # Calculate R^2
                    scores[l,k],r = activityPrediction(data[kFoldsTRs[k],:],parCorr)
    
    # Find the best param according to each performance metric 
    meanScores = np.mean(scores,axis=1)
    if optMethod == 'loglikelihood':
        bestParam = L1s[meanScores==np.amin(meanScores)]
    elif optMethod == 'R2':
        bestParam = L1s[meanScores==np.amax(meanScores)]
    
    # Estimate the regularized partial correlation using all data and the optimal hyperparameters
    parCorr,prec = graphicalLasso(data,bestParam)
    
    if saveFiles >= 1:
        np.save(f'{outDir}/bestL1_opt-{optMethod}.npy',bestParam)
        np.save(f'{outDir}/graphLasso_opt-{optMethod}.npy',parCorr)
    
    cvResults = {'bestParam': bestParam, optMethod: scores, 'L1s': L1s}
    
    return parCorr,cvResults


#-----
def graphicalLasso(data,L1):
    
    nNodes = data.shape[1]
    
    # Z-score the data
    data_scaled = stats.zscore(data,axis=0)
    
    # Estimate the empirical covariance
    empCov = np.cov(data_scaled,rowvar=False)
        
    # Number of timepoints in data
    nTRs = data.shape[0]
    
    # Run glasso
    glasso = glasso_problem(empCov,nTRs,reg_params={'lambda1':L1},latent=False,do_scaling=False)
    glasso.solve()
    prec = np.squeeze(glasso.solution.precision_)
    
    # Transform precision matrix into regularized partial correlation matrix
    denom = np.atleast_2d(1. / np.sqrt(np.diag(prec)))
    glassoParCorr = -prec * denom * denom.T
    np.fill_diagonal(glassoParCorr,0)
                
    return glassoParCorr,prec
      
    
#-----
def activityPrediction(activity,connMat):
    
    if activity.ndim == 2:
        activity = activity[:,:,np.newaxis]
        connMat = connMat[:,:,np.newaxis]
    
    nSubjs = activity.shape[2]
    nNodes = activity.shape[1]
    nodesList = np.arange(nNodes)
    
    R2 = np.zeros((nSubjs))
    pearson = np.zeros((nSubjs))
    
    prediction = np.zeros((activity.shape))
    
    for n in range(nNodes):
        otherNodes = nodesList!=n
        for s in range(nSubjs):
            X = activity[:,otherNodes,s]
            y = activity[:,n,s]
            betas = connMat[n,otherNodes,s]
            yPred = np.sum(X*betas,axis=1)
            prediction[:,n,s] = yPred
    
    for s in range(nSubjs):
        sumSqrReg = np.sum((activity[:,:,s]-prediction[:,:,s])**2)
        sumSqrTot = np.sum((activity[:,:,s]-np.mean(activity[:,:,s]))**2)
        R2[s] = 1 - (sumSqrReg/sumSqrTot)
        pearson[s] = np.corrcoef(activity[:,:,s].flatten(),prediction[:,:,s].flatten())[0,1]
    
    return R2,pearson