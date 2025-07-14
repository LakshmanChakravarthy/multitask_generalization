import pickle
import numpy as np
from sklearn import metrics as skm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from matplotlib import colors
import nibabel as nib
from scipy.linalg import subspace_angles
import os
from scipy.stats import ttest_rel
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

### Paths and parameters ###
projDir = '/home/ln275/f_mc1689_1/multitask_generalization/'
intermFilesDir = projDir + 'data/derivatives/local_upload/' 
helpfiles_dir = projDir + 'docs/experimentfiles/'
figoutdir = projDir + 'docs/figures/working/'

nSub = 24
nParcels = 360
nTask = 24
nTaskSubspace = 22 # as two out of 24 tasks have just one condition each
nTaskCond = 134
nBrainSystem = 3
nNetwork = 12

### Task Condition Info ###
taskCond_df = pd.read_csv(intermFilesDir + 'TaskConditionsList.csv')

taskCondName = np.array(taskCond_df.TaskCondName)
keyNames = np.array(taskCond_df.Key)
taskIndex = np.array(taskCond_df.TaskIndex)
stimType1 = np.array(taskCond_df.stimType1)
stimType2 = np.array(taskCond_df.stimType2)
stimType_sameflag = np.array(taskCond_df.stimType_sameflag)

# for select tasks for between/within task comparison: should have atleast two conditions within
selectTaskIndexList = []
for taskIndexIdx in range(1,nTask+1):
    thisTask_CondIdx = np.where(taskIndex == taskIndexIdx)[0]

    if len(thisTask_CondIdx)>1:
        selectTaskIndexList.append(taskIndexIdx)

### Parcellation and network details ###
# Glasser parcellation
glasserfilename = helpfiles_dir + 'Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser = np.squeeze(nib.load(glasserfilename).get_fdata())

# CAB-NP network definition
networkNames = ['VIS1','VIS2','SMN','CON','DAN','LAN','FPN','AUD','DMN','PMULTI','VMM','ORA']
networkdef = np.loadtxt(helpfiles_dir + 'cortex_parcel_network_assignments.txt')
# V1_M1_indices = [0,7,180,187]
# networkdef_revised = np.delete(networkdef,V1_M1_indices,axis=0)
# networkdef_revised = np.concatenate([networkdef_revised,[1.,3.]],axis=0)

# Parcels sorted by network definition
indsort = []
for netw in range(1,nNetwork+1):
    indsort.append(np.where(networkdef==netw)[0])
indsort = np.concatenate(indsort,axis=0)

# Network palette
networkpalette = ['royalblue','slateblue','paleturquoise','darkorchid','limegreen',
                  'lightseagreen','yellow','orchid','r','peru','orange','olivedrab']
networkpalette = np.asarray(networkpalette)
parcel_network_palette = []
for roi in range(nParcels): 
    parcel_network_palette.append(networkpalette[int(networkdef[roi]-1)])

roiIdx = []
for roi in range(1,nParcels+1):
    thisROIidx = np.where(glasser==roi)[0]
    roiIdx.append(thisROIidx)

# roiIdx_revised = list(np.delete(roiIdx_revised,[0,7,180,187],axis=0))
    
# for IOidx in range(IO_parcels.shape[0]):        
#     # first iteration is for V1, second for M1

#     roiIdxPooled = []
#     for roi in IO_parcels[IOidx]:
#         roiIdx = np.where(glasser==roi)[0]
#         roiIdxPooled.append(roiIdx)

#     roiIdxPooled = np.concatenate(roiIdxPooled,axis=0)
#     roiIdx_revised.append(roiIdxPooled) 

# def get_roiIdx_revised():
#     return roiIdx_revised
    
### [FOR LATER] Use parcelSize as covariate to check results' consistency ###

parcelSize = np.zeros(nParcels)
for roi in range(1,nParcels+1):
    parcelSize[roi-1] = int(np.sum(glasser==roi))
    
# V1_combined_size = parcelSize[0]+parcelSize[180]
# M1_combined_size = parcelSize[7]+parcelSize[187]

# parcelSize_revised = np.delete(parcelSize,V1_M1_indices,axis=0)
# parcelSize_revised = np.concatenate([parcelSize_revised,[V1_combined_size,M1_combined_size]],axis=0)

### Sensory, Association and Motor classification ###

# Following colors closest to 'magma' palette
color1, color2, color3 = colors.to_rgba('indigo'),colors.to_rgba('mediumvioletred'),colors.to_rgba('coral')

sensorynets = [1,2]
associationnets = [4,5,6,7,8,9,10,11,12]
motornets =[3]

tmp = {}
roi_id = np.zeros((nParcels,))
for netw in range(1,nNetwork+1):
    thisnetROIs = np.where(networkdef==netw)[0]
    for roi in thisnetROIs:
        if netw in sensorynets:
            tmp[roi] = color1 #'indigo'
            roi_id[roi] = 1

        elif netw in associationnets:
            tmp[roi] = color2 #'mediumvioletred'
            roi_id[roi] = 2

        elif netw in motornets:
            tmp[roi] = color3 #'coral'
            roi_id[roi] = 3

sensory_roi_id = np.where(roi_id == 1)[0]
association_roi_id = np.where(roi_id == 2)[0]
motor_roi_id = np.where(roi_id == 3)[0]
sensory_association_roi_id = np.where(np.logical_or(roi_id==1,roi_id==2))[0]
motor_association_roi_id = np.where(np.logical_or(roi_id==2,roi_id==3))[0]

def getBSwise_ROIid():
    return sensory_roi_id,association_roi_id,motor_roi_id,sensory_association_roi_id,motor_association_roi_id

roiColorsByNetwork = []
for roi in range(nParcels):
    roiColorsByNetwork.append(tmp[roi])
roiColorsByNetwork = np.array(roiColorsByNetwork)

bs_name = ['Sensory','Association','Motor']

### All networks classification ###

for netw in range(nNetwork):

    tmp = {}
    allnet_roi_id = np.zeros((nParcels,))
    for netw in range(1,nNetwork+1):
        thisnetROIs = np.where(networkdef==netw)[0]
        if (netw==1) or (netw==2):
            for roi in thisnetROIs:
                tmp[roi] = sns.color_palette()[0]
                allnet_roi_id[roi] = netw
        elif netw==5:
            for roi in thisnetROIs:
                tmp[roi] = sns.color_palette()[1]
                allnet_roi_id[roi] = netw
        elif netw==7:
            for roi in thisnetROIs:
                tmp[roi] = sns.color_palette()[2]
                allnet_roi_id[roi] = netw
        elif netw==9:
            for roi in thisnetROIs:
                tmp[roi] = sns.color_palette()[3]
                allnet_roi_id[roi] = netw
        elif netw==4:
            for roi in thisnetROIs:
                tmp[roi] = sns.color_palette()[4]
                allnet_roi_id[roi] = netw
        elif netw==3:
            for roi in thisnetROIs:
                tmp[roi] = sns.color_palette()[5]
                allnet_roi_id[roi] = netw
        else:
            for roi in thisnetROIs:
                tmp[roi] = colors.to_rgb('gray')
                allnet_roi_id[roi] = netw
        
    
allnet_roiColorsByNetwork = []
for roi in range(nParcels):
    allnet_roiColorsByNetwork.append(tmp[roi])
allnet_roiColorsByNetwork = np.array(allnet_roiColorsByNetwork)
    
### Task condition names ###

fullTaskCondLabelList = np.array(['Affective_pleasant', 'Affective_unpleasant',
                     'BiologicalMotion_happy', 'BiologicalMotion_sad',
                     'CPRO_doesnotfollowRule_buttonOne','CPRO_doesnotfollowRule_buttonTwo',
                     'CPRO_doesnotfollowRule_buttonThree','CPRO_doesnotfollowRule_buttonFour',
                     'CPRO_followsRule_buttonOne','CPRO_followsRule_buttonTwo',
                     'CPRO_followsRule_buttonThree','CPRO_followsRule_buttonFour',
                     'DigitJudgement_target_absent','DigitJudgement_target_present', 
                     'Emotional_happy','Emotional_sad', 
                     'GoNoGo_Go', 'GoNoGo_NoGo', 
                     'Ins_CPRO','Ins_GoNoGo', 'Ins_affective', 'Ins_arithmetic',
                     'Ins_checkerBoard', 'Ins_emotionProcess', 'Ins_emotional',
                     'Ins_intervalTiming', 'Ins_landscapeMovie', 'Ins_mentalRotation',
                     'Ins_motorImagery', 'Ins_nBack',  'Ins_natureMovie','Ins_prediction', 
                     'Ins_respAlt', 'Ins_romanceMovie','Ins_spatialMap','Ins_stroop', 
                     'Math_equation_correct', 'Math_equation_incorrect',
                     'Prediction_random', 'Prediction_valid', 'Prediction_violation',
                     'ScrambledMotion_fast', 'ScrambledMotion_slow',
                     'ToM_ToM_expected_false', 'ToM_ToM_expected_true',
                     'ToM_control_expected_false', 'ToM_control_expected_true',
                     'checkerBoard_CB','checkerBoard_images', 
                     'cong_color1', 'cong_color2', 'cong_color3','cong_color4', 
                     'incong_color1', 'incong_color2', 'incong_color3','incong_color4', 
                     'intervalTiming_long', 'intervalTiming_short',
                     'landscapeMovie', 
                     'mentalRotation_doesnotmatch_cond1','mentalRotation_doesnotmatch_cond2',
                     'mentalRotation_doesnotmatch_cond3', 'mentalRotation_match_cond1',
                     'mentalRotation_match_cond2', 'mentalRotation_match_cond3',
                     'motorSequence_complex', 'motorSequence_simple_digit1',
                     'motorSequence_simple_digit2', 'motorSequence_simple_digit3',
                     'motorSequence_simple_digit4', 'motor_imagery',
                     'nBackPic_notpossible2backfork.jpg',
                     'nBackPic_notpossible2backhydrant.jpg',
                     'nBackPic_notpossible2backlamp.jpg',
                     'nBackPic_notpossible2backplate.jpg',
                     'nBackPic_notpossible2backwhistle.jpg',
                     'nBackPic_notpossible2backzip.jpg',
                     'nBackPic_possible2backfork.jpg',
                     'nBackPic_possible2backhydrant.jpg',
                     'nBackPic_possible2backlamp.jpg',
                     'nBackPic_possible2backplate.jpg',
                     'nBackPic_possible2backwhistle.jpg',
                     'nBackPic_possible2backzip.jpg', 
                     'nBack_notpossible2backA','nBack_notpossible2backB', 
                     'nBack_notpossible2backC','nBack_possible2backA',
                     'nBack_possible2backB','nBack_possible2backC', 
                     'natureMovie',
                     'respAlt_setsizecond1_targPos1', 'respAlt_setsizecond1_targPos2',
                     'respAlt_setsizecond1_targPos3', 'respAlt_setsizecond1_targPos4',
                     'respAlt_setsizecond2_targPos1', 'respAlt_setsizecond2_targPos2',
                     'respAlt_setsizecond2_targPos3', 'respAlt_setsizecond2_targPos4',
                     'respAlt_setsizecond3_targPos1', 'respAlt_setsizecond3_targPos2',
                     'respAlt_setsizecond3_targPos3', 'respAlt_setsizecond3_targPos4',
                     'romanceMovie', 
                     'spatialMap_setsizecond1_targNum1','spatialMap_setsizecond1_targNum2',
                     'spatialMap_setsizecond1_targNum3','spatialMap_setsizecond1_targNum4',
                     'spatialMap_setsizecond2_targNum1','spatialMap_setsizecond2_targNum2',
                     'spatialMap_setsizecond2_targNum3','spatialMap_setsizecond2_targNum4',
                     'spatialMap_setsizecond3_targNum1','spatialMap_setsizecond3_targNum2',
                     'spatialMap_setsizecond3_targNum3','spatialMap_setsizecond3_targNum4',
                     'spatialNavigation','verbGeneration', 'videoActions', 'videoKnots',
                     'visualSearch_target_absent_setsize12',
                     'visualSearch_target_absent_setsize4',
                     'visualSearch_target_absent_setsize8',
                     'visualSearch_target_present_setsize12',
                     'visualSearch_target_present_setsize4',
                     'visualSearch_target_present_setsize8', 'wordRead',
                     'Ins_ToM','Ins_actionObservation','Ins_motorSequence','Ins_nBackPic', 
                     'Ins_spatialNavigation','Ins_verbGeneration','Ins_visualSearch'])

fullTaskList = np.array(['Affective','BiologicalMotion','CPRO','Emotional','Go-NoGo','Instruction',
                         'Arithmetic','Prediction','TheoryOfMind','ObjectViewing','Stroop',
                         'IntervalTiming','MovieWatching','MentalRotation','MotorSequence',
                         'MotorImagery','nBackPic','nBackLetters','ResponseAlternatives','spatialMap',
                         'SpatialNavigation','VerbGeneration','ActionObservation','VisualSearch'])

#################################################################

def generate_GTSimilarityRSM():
    
    file_path = intermFilesDir + 'readGTSimilarityRSM.pkl'
    
    if os.path.exists(file_path):
        
        with open(file_path, 'rb') as f:
            theoMotorRSM = pickle.load(f)
            theoStimRSM = pickle.load(f)
    
    else:
    
        # theoMotor RSM
        theoMotorRSM = np.zeros((nTaskCond,nTaskCond))
        for taskCondA in range(nTaskCond):
            for taskCondB in range(nTaskCond):

                if taskCondA>=taskCondB: 
                    continue

                if keyNames[taskCondA]==keyNames[taskCondB]:
                    theoMotorRSM[taskCondA,taskCondB] = 1

                if keyNames[taskCondA]=='all':
                    theoMotorRSM[taskCondA,:] = 1

        theoMotorRSM = theoMotorRSM + theoMotorRSM.T
        np.fill_diagonal(theoMotorRSM,1*np.ones(nTaskCond))

        # theostim RSM
        theoStimRSM = np.zeros((nTaskCond,nTaskCond))
        for taskCondA in range(nTaskCond):
            for taskCondB in range(nTaskCond):

                if taskCondA>=taskCondB: 
                    continue

                if stimType1[taskCondA] == stimType1[taskCondB]:
                    if stimType2[taskCondA] == stimType2[taskCondB]:
                        if stimType_sameflag[taskCondA]==stimType_sameflag[taskCondB]:
                            theoStimRSM[taskCondA,taskCondB] = 3
                        else:
                            theoStimRSM[taskCondA,taskCondB] = 2
                    else:
                        theoStimRSM[taskCondA,taskCondB] = 1
                else:
                    theoStimRSM[taskCondA,taskCondB] = 0

        theoStimRSM = theoStimRSM + theoStimRSM.T
        np.fill_diagonal(theoStimRSM,3*np.ones(nTaskCond))
        
        with open(intermFilesDir + 'readGTSimilarityRSM.pkl', 'wb') as f:
            pickle.dump(theoMotorRSM,f)
            pickle.dump(theoStimRSM,f)
    
    return theoMotorRSM, theoStimRSM

def displayMatrix(data,outname,level=None,labels=[]):
    
    # level: taskCond, task
    
    if level == 'taskCond':
        sns.set(font_scale=0.5)
        labels = fullTaskCondLabelList
    elif level == 'task':
        labels = fullTaskList

    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(data, xticklabels=labels, yticklabels=labels, linewidth=0, center=0)

    plt.tight_layout()
    sns.despine()
    plt.savefig(figoutdir + outname,transparent=True)

def readRSM():
    
    with open(intermFilesDir + 'allSub_allReg_taskCond_RSM.pkl', 'rb') as f:
        allSubRSM = pickle.load(f)

    #with open(intermFilesDir + 'allSub_taskCond_V1M1_RSM.pkl', 'rb') as f:
    #    allSub_V1M1_RSM = pickle.load(f)

    #allSubRSM_revised = np.delete(allSubRSM,V1_M1_indices,axis=1)
    #allSubRSM_revised = np.concatenate([allSubRSM_revised,allSub_V1M1_RSM],axis=1)

    meanSubRSM = np.mean(allSubRSM,axis=0)
    #meanSub_V1M1_RSM = np.mean(allSub_V1M1_RSM,axis=0)
    #meanSubRSM_revised = np.delete(meanSubRSM,V1_M1_indices,axis=0)
    #meanSubRSM_revised = np.concatenate([meanSubRSM_revised,meanSub_V1M1_RSM],axis=0)
    
    return allSubRSM,meanSubRSM

def getRepresentationalAlignment(meanSubRSM,thrVal=0.8):
    
    file_path = intermFilesDir + 'readRepresentationalAlignment.pkl'
    
    if os.path.exists(file_path):
        
        with open(file_path, 'rb') as f:
            gradients = pickle.load(f)
            RA = pickle.load(f)
    
    else:
    
        rsm_triu_ind = np.triu_indices(nTaskCond,k=0) # include diagonal as RSM is crossvalidated

        RA = np.zeros((nParcels,nParcels))
        for roiA in range(nParcels):
            for roiB in range(nParcels):

                if roiA>=roiB: continue

                vector1 = meanSubRSM[roiA,:,:][rsm_triu_ind].reshape(1,-1)
                vector2 = meanSubRSM[roiB,:,:][rsm_triu_ind].reshape(1,-1)

                RA[roiA,roiB] = skm.pairwise.cosine_similarity(vector1,vector2)[0][0]

        RA = RA + RA.T

        np.fill_diagonal(RA,np.ones(nParcels))

        # thresholding
        RAflat = np.ndarray.flatten(RA)
        RAflat_sorted = np.sort(RAflat)
        thrIdx = int(np.floor(thrVal*len(RAflat)))
        thr = RAflat_sorted[thrIdx]
        RAthr = RA.copy()
        RAthr[np.where(RAthr<thr)] = 0

        # Computing gradients
        ncomponents = 3
        pca = PCA(n_components=ncomponents,whiten=False)
        gradients = pca.fit_transform(RAthr)
        eigenvalues = pca.explained_variance_
        varianceExplained = pca.explained_variance_ratio_
        #print('varianceExplained:',varianceExplained)

        with open(intermFilesDir + 'readRepresentationalAlignment.pkl', 'wb') as f:
            pickle.dump(gradients,f)
            pickle.dump(RA,f)
    
    return gradients,RA

def getRAwithV1_M1(allSubRSM,niter=100):
    
    file_path = intermFilesDir + 'readRAwithV1_M1.pkl'
    
    if os.path.exists(file_path):
        
        with open(file_path, 'rb') as f:
            allSubRA_withV1 = pickle.load(f)
            allSubRA_withM1 = pickle.load(f)
            allSubRA_withV1_rand = pickle.load(f)
            allSubRA_withM1_rand = pickle.load(f)
    
    else:
    
        rsm_triu_ind = np.triu_indices(nTaskCond,k=0) # include diagonal as RSM is crossvalidated

        allSubRA_withV1 = []
        allSubRA_withM1 = []
        allSubRA_withV1_rand = []
        allSubRA_withM1_rand = []
        for subIdx in range(nSub):
            print(subIdx)
        
            avgV1RSM = np.mean([allSubRSM[subIdx][0],allSubRSM[subIdx][180]],axis=0)
            avgM1RSM = np.mean([allSubRSM[subIdx][7],allSubRSM[subIdx][187]],axis=0)

            RA_withV1 = np.zeros(nParcels)
            RA_withV1_rand = np.zeros((nParcels,niter))
            for roi in range(nParcels):
                vector1 = avgV1RSM[rsm_triu_ind].reshape(1,-1)
                vector2 = allSubRSM[subIdx][roi,:,:][rsm_triu_ind].reshape(1,-1)
                RA_withV1[roi] = skm.pairwise.cosine_similarity(vector1,vector2)[0][0]
                
                for i in range(niter):
                    randvector2 = np.random.permutation(allSubRSM[subIdx][roi,:,:][rsm_triu_ind]).reshape(1,-1)
                    RA_withV1_rand[roi,i] = skm.pairwise.cosine_similarity(vector1,randvector2)[0][0]

            RA_withM1 = np.zeros(nParcels)
            RA_withM1_rand = np.zeros((nParcels,niter))
            for roi in range(nParcels):
                vector1 = avgM1RSM[rsm_triu_ind].reshape(1,-1)
                vector2 = allSubRSM[subIdx][roi,:,:][rsm_triu_ind].reshape(1,-1)
                RA_withM1[roi] = skm.pairwise.cosine_similarity(vector1,vector2)[0][0]
                    
                for i in range(niter):
                    randvector2 = np.random.permutation(allSubRSM[subIdx][roi,:,:][rsm_triu_ind]).reshape(1,-1)
                    RA_withM1_rand[roi,i] = skm.pairwise.cosine_similarity(vector1,randvector2)[0][0]

            allSubRA_withV1.append(RA_withV1)
            allSubRA_withM1.append(RA_withM1)
            allSubRA_withV1_rand.append(RA_withV1_rand)
            allSubRA_withM1_rand.append(RA_withM1_rand)
            
        allSubRA_withV1 = np.array(allSubRA_withV1)
        allSubRA_withM1 = np.array(allSubRA_withM1)
        allSubRA_withV1_rand = np.array(allSubRA_withV1_rand)
        allSubRA_withM1_rand = np.array(allSubRA_withM1_rand)
            
        with open(intermFilesDir + 'readRAwithV1_M1.pkl', 'wb') as f:
            pickle.dump(allSubRA_withV1,f)
            pickle.dump(allSubRA_withM1,f)
            pickle.dump(allSubRA_withV1_rand,f)
            pickle.dump(allSubRA_withM1_rand,f)
    
    return allSubRA_withV1,allSubRA_withM1,allSubRA_withV1_rand,allSubRA_withM1_rand

def displaySortedValueAsBand(value,outname,cmap_name='Reds',invertXaxis=False):

    fig, ax = plt.subplots(figsize=(10, 2))
    
    sorted_value = np.sort(value.reshape(-1,1),axis=0)
    
    ax = sns.heatmap(sorted_value.T, xticklabels=[], yticklabels=[], linewidth=0, center=0, cmap=cmap_name)
    if invertXaxis:
        plt.gca().invert_xaxis()
    plt.tight_layout()
    sns.despine()
    plt.savefig(figoutdir + outname,transparent=True)

def customMeanScatterPlot(X,Y,xlabel,ylabel,xlim,ylim,outname,invert_xaxis=False,networklist=[2,5,7,9,4,3]):
    
    fig, ax = plt.subplots(figsize=(5,5))

    if invert_xaxis:
        plt.gca().invert_xaxis()
        
    # Compute means and overlay them
    
    df_system = {}
    df_system['Xvalue'] = []
    df_system['Yvalue'] = []
    df_system['group_by'] = []
    df_system['roi'] = []

    for netw in networklist:
        
        if netw == 2:  # VIS2 , include VIS1 with it
            thisNetROIs = np.where((networkdef==1) | (networkdef==2))[0]
        else:
            thisNetROIs = np.where(networkdef==netw)[0]
        
        for roi in thisNetROIs:
            df_system['Xvalue'].append(X[roi])
            df_system['Yvalue'].append(Y[roi])
            if netw == 2:  # VIS2 , include VIS1 with it
                df_system['group_by'].append('VIS')
            else:
                df_system['group_by'].append(networkNames[netw-1])
            df_system['roi'].append(roi)
            
    df_system = pd.DataFrame(df_system)
    
    tmp = df_system.groupby('group_by',sort=False).mean()
    sns.scatterplot(x="Xvalue",y="Yvalue",size=[40]*6,data=tmp,markers=["+"],hue="group_by",
                    palette=sns.color_palette("magma"),zorder=4)
    
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(figoutdir + outname,transparent=True)
    
def customScatterPlot(X,Y,RAaxis,xlabel,ylabel,outname,legLoc,invert_xaxis=False):

    if RAaxis == 'against_V1':
        Xselect = X[sensory_association_roi_id] 
        Yselect = Y[sensory_association_roi_id] 
        colorsSelect = allnet_roiColorsByNetwork[sensory_association_roi_id]
    elif RAaxis == 'against_M1':
        Xselect = X[motor_association_roi_id]
        Yselect = Y[motor_association_roi_id]
        colorsSelect = allnet_roiColorsByNetwork[motor_association_roi_id]
    elif RAaxis == 'full':
        Xselect = X
        Yselect = Y
        colorsSelect = allnet_roiColorsByNetwork
    
    fig, ax = plt.subplots(figsize=(5,5))
    sns.regplot(x=Xselect,y=Yselect,color='k',order=1,
                scatter_kws={'s':10,'color':colorsSelect,'alpha':0.4})
    plt.xlabel(xlabel,fontsize=12)
    plt.ylabel(ylabel,fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    if invert_xaxis:
        plt.gca().invert_xaxis()

    rho, p = stats.pearsonr(Xselect,Yselect)
    rho = round(rho,2)

    if legLoc == 'TL':
        legX,legY = 0.1,0.9     
    elif legLoc == 'TR':
        legX,legY = 0.7,0.9
    elif legLoc == 'BL':
        legX,legY = 0.1,0.15
    elif legLoc == 'BR':
        legX,legY = 0.7,0.15
    
    plt.annotate(r'$r$'+ ' = ' + str(rho),
                 xy=(legX,legY),fontsize=12,xycoords='axes fraction')
    plt.annotate(r'$p$'+ ' = ' + "{:.2e}".format(p),
                 xy=(legX,legY-0.05),fontsize=12,xycoords='axes fraction')
    
    plt.tight_layout()
    plt.savefig(figoutdir + outname,transparent=True)
    
def customTwoScatterPlot(X1,Y1,X2,Y2,leglabel1,leglabel2,RAaxis,xlabel,ylabel,outname,legLoc,invert_xaxis=False):

    if RAaxis == 'against_V1':
        X1select = X1[sensory_association_roi_id] 
        Y1select = Y1[sensory_association_roi_id] 
        X2select = X2[sensory_association_roi_id] 
        Y2select = Y2[sensory_association_roi_id] 
        colorsSelect = allnet_roiColorsByNetwork[sensory_association_roi_id]
    elif RAaxis == 'against_M1':
        X1select = X1[motor_association_roi_id]
        Y1select = Y1[motor_association_roi_id]
        X2select = X2[motor_association_roi_id]
        Y2select = Y2[motor_association_roi_id]
        colorsSelect = allnet_roiColorsByNetwork[motor_association_roi_id]
    elif RAaxis == 'full':
        X1select = X1
        Y1select = Y1
        X2select = X2
        Y2select = Y2
        colorsSelect = allnet_roiColorsByNetwork
    
    fig, ax = plt.subplots(figsize=(5,5))
    
    sns.regplot(x=X1select,y=Y1select,color='k',order=1,label=leglabel1,
                scatter_kws={'s':16,'color':colorsSelect,'alpha':0.4})
    ax = sns.regplot(x=X2select,y=Y2select,color='k',order=1,marker="x",label=leglabel2,
                scatter_kws={'s':16,'color':colorsSelect,'alpha':0.4})
    ax.lines[1].set_linestyle("--")
    plt.legend(fontsize=10)
    plt.xlabel(xlabel,fontsize=12)
    plt.ylabel(ylabel,fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    if invert_xaxis:
        plt.gca().invert_xaxis()
        


#     rho, p = stats.pearsonr(Xselect,Yselect)
#     rho = round(rho,2)

#     if legLoc == 'TL':
#         legX,legY = 0.1,0.9     
#     elif legLoc == 'TR':
#         legX,legY = 0.7,0.9
#     elif legLoc == 'BL':
#         legX,legY = 0.1,0.15
#     elif legLoc == 'BR':
#         legX,legY = 0.7,0.15
    
#     plt.annotate(r'$r$'+ ' = ' + str(rho),
#                  xy=(legX,legY),fontsize=10,xycoords='axes fraction')
#     plt.annotate(r'$p$'+ ' = ' + "{:.2e}".format(p),
#                  xy=(legX,legY-0.05),fontsize=10,xycoords='axes fraction')
    
    plt.tight_layout()
    plt.savefig(figoutdir + outname,transparent=True)
    
def customScatterPlotPerNetwork(X,Y,networklist,xlabel,ylabel,outname,legLoc,xlim,ylim,invert_xaxis=False):

    fig, ax = plt.subplots(figsize=(5*3+1,5*2))
    
    for netwIdx, netw in enumerate(networklist):
        if netw == 2:  # VIS2 , include VIS1 with it
            thisNetROIs = np.where((networkdef==1) | (networkdef==2))[0]
        else:
            thisNetROIs = np.where(networkdef==netw)[0]
        
        Xselect = X[thisNetROIs]
        Yselect = Y[thisNetROIs]
        colorsSelect = allnet_roiColorsByNetwork[thisNetROIs]
        
        plt.subplot(2,3, netwIdx+1)
        sns.regplot(x=Xselect,y=Yselect,color='k',order=1,
                    scatter_kws={'s':30,'color':sns.color_palette()[netwIdx],'alpha':0.4})
        if netw == 2:  # VIS2 , include VIS1 with it
            plt.title('VIS',fontsize=18)
        else:
            plt.title(networkNames[netw-1],fontsize=18)
            
        plt.xticks(fontsize=18)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.yticks(fontsize=18)
        if invert_xaxis:
            plt.gca().invert_xaxis()

        rho, p = stats.pearsonr(Xselect,Yselect)
        rho = round(rho,2)

        if legLoc[netwIdx] == 'TL':
            legX,legY = 0.1,0.9     
        elif legLoc[netwIdx] == 'TR':
            legX,legY = 0.6,0.9
        elif legLoc[netwIdx] == 'BL':
            legX,legY = 0.1,0.15
        elif legLoc[netwIdx] == 'BR':
            legX,legY = 0.6,0.15

        plt.annotate(r'$r$'+ ' = ' + str(rho),
                     xy=(legX,legY),fontsize=16,xycoords='axes fraction')
        if p<0.001:
            plt.annotate(r'$p$'+ ' = ' + "{:.2e}".format(p),
                     xy=(legX,legY-0.06),fontsize=16,xycoords='axes fraction')
        else:
            plt.annotate(r'$p$'+ ' = ' + str(round(p,3)),
                     xy=(legX,legY-0.06),fontsize=16,xycoords='axes fraction')
    
    fig.supxlabel(xlabel,fontsize=18)
    fig.supylabel(ylabel,fontsize=18,x=ax.get_position().x0 -.12)
    plt.tight_layout()
    plt.savefig(figoutdir + outname,transparent=True)

    
def customTwoScatterPlotPerNetwork(X1,Y1,X2,Y2,leglabel1,leglabel2,networklist,xlabel,ylabel,outname,legLoc,xlim,ylim,invert_xaxis=False):

    fig, ax1 = plt.subplots(figsize=(5*3+1,5*2))
    
    for netwIdx, netw in enumerate(networklist):
        if netw == 2:  # VIS2 , include VIS1 with it
            thisNetROIs = np.where((networkdef==1) | (networkdef==2))[0]
        else:
            thisNetROIs = np.where(networkdef==netw)[0]
        
        X1select = X1[thisNetROIs]
        Y1select = Y1[thisNetROIs]
        X2select = X2[thisNetROIs]
        Y2select = Y2[thisNetROIs]
        
        plt.subplot(2,3, netwIdx+1)
        
        sns.regplot(x=X1select,y=Y1select,color='k',order=1,label=leglabel1,
                scatter_kws={'s':30,'color':sns.color_palette()[netwIdx],'alpha':0.4})
        ax = sns.regplot(x=X2select,y=Y2select,color='k',order=1,marker="x",label=leglabel2,
                    scatter_kws={'s':30,'color':sns.color_palette()[netwIdx],'alpha':0.4})
        ax.lines[1].set_linestyle("--")
        plt.legend(fontsize=12)
        
        if netw == 2:  # VIS2 , include VIS1 with it
            plt.title('VIS',fontsize=18)
        else:
            plt.title(networkNames[netw-1],fontsize=18)
            
        plt.xticks(fontsize=18)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.yticks(fontsize=18)
        if invert_xaxis:
            plt.gca().invert_xaxis()

#         rho, p = stats.pearsonr(Xselect,Yselect)
#         rho = round(rho,2)

#         if legLoc[netwIdx] == 'TL':
#             legX,legY = 0.1,0.9     
#         elif legLoc[netwIdx] == 'TR':
#             legX,legY = 0.6,0.9
#         elif legLoc[netwIdx] == 'BL':
#             legX,legY = 0.1,0.15
#         elif legLoc[netwIdx] == 'BR':
#             legX,legY = 0.6,0.15

#         plt.annotate(r'$r$'+ ' = ' + str(rho),
#                      xy=(legX,legY),fontsize=16,xycoords='axes fraction')
#         if p<0.001:
#             plt.annotate(r'$p$'+ ' = ' + "{:.2e}".format(p),
#                      xy=(legX,legY-0.06),fontsize=16,xycoords='axes fraction')
#         else:
#             plt.annotate(r'$p$'+ ' = ' + str(round(p,3)),
#                      xy=(legX,legY-0.06),fontsize=16,xycoords='axes fraction')
    
    fig.supxlabel(xlabel,fontsize=18)
    fig.supylabel(ylabel,fontsize=18,x=ax1.get_position().x0 -.12)
    plt.tight_layout()
    plt.savefig(figoutdir + outname,transparent=True)

def customClusterPlot(mat,title,xlabel,ylabel,outname):
    
    sorted_mat = mat[indsort,:][:,indsort]

    ax = sns.clustermap(data=sorted_mat,
                        row_cluster=False,col_cluster=False,
                        row_colors=np.squeeze(np.asarray(parcel_network_palette)[indsort]),
                        col_colors=np.squeeze(np.asarray(parcel_network_palette)[indsort]),
                        xticklabels=False,yticklabels=False,
                        figsize=(4,4),center=0,cmap='seismic')
    ax.ax_heatmap.invert_yaxis()
    ax.ax_row_colors.invert_yaxis()
    ax_col_colors = ax.ax_col_colors
    box = ax_col_colors.get_position()
    box_heatmap = ax.ax_heatmap.get_position()
    ax_col_colors.set_position([box_heatmap.min[0], box_heatmap.y0-box.height, box.width, box.height])
    ax.ax_cbar.set_position([box_heatmap.max[0] + box.height, box_heatmap.y0,
                             box.height, box_heatmap.height])
    ax.cax.tick_params(labelsize=8)
    ax.fig.suptitle(title, x=0.62, y=box_heatmap.y1+.1,fontsize=10)
    ax_heatmap = ax.ax_heatmap
    ax_heatmap.set_xlabel(xlabel,fontsize=10)
    ax_heatmap.xaxis.set_label_coords(.5, -.1)
    ax_heatmap.set_ylabel(ylabel,fontsize=10)
    ax_heatmap.yaxis.set_label_coords(-.1, .5)
    ax_heatmap.yaxis.set_label_position('left')
    ax.savefig(figoutdir + outname,transparent=True)

def acrossSubBoxPlotPerNetwork(Y,ylabel,networklist=[2,5,7,9,4,3]):
    df_system = {}
    df_system['Yvalue'] = []
    df_system['group_by'] = []

    for netwIdx,netw in enumerate(networklist):
        for subIdx in range(nSub):
            df_system['Yvalue'].append(Y[subIdx,netwIdx])
            if netw == 2:  # VIS2 , include VIS1 with it
                df_system['group_by'].append('VIS')
            else:
                df_system['group_by'].append(networkNames[netw-1])


    df_system = pd.DataFrame(df_system)

    plt.figure(figsize=(5,5))
    ax = sns.boxplot(x="group_by",y="Yvalue",sym='',whis=[2.5,97.5],data=df_system,palette='magma') 
    sns.stripplot(x="group_by",y="Yvalue",alpha=0.4,s=6,data=df_system,palette='magma',zorder=0)
    tmp = df_system.groupby('group_by',sort=False).mean()
    sns.lineplot(x="group_by",y="Yvalue",data=tmp,markers=["o"],
                 color=sns.color_palette("Set2")[6],linewidth=2,zorder=4)
    sns.scatterplot(x="group_by",y="Yvalue",data=tmp,markers=["o"],
                    color=sns.color_palette("Set2")[6],zorder=4)

    plt.xlabel('Network',fontsize=18)
    plt.ylabel(None)
    plt.ylabel(ylabel, fontsize=12,labelpad=0.0)
    # plt.title('Dimensionality compression-\nthen-expansion', fontsize=10);
    #plt.xticks(fontsize=7,rotation=-15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    sns.despine()
    plt.savefig(figoutdir + 'acrossSub' + ylabel +'XNetworkBoxPlot.pdf',transparent=True)
    
def acrossSubTwoBoxPlotPerNetwork(Y,label1,label2,ylabel,networklist=[2,5,7,9,4,3]):
    df_system = {}
    df_system['Yvalue'] = []
    df_system['group_by'] = []
    df_system['nest_by'] = []

    for netwIdx,netw in enumerate(networklist):
        for subIdx in range(nSub):
            for nestIdx in range(2):
                df_system['Yvalue'].append(Y[subIdx,netwIdx,nestIdx])
                if netw == 2:  # VIS2 , include VIS1 with it
                    df_system['group_by'].append('VIS')
                else:
                    df_system['group_by'].append(networkNames[netw-1])
                
                if nestIdx==0:
                    df_system['nest_by'].append(label1)
                elif nestIdx==1:
                    df_system['nest_by'].append(label2)

    df_system = pd.DataFrame(df_system)

    plt.figure(figsize=(7,5))
    ax = sns.boxplot(x="group_by",y="Yvalue",hue="nest_by",sym='',whis=[2.5,97.5],data=df_system,palette='mako',dodge=True) 
    sns.stripplot(x="group_by",y="Yvalue",hue="nest_by",alpha=0.4,s=6,data=df_system,palette='mako',zorder=0,dodge=True)
    tmp = df_system.groupby(['group_by','nest_by'],sort=False).mean()
    sns.lineplot(x="group_by",y="Yvalue",hue="nest_by",data=tmp,markers=["o"],
                 palette=sns.color_palette(['wheat'], 2),linewidth=2,zorder=4)
    sns.scatterplot(x="group_by",y="Yvalue",hue="nest_by",data=tmp,markers=["o"],
                    palette =sns.color_palette(['wheat'], 2),zorder=4)
    
    handles, labels = ax.get_legend_handles_labels()
    l = plt.legend(handles[0:2], labels[0:2], fontsize=12)
    #l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.xlabel('Network',fontsize=18)
    plt.ylabel(None)
    plt.ylabel(ylabel, fontsize=12,labelpad=0.0)
    # plt.title('Dimensionality compression-\nthen-expansion', fontsize=10);
    #plt.xticks(fontsize=7,rotation=-15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    sns.despine()
    plt.savefig(figoutdir + 'acrossSub' + ylabel +'XTwoNetworkBoxPlot.pdf',transparent=True)

    # Comparing between 'nest_by' for each 'group_by'
    grouped = df_system.groupby('group_by')

    def compute_paired_ttest(group):
        group_a = group[group['nest_by'] == label1]['Yvalue']
        group_b = group[group['nest_by'] == label2]['Yvalue']
        t_stat, p_value = ttest_rel(group_a, group_b)
        return pd.Series({'t_stat': t_stat, 'p_value': p_value})

    ttest_results = grouped.apply(compute_paired_ttest).reset_index()

    print(ttest_results)
    
def brainSystemBoxPlot(Y,ylabel):
    df_system = {}
    df_system['Yvalue'] = []
    df_system['group_by'] = []

    this_nSub = Y.shape[0]
    
    for bsIdx in range(nBrainSystem):
        for subIdx in range(this_nSub):
            df_system['Yvalue'].append(Y[subIdx,bsIdx])
            df_system['group_by'].append(bs_name[bsIdx])


    df_system = pd.DataFrame(df_system)

    plt.figure(figsize=(5,5))
    ax = sns.boxplot(x="group_by",y="Yvalue",sym='',whis=[2.5,97.5],data=df_system,palette='magma') 
    sns.stripplot(x="group_by",y="Yvalue",alpha=0.4,s=3,data=df_system,palette='magma',zorder=0)
    tmp = df_system.groupby('group_by',sort=False).mean()
    sns.lineplot(x="group_by",y="Yvalue",data=tmp,markers=["o"],
                 color=sns.color_palette("Set2")[6],linewidth=2,zorder=4)
    sns.scatterplot(x="group_by",y="Yvalue",data=tmp,markers=["o"],
                    color=sns.color_palette("Set2")[6],zorder=4)

    plt.xticks(fontsize=10);
    plt.xlabel('Brain systems',fontsize=12);
    plt.ylabel(None)
    plt.ylabel(ylabel, fontsize=12,labelpad=0.0)
    # plt.title('Dimensionality compression-\nthen-expansion', fontsize=10);
    #plt.xticks(fontsize=7,rotation=-15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    sns.despine()
    plt.savefig(figoutdir + ylabel +'XBrainSystemBoxPlot.pdf',transparent=True)

    # Statistical testing
    df_assoc = df_system.loc[df_system.group_by=='Association']
    df_sensory = df_system.loc[df_system.group_by=='Sensory']
    df_motor = df_system.loc[df_system.group_by=='Motor']

    t, p = stats.ttest_ind(df_sensory.Yvalue.values,df_assoc.Yvalue.values)
    print('Sensory vs. Association: t =', t, '| p =', p, '| df =', 
          len(df_sensory.Yvalue.values) + len(df_assoc.Yvalue.values)-2)
    t, p = stats.ttest_ind(df_motor.Yvalue.values,df_assoc.Yvalue.values)
    print('Motor vs. Association: t =', t, '| p =', p, '| df =', 
          len(df_motor.Yvalue.values) + len(df_assoc.Yvalue.values)-2) 
    
def allNetsBoxPlot(Y,ylabel,networklist):
    df_system = {}
    df_system['Yvalue'] = []
    df_system['group_by'] = []
    df_system['roi'] = []

    for netw in networklist:
        
        if netw == 2:  # VIS2 , include VIS1 with it
            thisNetROIs = np.where((networkdef==1) | (networkdef==2))[0]
        else:
            thisNetROIs = np.where(networkdef==netw)[0]
        
        for roi in thisNetROIs:
            df_system['Yvalue'].append(Y[roi])
            if netw == 2:  # VIS2 , include VIS1 with it
                df_system['group_by'].append('VIS')
            else:
                df_system['group_by'].append(networkNames[netw-1])
            df_system['roi'].append(roi)
            

    df_system = pd.DataFrame(df_system)

    plt.figure(figsize=(5,5))
    ax = sns.boxplot(x="group_by",y="Yvalue",sym='',whis=[2.5,97.5],data=df_system,palette='magma') 
    sns.stripplot(x="group_by",y="Yvalue",alpha=0.4,s=6,data=df_system,palette='magma',zorder=0)
    tmp = df_system.groupby('group_by',sort=False).mean()
    sns.lineplot(x="group_by",y="Yvalue",data=tmp,markers=["o"],
                 color=sns.color_palette("Set2")[6],linewidth=2,zorder=4)
    sns.scatterplot(x="group_by",y="Yvalue",data=tmp,markers=["o"],
                    color=sns.color_palette("Set2")[6],zorder=4)

    plt.xlabel('Network',fontsize=18);
    plt.ylabel(None)
    plt.ylabel(ylabel, fontsize=12,labelpad=0.0)
    #plt.ylim([0,1])
    # plt.title('Dimensionality compression-\nthen-expansion', fontsize=10);
    #plt.xticks(fontsize=7,rotation=-15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    sns.despine()
    #plt.savefig(figoutdir + ylabel +'XAllNetworksBoxPlot.pdf',transparent=True)
    
    return df_system
    
def allNetsBoxPlot_two(Y1,Y2,ylabel,networklist,label1,label2):
    
    def returnNetworkAverages(metric): 
        
        metricAllNets = np.zeros((nSub,len(networklist)))
        for subIdx in range(nSub):
            for netwIdx,netw in enumerate(networklist):
                if netw == 2:
                    metricAllNets[subIdx,netwIdx] = np.mean(metric[subIdx,np.where((networkdef==1) |(networkdef==2))[0]])
                else:
                    metricAllNets[subIdx,netwIdx] = np.mean(metric[subIdx,np.where(networkdef==netw)[0]])
                
        return metricAllNets
                
    metricAllNets_Y1 = returnNetworkAverages(Y1)
    metricAllNets_Y2 = returnNetworkAverages(Y2)
    
    df = {}
    df['Yvalue'] = []
    df['group_by'] = []
    df['nest_by'] = []

    for netwIdx,netw in enumerate(networklist):
        
        if netw == 2:  # VIS2 , include VIS1 with it
            thisNetROIs = np.where((networkdef==1) | (networkdef==2))[0]
        else:
            thisNetROIs = np.where(networkdef==netw)[0]
        
        for subIdx in range(nSub):
            for nest_byIdx in range(2):
                
                if netw == 2:  # VIS2 , include VIS1 with it
                    df['group_by'].append('VIS')
                else:
                    df['group_by'].append(networkNames[netw-1])
                
                if nest_byIdx == 0:
                    df['Yvalue'].append(metricAllNets_Y1[subIdx,netwIdx])
                    df['nest_by'].append(label1)
                elif nest_byIdx == 1:
                    df['Yvalue'].append(metricAllNets_Y2[subIdx,netwIdx])
                    df['nest_by'].append(label2)

    df = pd.DataFrame(df)

    plt.figure(figsize=(6,5))
    
    ax = sns.boxplot(x="group_by",y="Yvalue",hue="nest_by",sym='',whis=[2.5,97.5],data=df,palette='magma',dodge=True) 
    sns.stripplot(x="group_by",y="Yvalue",hue="nest_by",alpha=0.4,s=6,data=df,palette='magma',zorder=0,dodge=True)
    tmp = df.groupby(['group_by','nest_by'],sort=False).mean()
    sns.lineplot(x="group_by",y="Yvalue",hue="nest_by",data=tmp,markers=["o"],
                 palette=sns.color_palette(['wheat'], 2),linewidth=2,zorder=4)
    sns.scatterplot(x="group_by",y="Yvalue",hue="nest_by",data=tmp,markers=["o"],
                    palette=sns.color_palette(['wheat'], 2),zorder=4)
    
    handles, labels = ax.get_legend_handles_labels()
    l = plt.legend(handles[0:2], labels[0:2], fontsize=12)
    
    
    plt.xlabel('Network',fontsize=18);
    plt.ylabel(None)
    plt.ylabel(ylabel, fontsize=12,labelpad=0.0)
    #plt.ylim([0,1])
    # plt.title('Dimensionality compression-\nthen-expansion', fontsize=10);
    #plt.xticks(fontsize=7,rotation=-15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    sns.despine()
    plt.savefig(figoutdir + ylabel +'XAllNetworksBoxPlotTwo.pdf',transparent=True)
    
def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='ward')
    
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx],idx,linkage

def plotDendrogram(data,ylimVal,outname):
    
    corr_array, idx, linkage = cluster_corr(data)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    dendrogrm = sch.dendrogram(linkage)
    plt.ylabel('Euclidean distance')
    plt.xticks(rotation=90, fontsize=5)  # Rotate x-axis tick labels if needed
    plt.gca().set_xticklabels(fullTaskCondLabelList[idx])
    plt.ylim(0,ylimVal)
    plt.savefig(figoutdir + outname,bbox_inches='tight')
    plt.show()
    
    return corr_array,idx

def runMetric(metricName,data,networklist=[2,5,7,9,4,3],niter=100):
    
    file_path = intermFilesDir + 'read' + metricName + '.pkl'
    
    if os.path.exists(file_path):
        
        with open(file_path, 'rb') as f:
            meanSubMetric = pickle.load(f)
            meanSubMetric_rand = pickle.load(f)
            metricAllNets = pickle.load(f)
            metricAllNets_rand = pickle.load(f)
            metric = pickle.load(f)
    
    else:
    
        metric = np.zeros((nSub,nParcels))
        metric_rand = np.zeros((nSub,nParcels,niter))
        metricAllNets = np.zeros((nSub,len(networklist)))
        metricAllNets_rand = np.zeros((nSub,len(networklist),niter))
        
        for subIdx in range(nSub):
            print(subIdx)
            for roi in range(nParcels):

                RSM = data[subIdx][roi]
                
                if metricName == 'Dimensionality':
                    metric[subIdx,roi] = getDimensionality(RSM)
                    for i in range(niter):
                        RSM_shuffled = RSM[:,np.random.permutation(nTaskCond)][np.random.permutation(nTaskCond),:]
                        metric_rand[subIdx,roi,i] = getDimensionality(RSM_shuffled)
                elif metricName == 'TaskCondDecodability':
                    metric[subIdx,roi] = getTaskCondDecodability(RSM)
                    for i in range(niter):
                        RSM_shuffled = RSM[:,np.random.permutation(nTaskCond)][np.random.permutation(nTaskCond),:]
                        metric_rand[subIdx,roi,i] = getTaskCondDecodability(RSM_shuffled)
                elif metricName == 'OverallSimilarity':
                    metric[subIdx,roi] = getOverallSimilarity(RSM)
                elif metricName == 'Reliability':
                    metric[subIdx,roi] = getAcrossSessionReliability(RSM)
                elif metricName == 'BetweenTaskDistance':
                    metric[subIdx,roi] = getBetweenTaskDistance(RSM)
                    for i in range(niter):
                        RSM_shuffled = RSM[:,np.random.permutation(nTaskCond)][np.random.permutation(nTaskCond),:]
                        metric_rand[subIdx,roi,i] = getBetweenTaskDistance(RSM_shuffled)
                elif metricName == 'WithinTaskDistance':
                    metric[subIdx,roi] = getWithinTaskDistance(RSM)
                    for i in range(niter):
                        RSM_shuffled = RSM[:,np.random.permutation(nTaskCond)][np.random.permutation(nTaskCond),:]
                        metric_rand[subIdx,roi,i] = getWithinTaskDistance(RSM_shuffled)
                elif metricName == 'NoOfClusters':
                    nClust,clustSize = getClusterMetrics(RSM)
                    metric[subIdx,roi] = nClust
                elif metricName == 'ClusterSize':
                    nClust,clustSize = getClusterMetrics(RSM)
                    metric[subIdx,roi] = clustSize
                elif metricName == 'StimGTSimilarity':
                    motorGT_RSM, stimGT_RSM = generate_GTSimilarityRSM()
                    metric[subIdx,roi] = getSimilarityWithGT_RSM(RSM,theoRSM=stimGT_RSM)
                    for i in range(niter):
                        RSM_shuffled = RSM[:,np.random.permutation(nTaskCond)][np.random.permutation(nTaskCond),:]
                        metric_rand[subIdx,roi,i] = getSimilarityWithGT_RSM(RSM_shuffled,theoRSM=stimGT_RSM)
                elif metricName == 'MotorGTSimilarity':
                    motorGT_RSM, stimGT_RSM = generate_GTSimilarityRSM()
                    metric[subIdx,roi] = getSimilarityWithGT_RSM(RSM,theoRSM=motorGT_RSM)
                    for i in range(niter):
                        RSM_shuffled = RSM[:,np.random.permutation(nTaskCond)][np.random.permutation(nTaskCond),:]
                        metric_rand[subIdx,roi,i] = getSimilarityWithGT_RSM(RSM_shuffled,theoRSM=motorGT_RSM)

            for netwIdx,netw in enumerate(networklist):
                if netw == 2:
                    metricAllNets[subIdx,netwIdx] = np.mean(metric[subIdx,np.where((networkdef==1) |(networkdef==2))[0]])
                    for i in range(niter):
                        metricAllNets_rand[subIdx,netwIdx,i] = np.mean(metric_rand[subIdx,np.where((networkdef==1) |(networkdef==2))[0],i])
                else:
                    metricAllNets[subIdx,netwIdx] = np.mean(metric[subIdx,np.where(networkdef==netw)[0]])
                    for i in range(niter):
                        metricAllNets_rand[subIdx,netwIdx,i] = np.mean(metric_rand[subIdx,np.where(networkdef==netw)[0],i])

        meanSubMetric = np.mean(metric,axis=0)
        meanSubMetric_rand = np.mean(metric_rand,axis=0)

        with open(intermFilesDir + 'read' + metricName + '.pkl', 'wb') as f:
                pickle.dump(meanSubMetric,f)
                pickle.dump(meanSubMetric_rand,f)
                pickle.dump(metricAllNets,f)
                pickle.dump(metricAllNets_rand,f)
                pickle.dump(metric,f)
    
    return meanSubMetric,meanSubMetric_rand,metricAllNets,metricAllNets_rand,metric

def runSubspaceMetric(metricName,betas,subspace_dim = 'reduced',thrVarExp=0.9):
    allSubtaskPairAngle = np.zeros((nSub,nParcels))
    allROItaskPairAngleBS = np.zeros((nSub,nBrainSystem))
    allROI_meanTask_nCompBS = np.zeros((nSub,nBrainSystem))

    allSub_meanTask_nComp = []
    for subIdx in range(nSub):
        
        print('subIdx:',subIdx)
        
        if metricName == 'TaskSubspaceAngle':
            taskSubspaceInfo = getTaskSubspace(betas[subIdx],subspace_dim,thrVarExp)

            allROITaskVecDiff = taskSubspaceInfo[0]
            allROITaskVecDiff_reduced = taskSubspaceInfo[1]
            allROITaskVecDiff_comp = taskSubspaceInfo[2]
            allROITaskVecDiff_varExp = taskSubspaceInfo[3]
            allTaskName = taskSubspaceInfo[4]
            allROI_allTask_nComp = taskSubspaceInfo[5] #size: nROI x nTask

            allROItaskPairAngle = getSubspaceAngles(subspace_components = allROITaskVecDiff_comp)

            # Get upper-triangular values
            avg_taskSpaceAngles = np.zeros(nParcels)
            task_triu_ind = np.triu_indices(nTaskSubspace,k=1) # no diagonal
            for roi in range(nParcels):
                allSubtaskPairAngle[subIdx,roi] = np.mean(allROItaskPairAngle[roi][task_triu_ind])
            
            allROI_meanTask_nComp = np.mean(allROI_allTask_nComp,axis=1)
            allSub_meanTask_nComp.append(allROI_meanTask_nComp)

        for bs in range(1,nBrainSystem+1):
            allROItaskPairAngleBS[subIdx,bs-1] = np.mean(allSubtaskPairAngle[subIdx,roi_id==bs])
            
            
            allROI_meanTask_nCompBS[subIdx,bs-1] = np.mean(allROI_meanTask_nComp[roi_id==bs])

    meanSubtaskPairAngle = np.mean(allSubtaskPairAngle,axis=0)
    
    allSub_meanTask_nComp = np.array(allSub_meanTask_nComp)
    meanSub_meanTask_nComp = np.mean(allSub_meanTask_nComp,axis=0)
    
    
    return meanSubtaskPairAngle,meanSub_meanTask_nComp,allROItaskPairAngleBS,allROI_meanTask_nCompBS

def getRSMavg_bySimilar_Dissimilar_categories(data,suffix,theoRSM,networklist=[2,5,7,9,4,3]):
    
    file_path = intermFilesDir + 'readRSMavg_bySimilar_Dissimilar_categories'+suffix+'.pkl'
    
    if os.path.exists(file_path):
        
        with open(file_path, 'rb') as f:
            meanSubRSMavg = pickle.load(f)
            RSMavgAllNets = pickle.load(f)
            RSMavg_bySimilar_Dissimilar_categories = pickle.load(f)
    
    else:
    
        RSMavg_bySimilar_Dissimilar_categories = np.zeros((nSub,nParcels,2))
        RSMavgAllNets = np.zeros((nSub,len(networklist),2))

        for subIdx in range(nSub):
            for roi in range(nParcels):
                RSM = data[subIdx][roi]

                simCatIdx = np.where(theoRSM != 0)
                dissimCatIdx = np.where(theoRSM == 0)

                RSMavg_bySimilar_Dissimilar_categories[subIdx,roi,0] = np.mean(RSM[simCatIdx])
                RSMavg_bySimilar_Dissimilar_categories[subIdx,roi,1] = np.mean(RSM[dissimCatIdx])

            for netwIdx,netw in enumerate(networklist):
                if netw == 2:
                    RSMavgAllNets[subIdx,netwIdx,0] = np.mean(RSMavg_bySimilar_Dissimilar_categories[subIdx,np.where((networkdef==1) |(networkdef==2))[0],0])
                    RSMavgAllNets[subIdx,netwIdx,1] = np.mean(RSMavg_bySimilar_Dissimilar_categories[subIdx,np.where((networkdef==1) |(networkdef==2))[0],1])
                else:
                    RSMavgAllNets[subIdx,netwIdx,0] = np.mean(RSMavg_bySimilar_Dissimilar_categories[subIdx,np.where(networkdef==netw)[0],0])
                    RSMavgAllNets[subIdx,netwIdx,1] = np.mean(RSMavg_bySimilar_Dissimilar_categories[subIdx,np.where(networkdef==netw)[0],1])

        meanSubRSMavg = np.mean(RSMavg_bySimilar_Dissimilar_categories,axis=0)

        with open(intermFilesDir + 'readRSMavg_bySimilar_Dissimilar_categories'+suffix+'.pkl', 'wb') as f:
                    pickle.dump(meanSubRSMavg,f)
                    pickle.dump(RSMavgAllNets,f)
                    pickle.dump(RSMavg_bySimilar_Dissimilar_categories,f)
    
    return meanSubRSMavg,RSMavgAllNets,RSMavg_bySimilar_Dissimilar_categories

def getDimensionality(data):
    """
    data needs to be a square, symmetric matrix
    """
    corrmat = data
    eigenvalues, eigenvectors = np.linalg.eig(corrmat)
    dimensionality_nom = 0
    dimensionality_denom = 0
    for eig in eigenvalues:
        dimensionality_nom += np.real(eig)
        dimensionality_denom += np.real(eig)**2

    dimensionality = dimensionality_nom**2/dimensionality_denom

    return dimensionality

def getEffNumPCs_inTaskSpace(RSMdata,thrExpVar):
    """
    data needs to be a square, symmetric matrix. Isn't symmetric. So taking the real parts of eigen values and eigen vectors
    """
    eigenvalues, eigenvectors = np.linalg.eig(RSMdata)
    eigenvalues = np.real(eigenvalues)
    eigenvaluesRatio = eigenvalues/np.sum(eigenvalues)
    eff_nComp = np.min(np.where(np.cumsum(eigenvaluesRatio) > thrExpVar)[0])+1

    eigenvectors = np.real(eigenvectors)
    eff_eigenvec = eigenvectors[:,:eff_nComp]

    return eff_nComp,eff_eigenvec

def getTaskCondDecodability(data):

    taskCondDecodabilityFlag = np.zeros(nTaskCond)
    for taskCondIdx in range(nTaskCond): 
        thisTaskCondDia = data[taskCondIdx,taskCondIdx]
        thisTaskCondCol = data[:,taskCondIdx]
        thisTaskCondCol = np.delete(thisTaskCondCol,[taskCondIdx])
        thisTaskCondColMax = np.max(thisTaskCondCol)
        thisTaskCondRow = data[taskCondIdx,:]
        thisTaskCondRow = np.delete(thisTaskCondRow,[taskCondIdx])
        thisTaskCondRowMax = np.max(thisTaskCondRow)

        if (thisTaskCondDia > thisTaskCondColMax) and (thisTaskCondDia > thisTaskCondRowMax):
            taskCondDecodabilityFlag[taskCondIdx] = 1

    taskCondDecodability = np.mean(taskCondDecodabilityFlag)

    return taskCondDecodability

def getOverallSimilarity(data):
    
    overallSimilarity = np.mean(data) 
    return overallSimilarity

def getAcrossSessionReliability(data):
    
    reliability = np.mean(np.diagonal(data)) 
    return reliability

def getClusterMetrics(data,thrDist=10):
    
    corr_array, idx, linkage = cluster_corr(data)
    clustParcellation = sch.cut_tree(linkage,height=thrDist)
    unique,counts = np.unique(clustParcellation,return_counts=True)
    nClust,clustSize = np.max(unique)+1,np.mean(counts) 
    #cut_tree produces cluster labels from 0 to n-1 clusters
    
    return nClust,clustSize

def getSimilarityWithGT_RSM(data,theoRSM):
    
    rsm_triu_ind = np.triu_indices(nTaskCond,k=1) # no principal diagonal
    theoRSMvec = theoRSM[rsm_triu_ind].reshape(1,-1)
    roi_RSMvec = data[rsm_triu_ind].reshape(1,-1)
    similarityWithGT_RSM = skm.pairwise.cosine_similarity(theoRSMvec,roi_RSMvec)[0][0]
    
    return similarityWithGT_RSM

def getTaskSubspace(betas,subspace_dim,thrVarExp=[]):
    
    # Find (nCond-1) subspace of each task in all ROIs for given set of beta vectors
    # betas shape: (nTaskCond x nVertices)
    # subspace_dim: full: (nCond-1) subspace, 
    #               reduced: subspace smaller than (nCond-1) dimensions, where varExp crosses threshold
    #               or integer value of ncomponents desired
    
    allROITaskVecDiff = []
    allROITaskVecDiff_reduced = []
    allROITaskVecDiff_comp = []
    allROITaskVecDiff_varExp = []
    allROI_allTask_nComp = np.zeros((nParcels,nTaskSubspace))
    for roi in range(nParcels):

        roiIdx = roiIdx[roi]

        allTaskVecDiff = []
        allTaskVecDiff_reduced = []
        allTaskVecDiff_comp = []
        allTaskVecDiff_varExp = []
        allTaskNameIdx = []
        taskCount = 0
        for taskIndex_ID in range(1,nTask+1):
            thisTaskIdx = np.where(taskIndex==taskIndex_ID)[0]
            if len(thisTaskIdx) > 1:
                taskVec = betas[thisTaskIdx,:][:,roiIdx]
                taskVecDiff = taskVec - np.mean(taskVec,axis=0)

                if subspace_dim=='full':
                    ncomponents = len(thisTaskIdx)-1 # nCond-1 dimensions
                elif subspace_dim=='reduced':
                    ncomponents = getReduced_nComp(taskVecDiff,thrVarExp,thisTaskIdx)
                else:
                    ncomponents = subspace_dim
                
                allROI_allTask_nComp[roi,taskCount] = ncomponents
                taskCount = taskCount+1
                
                pca = PCA(n_components=ncomponents,whiten=False)
                reduced = pca.fit_transform(taskVecDiff)
                comp = pca.components_
                varExp = pca.explained_variance_ratio_

                allTaskVecDiff.append(taskVecDiff)
                allTaskVecDiff_reduced.append(reduced)
                allTaskVecDiff_comp.append(comp)
                allTaskVecDiff_varExp.append(np.max(np.cumsum(varExp)))
                allTaskNameIdx.append(taskIndex_ID)
        allROITaskVecDiff.append(allTaskVecDiff)
        allROITaskVecDiff_reduced.append(allTaskVecDiff_reduced)
        allROITaskVecDiff_comp.append(allTaskVecDiff_comp)
        allROITaskVecDiff_varExp.append(allTaskVecDiff_varExp)

    allROITaskVecDiff_varExp = np.array(allROITaskVecDiff_varExp)
    allTaskName = fullTaskList[np.array(allTaskNameIdx)-1]

    return allROITaskVecDiff,allROITaskVecDiff_reduced,allROITaskVecDiff_comp,allROITaskVecDiff_varExp,allTaskName,allROI_allTask_nComp
  
def getReduced_nComp(taskVecDiff,thrVarExp,thisTaskIdx):
    
    ncomponents = len(thisTaskIdx)-1 # max value is nComp-1
    pca = PCA(n_components=ncomponents,whiten=False)
    reduced = pca.fit_transform(taskVecDiff)
    varExp = pca.explained_variance_ratio_
    effective_nComp = np.where(np.cumsum(varExp) > thrVarExp)[0] + 1
    #print('effective_nComp:', effective_nComp[0])
    
    return effective_nComp[0]

def getEffNumPCs(betas,thrVarExp):
    
    effective_nComp = np.zeros(nParcels)
    taskCondDimensionality = np.zeros(nParcels)
    for roi in range(nParcels):

        roiIdx = roiIdx_revised[roi]
        thisROIbetas = betas[roiIdx,:]
        thisROIbetasDiff = thisROIbetas - np.mean(thisROIbetas,axis=0)
        
        ncomponents = min(betas.shape[1]-1,len(roiIdx)) # max value is nTaskCond-1 if betas is alltaskCond
        pca = PCA(n_components=ncomponents,whiten=False)
        reduced = pca.fit_transform(thisROIbetasDiff)
        varExp = pca.explained_variance_ratio_
        effective_nComp[roi] = np.min(np.where(np.cumsum(varExp) > thrVarExp)[0]) + 1
        
        TaskCondCovarianceMat = np.dot(thisROIbetasDiff.T,thisROIbetasDiff)
        taskCondDimensionality[roi] = getDimensionality(TaskCondCovarianceMat)
        
    return effective_nComp,taskCondDimensionality

def displayTaskValuesAlongRAaxes(allROI_allTask_variable,RA_withV1,RA_withM1,outname):
    
    RA_withV1_argsort = np.argsort(RA_withV1)
    RA_withM1_argsort = np.argsort(RA_withM1)
    
    meanROI_allTask_variable = np.mean(allROI_allTask_variable,axis=0)
    allTask_argsort = np.argsort(meanROI_allTask_variable)
    
    Sen_Asso_Slice = allROI_allTask_variable[sensory_association_roi_id,:][:,allTask_argsort]
    Mot_Asso_Slice = allROI_allTask_variable[motor_association_roi_id,:][:,allTask_argsort]
    
    variableAlong_V1 = Sen_Asso_Slice[RA_withV1_argsort].T
    variableAlong_M1 = Sen_Asso_Slice[RA_withM1_argsort].T
    labels = fullTaskList[allTask_argsort]
    
    # Along_RAwithV1
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(variableAlong_V1, xticklabels=[], yticklabels=labels, linewidth=0, center=0)
    plt.gca().invert_xaxis()
    
    plt.tight_layout()
    sns.despine()
    plt.savefig(figoutdir + 'RAwithV1_' + outname,transparent=True)
    
    # Along_RAwithM1
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(variableAlong_M1, xticklabels=[], yticklabels=labels, linewidth=0, center=0)
    
    plt.tight_layout()
    sns.despine()
    plt.savefig(figoutdir + 'RAwithM1_' + outname,transparent=True)
    


def getProjectionsInSubspace(betas,subspace_dim,taskIndex_ID,thrVarExp=[]):
    
    allROI_projectedTaskVec = []
    for roi in range(nParcels):

        roiIdx = roiIdx[roi]
        thisTaskIdx = np.where(taskIndex==taskIndex_ID)[0]
        taskVec = betas[thisTaskIdx,:][:,roiIdx]
        taskVecDiff = taskVec - np.mean(taskVec,axis=0)

        if subspace_dim=='full':
            ncomponents = len(thisTaskIdx)-1 # nCond-1 dimensions
        elif subspace_dim=='reduced':
            ncomponents = getReduced_nComp(taskVecDiff,thrVarExp,thisTaskIdx)
        else:
            ncomponents = subspace_dim

        pca = PCA(n_components=ncomponents,whiten=False)
        reduced = pca.fit_transform(taskVecDiff)
        projectedTaskVec = pca.transform(taskVec)
        allROI_projectedTaskVec.append(projectedTaskVec)
    
    return allROI_projectedTaskVec
        
    
def getSubspaceAngles(subspace_components):
    
    allROItaskPairAngle = []
    for roi in range(nParcels):

        taskPairAngle = np.zeros((nTaskSubspace,nTaskSubspace))
        for taskAidx in range(nTaskSubspace):
            for taskBidx in range(nTaskSubspace):
                taskPairAngle[taskAidx,taskBidx] = np.rad2deg(subspace_angles(
                    subspace_components[roi][taskAidx].T, 
                    subspace_components[roi][taskBidx].T))[0]
        allROItaskPairAngle.append(taskPairAngle)
    allROItaskPairAngle = np.array(allROItaskPairAngle)
    
    return allROItaskPairAngle
    
def getPairwiseDistancesFromRSM(RSMsubmatrix):
    
    pairwiseDistance = []
    for condA in range(len(RSMsubmatrix)-1):
        for condB in range(condA+1,len(RSMsubmatrix)):
            pairwiseDistance.append(RSMsubmatrix[condA,condA] 
                                   + RSMsubmatrix[condB,condB]
                                   - RSMsubmatrix[condA,condB]
                                   - RSMsubmatrix[condB,condA])
    meanPairwiseDistance = np.mean(pairwiseDistance)
    
    return meanPairwiseDistance

def getBetweenTaskDistance(RSM):
    
    BtwTaskDistance = []
    for taskIndexIdxA in range(len(selectTaskIndexList)-1):
        for taskIndexIdxB in range(taskIndexIdxA,len(selectTaskIndexList)):

            taskA_CondIdx = np.where(taskIndex == selectTaskIndexList[taskIndexIdxA])[0]
            taskB_CondIdx = np.where(taskIndex == selectTaskIndexList[taskIndexIdxB])[0]

            taskAA_RSMsubmatrix = RSM[taskA_CondIdx,:][:,taskA_CondIdx]
            taskBB_RSMsubmatrix = RSM[taskB_CondIdx,:][:,taskB_CondIdx]
            taskAB_RSMsubmatrix = RSM[taskA_CondIdx,:][:,taskB_CondIdx]
            taskBA_RSMsubmatrix = RSM[taskB_CondIdx,:][:,taskA_CondIdx]

            taskLevelAveragedRSM_submatrix = np.array([[np.mean(taskAA_RSMsubmatrix),np.mean(taskAB_RSMsubmatrix)],
                                               [np.mean(taskBA_RSMsubmatrix),np.mean(taskBB_RSMsubmatrix)]])

            taskA_taskB_distance = getPairwiseDistancesFromRSM(taskLevelAveragedRSM_submatrix)
            BtwTaskDistance.append(taskA_taskB_distance)
        
    meanBtwTaskDistance = np.mean(BtwTaskDistance)
            
    return meanBtwTaskDistance

def getWithinTaskDistance(RSM):
    
    BtwTaskCondDistance = []
    for taskIndexIdx in range(len(selectTaskIndexList)):
        task_CondIdx = np.where(taskIndex == selectTaskIndexList[taskIndexIdx])[0]
        task_RSMsubmatrix = RSM[task_CondIdx,:][:,task_CondIdx]
        task_btwCondDistance = getPairwiseDistancesFromRSM(task_RSMsubmatrix)
        BtwTaskCondDistance.append(task_btwCondDistance)
        
    meanBtwTaskCondDistance = np.mean(BtwTaskCondDistance)
            
    return meanBtwTaskCondDistance

def getMeanParVecCosTheta(parVecIdx,allROI_projectedTaskVec):
    
    nCombination = parVecIdx.shape[0]
    nVec = parVecIdx.shape[1]


    MeanParVecCosTheta = np.zeros((nParcels,nCombination))
    for combination in range(nCombination):

        thisCombParallelVec = []
        for vec in range(nVec):
            thisROI_thisCombParallelVec = []
            for roi in range(nParcels):
                thisROI_projectedTaskVec = allROI_projectedTaskVec[roi]
                thisROI_thisCombParallelVec.append(thisROI_projectedTaskVec[parVecIdx[combination,vec,0],:] 
                                       - thisROI_projectedTaskVec[parVecIdx[combination,vec,1],:])
            thisCombParallelVec.append(thisROI_thisCombParallelVec)

        # Measure mean cos theta
        for roi in range(nParcels):

            parVecCosTheta = []
            for parVecAIdx in range(nVec-1):
                for parVecBIdx in range(parVecAIdx,nVec):

                    parVecA = np.array(thisCombParallelVec[parVecAIdx][roi]).reshape(1,-1)
                    parVecB = np.array(thisCombParallelVec[parVecBIdx][roi]).reshape(1,-1)

                    parVecCosTheta.append(skm.pairwise.cosine_similarity(parVecA,parVecB)[0][0])

            MeanParVecCosTheta[roi,combination] = np.mean(parVecCosTheta)

    AvgCombMeanParVecCosTheta = np.mean(MeanParVecCosTheta,axis=1)
    
    return AvgCombMeanParVecCosTheta

# Across-sub statistical test function

def acrossSub_tTestStats(X,Y,nSub):

    '''
    X,Y of size (nSub,nParcels)
    '''

    allSub_r_full = np.zeros(nSub)
    allSub_r_onlysensory = np.zeros(nSub)
    allSub_r_onlyassociation = np.zeros(nSub)
    allSub_r_onlymotor = np.zeros(nSub)

    for subIdx in range(nSub):

        r_full,p1 = stats.pearsonr(X[subIdx,:],Y[subIdx,:])
        r_sensory,p2 = stats.pearsonr(X[subIdx,sensory_roi_id],Y[subIdx,sensory_roi_id])
        r_association,p3 = stats.pearsonr(X[subIdx,association_roi_id],Y[subIdx,association_roi_id])
        r_motor,p4 = stats.pearsonr(X[subIdx,motor_roi_id],Y[subIdx,motor_roi_id])

        allSub_r_full[subIdx] = r_full
        allSub_r_onlysensory[subIdx] = r_sensory
        allSub_r_onlyassociation[subIdx] = r_association
        allSub_r_onlymotor[subIdx] = r_motor

    # Statistical test
    allSub_r_fisher = np.arctanh(allSub_r_full)
    allSub_r_onlysensory_fisher = np.arctanh(allSub_r_onlysensory)
    allSub_r_onlyassociation_fisher = np.arctanh(allSub_r_onlyassociation)
    allSub_r_onlymotor_fisher = np.arctanh(allSub_r_onlymotor)

    t_full,p_full = stats.ttest_1samp(allSub_r_fisher, popmean=0, axis=0)
    t_sensory,p_sensory = stats.ttest_1samp(allSub_r_onlysensory_fisher, popmean=0, axis=0)
    t_association,p_association = stats.ttest_1samp(allSub_r_onlyassociation_fisher, popmean=0, axis=0)
    t_motor,p_motor = stats.ttest_1samp(allSub_r_onlymotor_fisher, popmean=0, axis=0)

    ## Comparison between systems
    
    # repeated measures ANOVA
    
    # Create a DataFrame in long format
    data = {
        'subject': np.tile(np.arange(1, nSub + 1), 3),
        'condition': np.repeat(['sensory', 'association', 'motor'], nSub),
        'value': np.concatenate([allSub_r_onlysensory_fisher, allSub_r_onlyassociation_fisher, allSub_r_onlymotor_fisher])
    }

    df = pd.DataFrame(data)

    # Perform repeated measures ANOVA
    aovrm = AnovaRM(df, 'value', 'subject', within=['condition'])
    res = aovrm.fit()

    print(res)
    
    t_SenAsso,p_SenAsso = stats.ttest_rel(allSub_r_onlysensory_fisher,allSub_r_onlyassociation_fisher, axis=0)
    t_AssoMot,p_AssoMot = stats.ttest_rel(allSub_r_onlyassociation_fisher,allSub_r_onlymotor_fisher, axis=0)
    t_SenMot,p_SenMot = stats.ttest_rel(allSub_r_onlysensory_fisher,allSub_r_onlymotor_fisher, axis=0)

    print('t_full,p_full: ',t_full,p_full)
    print('t_sensory,p_sensory: ',t_sensory,p_sensory)
    print('t_association,p_association: ',t_association,p_association)
    print('t_motor,p_motor: ',t_motor,p_motor)
    
    print('t_SenAsso,p_SenAsso: ',t_SenAsso,p_SenAsso, 'Corr. p=',3*p_SenAsso)
    print('t_AssoMot,p_AssoMot: ',t_AssoMot,p_AssoMot, 'Corr. p=',3*p_AssoMot)
    print('t_SenMot,p_SenMot: ',t_SenMot,p_SenMot, 'Corr. p=',3*p_SenMot)
    