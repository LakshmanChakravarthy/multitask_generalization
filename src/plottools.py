import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy.cluster.hierarchy as sch
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from matplotlib import colors
import nibabel as nib
from scipy.stats import ttest_rel
import wbplot

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

sensory_roi_id = np.where(roi_id==1)[0]
association_roi_id = np.where(roi_id==2)[0]
motor_roi_id = np.where(roi_id==3)[0]
            
sensory_association_roi_id = np.where(np.logical_or(roi_id==1,roi_id==2))[0]
motor_association_roi_id = np.where(np.logical_or(roi_id==2,roi_id==3))[0]

def getBSwise_ROIid():
    return sensory_association_roi_id,motor_association_roi_id

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
        for roi in thisnetROIs:
            tmp[roi] = colors.to_rgb(networkpalette[netw-1])
            allnet_roi_id[roi] = netw
        
    
allnet_roiColorsByNetwork = []
for roi in range(nParcels):
    allnet_roiColorsByNetwork.append(tmp[roi])
allnet_roiColorsByNetwork = np.array(allnet_roiColorsByNetwork)
    
def get_roiColorsByNetwork():
    return roiColorsByNetwork
    
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
    
def displaySortedValueAsBand(value,outname,invertXaxis=False):

    fig, ax = plt.subplots(figsize=(10, 2))
    
    sorted_value = np.sort(value.reshape(-1,1),axis=0)
    
    ax = sns.heatmap(sorted_value.T, xticklabels=[], yticklabels=[], linewidth=0, center=0)
    if invertXaxis:
        plt.gca().invert_xaxis()
    plt.tight_layout()
    sns.despine()
    plt.savefig(figoutdir + outname,transparent=True)

def customMeanScatterPlot(X,Y,xlabel,ylabel,xlim,ylim,outname,invert_xaxis=False,networklist=[2,5,9,7,4,3]):
    
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
    
def customScatterPlot(X,Y,RAaxis,xlabel,ylabel,outname,legLoc,xmin,xmax,ymin,ymax,custom_select=[],invert_xaxis=False,deg=1,xtickbins=3,ytickbins=3,plotwidth=5.75,showstat=False,regplot=True):

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
    elif RAaxis == 'full_BS':
        Xselect = X
        Yselect = Y
        colorsSelect = roiColorsByNetwork
    elif RAaxis == 'full_BS_custom':
        Xselect = X[custom_select]
        Yselect = Y[custom_select]
        colorsSelect = roiColorsByNetwork[custom_select]
    elif RAaxis == 'onlysensory':
        Xselect = X[sensory_roi_id]
        Yselect = Y[sensory_roi_id]
        colorsSelect = roiColorsByNetwork[sensory_roi_id]
    elif RAaxis == 'onlyassociation':
        Xselect = X[association_roi_id]
        Yselect = Y[association_roi_id]
        colorsSelect = roiColorsByNetwork[association_roi_id]
    elif RAaxis == 'onlymotor':
        Xselect = X[motor_roi_id]
        Yselect = Y[motor_roi_id]
        colorsSelect = roiColorsByNetwork[motor_roi_id]
    
    
    fig, ax = plt.subplots(figsize=(plotwidth,5))
    
    if regplot:
        sns.regplot(x=Xselect,y=Yselect,color='k',order=deg,
                    scatter_kws={'s':70,'color':colorsSelect,'alpha':0.6},line_kws={'linewidth':4})
    else:
        sns.scatterplot(x=Xselect, y=Yselect, color=colorsSelect, alpha=0.6, s=70, edgecolor='none')
    
    plt.xlabel(xlabel,fontsize=28)
    plt.ylabel(ylabel,fontsize=28)
    plt.locator_params(axis='y', nbins=ytickbins)
    plt.locator_params(axis='x', nbins=xtickbins)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    ax.xaxis.set_tick_params(width=4)
    ax.yaxis.set_tick_params(width=4)
    if invert_xaxis:
        plt.gca().invert_xaxis()

    if showstat==True: 
        
        rho, p = stats.pearsonr(Xselect,Yselect)
        rho = round(rho,2)

        if legLoc == 'TL':
            legX,legY = 0.075,0.85     
        elif legLoc == 'TR':
            legX,legY = 0.55,0.85
        elif legLoc == 'BL':
            legX,legY = 0.075,0.1
        elif legLoc == 'BR':
            legX,legY = 0.55,0.1

        plt.annotate(r'$r$'+ ' = ' + str(rho),
                     xy=(legX,legY),fontsize=28,xycoords='axes fraction')
        # plt.annotate(r'$p$'+ ' = ' + "{:.2e}".format(p),
        #              xy=(legX,legY-0.05),fontsize=28,xycoords='axes fraction')
    
    sns.despine()
    plt.tight_layout()
    
    if outname:
        plt.savefig(outname,transparent=True)
    
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
    
def customScatterPlotPerNetwork(X,Y,Y_rand,networklist,xlabel,ylabel,outname,legLoc,xlim,ylim,invert_xaxis=False,niter=100,nullComp=True):
    
    fig, ax = plt.subplots(figsize=(5*3+1,5*2))
    
    allnet_rho = []
    allnet_rho_rand = [] 
    
    for netwIdx, netw in enumerate(networklist):
        if netw == 2:  # VIS2 , include VIS1 with it
            thisNetROIs = np.where((networkdef==1) | (networkdef==2))[0]
        else:
            thisNetROIs = np.where(networkdef==netw)[0]
        
        Xselect = X[thisNetROIs]
        Yselect = Y[thisNetROIs]
        if nullComp:
            Yselect_rand = Y_rand[thisNetROIs,:]
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
        adj_p = p*len(networklist)
        allnet_rho.append(rho)
        
        if nullComp:
            rho_rand = []
            for i in range(niter):
                rho2, p2 = stats.pearsonr(Xselect,Yselect_rand[:,i]) 
                rho_rand.append(rho2)
            allnet_rho_rand.append(rho_rand)

        if legLoc[netwIdx] == 'TL':
            legX,legY = 0.1,0.9     
        elif legLoc[netwIdx] == 'TR':
            legX,legY = 0.6,0.9
        elif legLoc[netwIdx] == 'BL':
            legX,legY = 0.1,0.15
        elif legLoc[netwIdx] == 'BR':
            legX,legY = 0.6,0.15

        plt.annotate(r'$r$'+ ' = ' + str(rho),
                     xy=(legX,legY),fontsize=24,xycoords='axes fraction')
        if nullComp==False:
            if p<0.001:
                plt.annotate(r'$adj. p$'+ ' = ' + "{:.2e}".format(adj_p),
                         xy=(legX,legY-0.06),fontsize=16,xycoords='axes fraction')
            else:
                plt.annotate(r'$adj. p$'+ ' = ' + str(round(adj_p,3)),
                         xy=(legX,legY-0.06),fontsize=16,xycoords='axes fraction')
    
    fig.supxlabel(xlabel,fontsize=18)
    fig.supylabel(ylabel,fontsize=18,x=ax.get_position().x0 -.12)
    plt.tight_layout()
    plt.savefig(figoutdir + outname,transparent=True)
    
    if nullComp:
        df = {}
        df['Yvalue'] = []
        df['Yvalue_rand'] = []
        df['group_by'] = []

        for netwIdx,netw in enumerate(networklist):
            for i in range(niter):
                df['Yvalue'].append(np.arctanh(allnet_rho[netwIdx]))
                df['Yvalue_rand'].append(np.arctanh(allnet_rho_rand[netwIdx][i]))
                if netw == 2:  # VIS2 , include VIS1 with it
                    df['group_by'].append('VIS')
                else:
                    df['group_by'].append(networkNames[netw-1])

        df = pd.DataFrame(df)
        plt.figure(figsize=(5,5))

        ax1 = sns.boxplot(x="group_by",y="Yvalue_rand",sym='',whis=[0.417,99.583],data=df,color=sns.color_palette()[7])

        tmp = df.groupby("group_by",sort=False).mean()
        sns.scatterplot(x="group_by",y="Yvalue",data=tmp,s=75,markers=["o"],
                        color=sns.color_palette("Set2")[6],zorder=4)  
        plt.xlabel('Network',fontsize=18)
        plt.ylabel('Correlation Coefficient(Z-transformed)')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        sns.despine()
        plt.savefig(figoutdir + 'nullCompStats' + outname,transparent=True)

    
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
    ax.cax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.cax.tick_params(labelsize=24)
    ax.fig.suptitle(title, x=0.62, y=box_heatmap.y1+.1,fontsize=10)
    ax_heatmap = ax.ax_heatmap
    ax_heatmap.set_xlabel(xlabel,fontsize=10)
    ax_heatmap.xaxis.set_label_coords(.5, -.1)
    ax_heatmap.set_ylabel(ylabel,fontsize=10)
    ax_heatmap.yaxis.set_label_coords(-.1, .5)
    ax_heatmap.yaxis.set_label_position('left')
    ax.savefig(figoutdir + outname,transparent=True)

def acrossSubBoxPlotPerNetwork(Y,Y_rand,ylabel,networklist,nullComp=True):
    df_system = {}
    df_system['Yvalue'] = []
    df_system['group_by'] = []
    
    networklistnames = ['VIS','DAN','DMN','FPN','CON','SMN']
    
    for netwIdx,netw in enumerate(networklist):
        for subIdx in range(nSub):
            df_system['Yvalue'].append(Y[subIdx,netwIdx])
            if netw == 2:  # VIS2 , include VIS1 with it
                df_system['group_by'].append('VIS')
            else:
                df_system['group_by'].append(networkNames[netw-1])
                
    df_system = pd.DataFrame(df_system)
    
    if nullComp:
        
        df = {}
        df['Yvalue'] = []
        df['group_by'] = []

        niter = Y_rand.shape[1]
                
        for netwIdx,netw in enumerate(networklist):
            for i in range(niter):
                df['Yvalue'].append(Y_rand[netwIdx,i])
                if netw == 2:  # VIS2 , include VIS1 with it
                    df['group_by'].append('VIS')
                else:
                    df['group_by'].append(networkNames[netw-1])

        df = pd.DataFrame(df)

    if nullComp:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
    else:
        plt.figure(figsize=(5,5))
    
    
    ax = sns.boxplot(x="group_by",y="Yvalue",sym='',whis=[0.417,99.583],data=df_system,palette='magma',order=networklistnames) 
    sns.stripplot(x="group_by",y="Yvalue",alpha=0.4,s=6,data=df_system,palette='magma',zorder=0,order=networklistnames)
    tmp = df_system.groupby('group_by',sort=False).mean()
    sns.lineplot(x="group_by",y="Yvalue",data=tmp,markers=["o"],
                 color=sns.color_palette("Set2")[6],linewidth=2,zorder=4)
    sns.scatterplot(x="group_by",y="Yvalue",data=tmp,markers=["o"],
                    color=sns.color_palette("Set2")[6],zorder=4)

    plt.xlabel('Network',fontsize=18)
    plt.ylabel(ylabel, fontsize=12,labelpad=0.0)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    if nullComp:
        plt.subplot(1,2,2)
        ax1 = sns.boxplot(x="group_by",y="Yvalue",sym='',whis=[2.5,97.5],data=df,color=sns.color_palette()[7],order=networklistnames)
        sns.scatterplot(x="group_by",y="Yvalue",data=tmp,s=75,markers=["o"],
                        color=sns.color_palette("Set2")[6],zorder=4)  
        plt.xlabel('Network',fontsize=18)
        plt.ylabel(None)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
    
    plt.tight_layout()
    sns.despine()
    plt.savefig(figoutdir + 'acrossSub' + ylabel +'XNetworkBoxPlot.pdf',transparent=True)
    
    tval = np.ones((len(networklist),len(networklist)))* np.nan
    pval = np.ones((len(networklist),len(networklist)))* np.nan
    for netwA in range(len(networklist)-1):
        for netwB in range(netwA+1,len(networklist)):
            t, p = stats.ttest_rel(Y[:,netwA],Y[:,netwB])
            t = round(t,3)
            adj_p = p*15
            if adj_p>0.001:
                adj_p = round(adj_p,3)
            else:
                adj_p = "{:.2e}".format(adj_p)
            
            tval[netwA,netwB] = t
            pval[netwA,netwB] = adj_p
            
    print('tval',tval)
    print('adj. pval',pval)
    
def acrossSubTwoBoxPlotPerNetwork(Y,label1,label2,ylabel,networklist):
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
    
def brainSystemBoxPlot(Y,ylabel,null_data,ytickbins=3,plotwidth=5,plotheight=5,rotval=20,nSub=24,use_null=True):
    df_system = {}
    df_system['Yvalue'] = []
    df_system['group_by'] = []
    
    for bsIdx in range(nBrainSystem):
        for subIdx in range(nSub):
            df_system['Yvalue'].append(Y[subIdx,bsIdx])
            df_system['group_by'].append(bs_name[bsIdx])

    df_system = pd.DataFrame(df_system)
    
    if use_null:
    
        # Flatten the null data
        df_null = {}
        df_null['Yvalue'] = []
        df_null['group_by'] = []

        for bsIdx in range(nBrainSystem):
            for permIdx in range(1000):  # 1000 permutations
                df_null['Yvalue'].append(null_data[bsIdx, permIdx])
                df_null['group_by'].append(bs_name[bsIdx])

        df_null = pd.DataFrame(df_null)

    plt.figure(figsize=(plotwidth,plotheight))
    ax = sns.boxplot(x="group_by",y="Yvalue",sym='',whis=[2.5,97.5],data=df_system,palette='magma',linewidth=4) 
    sns.stripplot(x="group_by",y="Yvalue",alpha=0.4,s=12,data=df_system,palette='magma',zorder=0)
    
    if use_null:
    
        # Add grey dots for the null permutations
        sns.stripplot(x="group_by", y="Yvalue", alpha=0.2, s=8, data=df_null, color='grey', zorder=-1)
    
    tmp = df_system.groupby('group_by',sort=False).mean()
    sns.lineplot(x="group_by",y="Yvalue",data=tmp,markers=["o"],
                 color=sns.color_palette("Set2")[6],linewidth=4,zorder=4)
    sns.scatterplot(x="group_by",y="Yvalue",data=tmp,markers=["o"],
                    color=sns.color_palette("Set2")[6],zorder=4,s=100)
    plt.locator_params(axis='y', nbins=ytickbins)
    plt.xlabel('Brain systems',fontsize=28);
    plt.ylabel(None)
    plt.ylabel(ylabel, fontsize=28,labelpad=0.0)
    # plt.title('Dimensionality compression-\nthen-expansion', fontsize=10);
    plt.xticks(fontsize=24,rotation=rotval)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    sns.despine()
    plt.savefig(figoutdir + ylabel +'XBrainSystemBoxPlot.pdf',transparent=True)

    # Statistical testing
    df_assoc = df_system.loc[df_system.group_by=='Association']
    df_sensory = df_system.loc[df_system.group_by=='Sensory']
    df_motor = df_system.loc[df_system.group_by=='Motor']

    t, p = stats.ttest_rel(df_sensory.Yvalue.values,df_assoc.Yvalue.values)
    print('Sensory vs. Association: t =', t, '| p =', p, '| df =', nSub-1)
    t, p = stats.ttest_rel(df_motor.Yvalue.values,df_assoc.Yvalue.values)
    print('Motor vs. Association: t =', t, '| p =', p, '| df =', nSub-1) 
    
def brainSystemBoxPlot_regionwise(Y,ylabel,ytickbins=3,plotwidth=5,plotheight=5,rotval=20,nSub=24):
    
    '''
    Y shape: (nParcels,)
    being grouped into three groups based on brain system (sensory/association/motor)
    '''
    
    df_system = {}
    df_system['Yvalue'] = []
    df_system['group_by'] = []

    for roiIdx in range(len(sensory_roi_id)):
        df_system['Yvalue'].append(Y[sensory_roi_id[roiIdx]])
        df_system['group_by'].append(bs_name[0])
        
    for roiIdx in range(len(association_roi_id)):
        df_system['Yvalue'].append(Y[association_roi_id[roiIdx]])
        df_system['group_by'].append(bs_name[1])
        
    for roiIdx in range(len(motor_roi_id)):
        df_system['Yvalue'].append(Y[motor_roi_id[roiIdx]])
        df_system['group_by'].append(bs_name[2])
        
        
    df_system = pd.DataFrame(df_system)

    plt.figure(figsize=(plotwidth,plotheight))
    ax = sns.boxplot(x="group_by",y="Yvalue",sym='',whis=[2.5,97.5],data=df_system,palette='magma',linewidth=4) 
    sns.stripplot(x="group_by",y="Yvalue",alpha=0.4,s=12,data=df_system,palette='magma',zorder=0)
    tmp = df_system.groupby('group_by',sort=False).mean()
    sns.lineplot(x="group_by",y="Yvalue",data=tmp,markers=["o"],
                 color=sns.color_palette("Set2")[6],linewidth=4,zorder=4)
    sns.scatterplot(x="group_by",y="Yvalue",data=tmp,markers=["o"],
                    color=sns.color_palette("Set2")[6],zorder=4,s=100)
    plt.locator_params(axis='y', nbins=ytickbins)
    plt.xlabel('Brain systems',fontsize=28);
    plt.ylabel(None)
    plt.ylabel(ylabel, fontsize=28,labelpad=0.0)
    # plt.title('Dimensionality compression-\nthen-expansion', fontsize=10);
    plt.xticks(fontsize=24,rotation=rotval)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    sns.despine()
    plt.savefig(figoutdir + ylabel +'XBrainSystemBoxPlot_Regionwise.pdf',transparent=True)

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
    
    metricAllNets_Y = returnNetworkAverages(Y,networklist)
    
    df = {}
    df['Yvalue'] = []
    df['group_by'] = []

    for netwIdx,netw in enumerate(networklist):
        
        if netw == 2:  # VIS2 , include VIS1 with it
            thisNetROIs = np.where((networkdef==1) | (networkdef==2))[0]
        else:
            thisNetROIs = np.where(networkdef==netw)[0]
        
        for subIdx in range(nSub):
                
            df['Yvalue'].append(metricAllNets_Y[subIdx,netwIdx])    
                
            if netw == 2:  # VIS2 , include VIS1 with it
                df['group_by'].append('VIS')
            else:
                df['group_by'].append(networkNames[netw-1])
            

    df = pd.DataFrame(df)

    plt.figure(figsize=(5,5))
    ax = sns.boxplot(x="group_by",y="Yvalue",sym='',whis=[2.5,97.5],data=df,palette='magma') 
    sns.stripplot(x="group_by",y="Yvalue",alpha=0.4,s=6,data=df,palette='magma',zorder=0)
    tmp = df.groupby('group_by',sort=False).mean()
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
    plt.savefig(figoutdir + ylabel +'XAllNetworksBoxPlot.pdf',transparent=True)
    
    tval = np.ones((len(networklist),len(networklist)))* np.nan
    pval = np.ones((len(networklist),len(networklist)))* np.nan
    for netwA in range(len(networklist)-1):
        for netwB in range(netwA+1,len(networklist)):
            t, p = stats.ttest_rel(metricAllNets_Y[:,netwA],metricAllNets_Y[:,netwB])
            t = round(t,3)
            adj_p = p*15
            if adj_p>0.001:
                adj_p = round(adj_p,3)
            else:
                adj_p = "{:.2e}".format(adj_p)
            
            tval[netwA,netwB] = t
            pval[netwA,netwB] = adj_p
            
    print('tval',tval)
    print('adj. pval',pval)
            

def returnNetworkAverages(metric,networklist): 

    metricAllNets = np.zeros((nSub,len(networklist)))
    for subIdx in range(nSub):
        for netwIdx,netw in enumerate(networklist):
            # if netw == 2:
            #     metricAllNets[subIdx,netwIdx] = np.mean(metric[subIdx,np.where((networkdef==1) |(networkdef==2))[0]])
            # else:
            metricAllNets[subIdx,netwIdx] = np.mean(metric[subIdx,np.where(networkdef==netw)[0]])

    return metricAllNets
  
def returnBSAverages(metric,nSub=24): 

    metricBS = np.zeros((nSub,3))
    for subIdx in range(nSub):
        for bsIdx in range(3):
            metricBS[subIdx,bsIdx] = np.mean(metric[subIdx,np.where(roi_id==bsIdx+1)[0]])

    return metricBS
    
def allNetsBoxPlot_two(Y1,Y2,ylabel,networklist,label1,label2):
                
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

def parcelPlot(inputdata,file_outname,title):

    # Flip hemispheres for CABNP
    inputdata_flipped = flipHemispheres(inputdata,scalar=True)

    #Set to all reds if no negative values
    if min(inputdata) >= 0:
        colormap='Reds'
    else:
        colormap='seismic'
    wbplot.pscalar(
            file_out=figoutdir+file_outname,
            pscalars=inputdata_flipped,
            cmap=colormap,
            transparent=True)
    this_img = img.imread(figoutdir+file_outname)
    plt.figure()
    plt.axis('off')
    plt.title(title)
    im = plt.imshow(this_img)
    
# def allNetwBoxPlot(inputdata,file_out,title,avgAsso_file_out=None,avgAsso=False):

#     # across CABNP networks

#     # Flip hemispheres for CABNP
#     inputdata_flipped = flipHemispheres(inputdata)

#     all_net_corr = []
#     for netw in range(1,13):
#         net_corr = np.mean(inputdata_flipped[:,networkdef==netw],axis=1)
#         all_net_corr.append(net_corr)

#     # Plot

#     fig = plt.figure(figsize =(10, 7))

#     # Creating axes instance
#     ax = fig.add_axes([0, 0, 1, 1])

#     # Creating plot
#     bp = ax.boxplot(list(all_net_corr), labels=networkNames)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.title(title)
#     plt.savefig(file_out,bbox_inches='tight')
#     # show plot
#     plt.show()
    
#     if avgAsso:
        
#         # averaging networks except VIS1 and SMN

#         all_net_corr = np.array(all_net_corr)
#         #print(all_net_corr.shape)

#         asso_net_avg = np.mean(all_net_corr[[1,3,4,5,6,7,8,9,10,11],:],axis=0)
#         #print(asso_net_avg.shape)
#         new_net_corr = np.array([all_net_corr[0,:],asso_net_avg,all_net_corr[2,:]])

#         # Plot

#         fig = plt.figure(figsize =(10, 7))

#         # Creating axes instance
#         ax = fig.add_axes([0, 0, 1, 1])

#         # Creating plot
#         bp = ax.boxplot(list(new_net_corr), labels=['VIS1','ASSO','SMN'])
#         plt.xticks(fontsize=16)
#         plt.yticks(fontsize=16)
#         plt.title(title)
#         plt.savefig(avgAsso_file_out,bbox_inches='tight')

#         # show plot
#         plt.show()
    
def flipHemispheres(inputdata,scalar=False):
    
    #flip hemispheres, since CAB-NP is ordered left-to-right, while wbplot uses right-to-left

    if scalar:
        inputdata_flipped=np.zeros(np.shape(inputdata))
        inputdata_flipped[0:180]=inputdata[180:360]
        inputdata_flipped[180:360]=inputdata[0:180]
    else:
        inputdata_flipped=np.zeros(np.shape(inputdata))
        inputdata_flipped[:,0:180]=inputdata[:,180:360]
        inputdata_flipped[:,180:360]=inputdata[:,0:180]
    
    return inputdata_flipped