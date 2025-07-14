# plot_utils

import numpy as np
import pandas as pd
import scipy.stats as stats

import wbplot
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.colors as clrs
from matplotlib.colors import ListedColormap
from matplotlib import colorbar
import seaborn as sns
import matplotlib.patches as mpatches


# Paths
projdir = '/home/ln275/f_mc1689_1/multitask_generalization/'
figoutdir = projdir + 'docs/figures/working/vertexwiseFC/'
helpfiles_dir = projdir + 'docs/experimentfiles/'


# Params
nParcels = 360
nNetwork = 12
networkdef = np.loadtxt(helpfiles_dir + 'cortex_parcel_network_assignments.txt')

# Following colors closest to 'magma' palette
color1, color2, color3 = clrs.to_rgba('indigo'),clrs.to_rgba('mediumvioletred'),clrs.to_rgba('coral')

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

roiColorsByNetwork = []
for roi in range(nParcels):
    roiColorsByNetwork.append(tmp[roi])
roiColorsByNetwork = np.array(roiColorsByNetwork)

# # Set the style for the plot
# plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_context("talk", font_scale=1.5)  # Increased font scale from 1.5 to 2.0 and context to "talk"


def get_brain_plot(inputdata,file_out,title,colormap=None,ignore_unvalid=False):

    #flip hemispheres, since CAB-NP is ordered left-to-right, while wbplot uses right-to-left
    inputdata_flipped=np.zeros(np.shape(inputdata))
    inputdata_flipped[0:180]=inputdata[180:360]
    inputdata_flipped[180:360]=inputdata[0:180]

    if ignore_unvalid:
        mask_valid_regions = inputdata != -9999
        min_inputdata = np.min(inputdata[mask_valid_regions])
    else:
        min_inputdata = np.min(inputdata)
    
  
    # Set to all reds if no negative values
    if min_inputdata >= 0:
        if colormap == None:
            colormap='Reds'
        else:
            colormap=colormap
        vmin, vmax = min_inputdata, np.max(inputdata)
    else:
        colormap='seismic'
        limit = np.max([np.abs(min_inputdata), np.abs(np.max(inputdata))])
        vmin,vmax = -1*limit, limit

    # Create a custom colormap
    base_cmap = plt.cm.get_cmap(colormap)
    colors = base_cmap(np.linspace(0, 1, 256))

    # Set the first color to transparent or gray for missing regions
    colors[0] = [0.5, 0.5, 0.5, 0.0]  # transparent gray
    custom_cmap = ListedColormap(colors)

    # Map your data so missing regions get index 0
    inputdata_mapped = np.where(mask_valid_regions, 
                               ((inputdata_flipped - vmin) / (vmax - vmin) * 254 + 1).astype(int),
                               0)

    wbplot.pscalar(
        file_out=figoutdir + file_out + '.png',
        pscalars=inputdata_mapped,
        vrange=(0, 255),
        cmap=custom_cmap,
        transparent=True
    )


    plt.figure(figsize=(3.5,3))
    ax = plt.subplot(111)
    im = img.imread(figoutdir + file_out + '.png') 
    plt.imshow(im)
    plt.axis('off')
    plt.title(title,fontsize=18)

    # vmin, vmax = -2, 2
    cnorm = clrs.Normalize(vmin=vmin, vmax=vmax)  # only important for tick placing
    cmap = plt.get_cmap(colormap)
    cax = ax.inset_axes([0.44, 0.48, 0.12, 0.07])
    cbar = colorbar.ColorbarBase(
        cax, cmap=cmap, norm=cnorm, orientation='horizontal')
    cax.get_xaxis().set_tick_params(length=0, pad=-2)
    cbar.set_ticklabels([])
    cbar.outline.set_visible(False)
    cax.text(-0.025, 0.4, str(round(vmin,2)), ha='right', va='center', transform=cax.transAxes,
             fontsize=12);
    cax.text(1.025, 0.4, str(round(vmax,2)), ha='left', va='center', transform=cax.transAxes,
             fontsize=12);
    plt.tight_layout()
    plt.savefig(figoutdir + file_out + '.pdf',transparent=True)
    
def customScatterPlot(X,Y,RAaxis,xlabel,ylabel,outname,legLoc,xmin,xmax,ymin,ymax,custom_select=[],invert_xaxis=False,deg=1,xtickbins=3,ytickbins=3,plotwidth=5.75,showstat=False,regplot=True):

    if RAaxis == 'full':
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
        
    return fig
        

# Function to create the visualization
def plot_sorted_scatterplot_with_null(observed_values, null_distributions, pvalues, ylabel, file_out, title=None, alpha=0.05, 
                                         figsize=(24, 12), region_subset=None, sort=True):  # Increased figure size
    """
    Create violin plots of null distributions with observed points and significance markers.
    
    Parameters:
    -----------
    observed_values : array-like, shape (n_regions,)
        The observed correlation values for each brain region
    null_distributions : array-like, shape (n_regions, n_nulls)
        The null distribution values for each brain region
    pvalues : array-like, shape (n_regions,)
        P-values for each region to determine significance
    alpha : float, default=0.05
        Significance threshold for p-values
    figsize : tuple, default=(24, 12)
        Size of the figure (increased)
    region_subset : int or None, default=None
        If not None, only plot this many regions (useful for large datasets)
    sort : bool, default=True
        Whether to sort the regions by observed value
    """
    # Convert to numpy arrays for safety
    observed = np.array(observed_values)
    nulls = np.array(null_distributions)
    pvals = np.array(pvalues)
    
    # Sort if requested
    if sort:
        sort_idx = np.argsort(observed)[::-1]  # Sort in descending order
        observed = observed[sort_idx]
        nulls = nulls[sort_idx]
        pvals = pvals[sort_idx]
    
    # Take subset if requested
    if region_subset is not None and region_subset < len(observed):
        observed = observed[:region_subset]
        nulls = nulls[:region_subset]
        pvals = pvals[:region_subset]
    
    # Prepare data for seaborn
    n_regions = len(observed)
    region_indices = np.arange(n_regions)
    
    # Create long-form DataFrame for violin plots
    violin_data = []
    for i in range(n_regions):
        for null_val in nulls[i]:
            violin_data.append({
                'Region Index': i,
                'Correlation': null_val,
                'Type': 'Null'
            })
    
    for i in range(n_regions):
        violin_data.append({
            'Region Index': i,
            'Correlation': observed[i],
            'Type': 'Observed'
        })
    
    df_violin = pd.DataFrame(violin_data)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot violins for null distributions
    sns.violinplot(x='Region Index', y='Correlation', data=df_violin[df_violin['Type'] == 'Null'],
                  inner=None, color='lightgray', scale='width', ax=ax)
    
    # Plot observed values as scatter points
    sns.scatterplot(x='Region Index', y='Correlation', 
                   data=df_violin[df_violin['Type'] == 'Observed'],
                   color='red', s=80, ax=ax)  # Increased point size
    
    # Connect observed values with a line
    ax.plot(region_indices, observed, color='red', alpha=0.7, linewidth=2.5)  # Thicker line
    
    # Add significance asterisks
    for i in range(n_regions):
        if pvals[i] < alpha:
            ax.text(i, np.max(observed) + 0.02, '*', fontsize=8,  # Increased asterisk size 
                   horizontalalignment='center', color='black', weight='bold')
    
    # Enhance the plot with larger font sizes
    ax.set_xlabel('Brain Region (Sorted)', fontsize=22)  # Increased font size
    ax.set_ylabel(ylabel, fontsize=22)    # Increased font size
    ax.set_title(title, fontsize=24)  # Increased font size
    
    # Create a custom legend with larger font and significance indicator
    null_patch = mpatches.Patch(color='lightgray', label='Null Distribution')
    observed_patch = mpatches.Patch(color='red', label='Observed Value')
    sig_indicator = mpatches.Patch(color='none', label=f'* p < {alpha}')  # Add p-value to legend
    
    plt.legend(handles=[null_patch, observed_patch, sig_indicator], 
               loc='upper right', fontsize=14, 
               frameon=True, framealpha=1, facecolor='white', edgecolor='black')
    
    # # Add additional information about significance with larger font
    # ax.text(0.9, 0.8, '* p < {:.3f}'.format(alpha), transform=ax.transAxes, 
    #        fontsize=18, verticalalignment='bottom', weight='bold')  # Increased font size
    
    # Improve x-axis ticks with larger font
    if n_regions > 20:
        step = max(1, n_regions // 20)
        ax.set_xticks(region_indices[::step])
        ax.set_xticklabels(region_indices[::step], rotation=45 if n_regions > 50 else 0, fontsize=16)  # Increased tick font size
    
    # Make sure tick labels are larger too
    ax.tick_params(axis='both', which='major', labelsize=16)  # Increase tick label size
    
    sns.despine()
    plt.tight_layout()
    
    plt.savefig(figoutdir + file_out + '.pdf',transparent=True)
    
    return fig

def plot_null_hist(obs_value, null_hist_data, xlabel, title):

    fig,ax = plt.subplots(figsize=(5,4))
    plt.hist(null_hist_data,bins=100,color='gray')
    plt.axvline(x=obs_value, color='red',linewidth=4)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    #plt.title('Null Distributions and Observed Values (Sorted by Observed Means)')
    plt.ylabel('Frequency',fontsize=24)
    plt.xlabel(xlabel,fontsize=24)
    plt.locator_params(axis='x', nbins=4)
    # plt.legend(fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(figoutdir + title + '.pdf',transparent=True)

    return fig