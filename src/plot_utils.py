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
nSub=18
nParcels = 360
nNetwork = 13
networkdef = np.loadtxt(helpfiles_dir + 'cortex_parcel_network_assignments_separate_motor.txt')


# Following colors closest to 'magma' palette
color1, color2, color3 = clrs.to_rgba('indigo'),clrs.to_rgba('mediumvioletred'),clrs.to_rgba('coral')

sensorynets = [1,2]
associationnets = [4,5,6,7,8,9,10,11,12,13]
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

# Differentiating 13 networks

# networkNames = ['VIS1','VIS2','SMN','CON','DAN','LAN','FPN','AUD','DMN','PMULTI','VMM','ORA']
networkNames = ['VIS1','VIS2','SMN-Fr','CON','DAN','LAN','FPN','AUD','DMN','PMULTI','VMM','ORA','SMN-Par']
networkpalette = np.array(['royalblue', 'slateblue', 'paleturquoise', 'darkorchid', 
                           'limegreen', 'lightseagreen', 'yellow', 'orchid', 'r', 
                           'peru', 'orange', 'olivedrab', 'teal'])

# Map each ROI to its network's color
allnet_roiColorsByNetwork = np.array([
    clrs.to_rgb(networkpalette[int(networkdef[roi]) - 1]) 
    for roi in range(nParcels)
])

# If you need the network IDs
allnet_roi_id = networkdef.copy()

# # Set the style for the plot
# plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_context("talk", font_scale=1.5)  # Increased font scale from 1.5 to 2.0 and context to "talk"

def get_network_definition_brain_plot():

    # Create a discrete colormap from your network colors
    network_colors_rgb = [clrs.to_rgb(c) for c in networkpalette]
    discrete_cmap = clrs.ListedColormap(network_colors_rgb)

    # Your input data is just the network assignment for each parcel
    inputdata_mapped = networkdef - 1  # 0-indexed network IDs for each of 360 parcels

    wbplot.pscalar(
        file_out=figoutdir + 'network_definition_brain_plot' + '.png',
        pscalars=inputdata_mapped,
        vrange=(0, len(networkNames) - 1),  # 0 to 12 for your 13 networks
        cmap=discrete_cmap,
        transparent=True
    )
    

def get_brain_plot(inputdata,file_out,title,colormap=None,ignore_unvalid=False):

    #flip hemispheres, since CAB-NP is ordered left-to-right, while wbplot uses right-to-left
    inputdata_flipped=np.zeros(np.shape(inputdata))
    inputdata_flipped[0:180]=inputdata[180:360]
    inputdata_flipped[180:360]=inputdata[0:180]

    if ignore_unvalid:
        mask_valid_regions = inputdata != -9999
        min_inputdata = np.min(inputdata[mask_valid_regions])
    else:
        mask_valid_regions = inputdata
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
    elif RAaxis == 'onlysensory':
        Xselect = X[sensory_roi_id]
        Yselect = Y[sensory_roi_id]
        colorsSelect = allnet_roiColorsByNetwork[sensory_roi_id]
    elif RAaxis == 'onlyassociation':
        Xselect = X[association_roi_id]
        Yselect = Y[association_roi_id]
        colorsSelect = allnet_roiColorsByNetwork[association_roi_id]
    elif RAaxis == 'onlymotor':
        Xselect = X[motor_roi_id]
        Yselect = Y[motor_roi_id]
        colorsSelect = allnet_roiColorsByNetwork[motor_roi_id]
    
    
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
        
        r, p = stats.pearsonr(Xselect,Yselect)
        r = round(r,2)

        if legLoc == 'TL':
            legX,legY = 0.075,0.85     
        elif legLoc == 'TR':
            legX,legY = 0.55,0.85
        elif legLoc == 'BL':
            legX,legY = 0.075,0.1
        elif legLoc == 'BR':
            legX,legY = 0.55,0.1

        plt.annotate(r'$r$'+ ' = ' + str(r),
                     xy=(legX,legY),fontsize=28,xycoords='axes fraction')
        # plt.annotate(r'$p$'+ ' = ' + "{:.2e}".format(p),
        #              xy=(legX,legY-0.05),fontsize=28,xycoords='axes fraction')
    
    sns.despine()
    plt.tight_layout()
    
    if outname:
        plt.savefig(outname,transparent=True)
        
    return fig,r,p
        

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

def allNetsBoxPlot(Y1, Y2, ylabel, xlabel, xmin=None, xmax=None, ymin=None, ymax=None, 
                   networklist=range(1, 14), Y2_is_gradient=True, pc_idx=0, plotwidth=5.75):
    """
    Create box plots for network-averaged metrics with x-positions determined by another metric.
    
    Parameters:
    -----------
    Y1 : array-like
        Primary metric values for y-axis (shape: n_subjects x n_regions)
    Y2 : array-like
        Metric for x-positioning. Either:
        - If Y2_is_gradient=True: gradient loadings (shape: n_regions x n_pcs)
        - If Y2_is_gradient=False: subject-wise values (shape: n_subjects x n_regions)
    ylabel : str
        Label for y-axis
    xlabel : str
        Label for x-axis
    xmin, xmax, ymin, ymax : float, optional
        Axis limits
    networklist : list or range, default=range(1, 14)
        List of network indices to plot (1-13, where 13 is SMN-Par)
    Y2_is_gradient : bool, default=True
        Whether Y2 is gradient loadings (True) or subject-wise metric (False)
    pc_idx : int, default=0
        Which principal component to use if Y2_is_gradient=True
    plotwidth : float, default=5.75
        Width of the figure
    """
    
    metricAllNets_Y1 = returnNetworkAverages(Y1, networklist)
    
    if Y2_is_gradient:
        metricAllNets_Y2 = None
    else:
        metricAllNets_Y2 = returnNetworkAverages(Y2, networklist)
    
    # Calculate x-positions and collect data
    x_positions = []
    network_labels = []
    network_colors = []
    
    for netw in networklist:
        thisNetROIs = np.where(networkdef == netw)[0]
        
        network_labels.append(networkNames[netw - 1])
        network_colors.append(networkpalette[netw - 1])
        
        if Y2_is_gradient:
            x_positions.append(np.mean(Y2[thisNetROIs, pc_idx]))
        else:
            x_positions.append(np.mean(Y2[:, thisNetROIs]))
    
    # Build dataframe
    df_list = []
    for netwIdx, netw in enumerate(networklist):
        thisNetROIs = np.where(networkdef == netw)[0]
        
        for subIdx in range(nSub):
            df_list.append({
                'Y1value': metricAllNets_Y1[subIdx, netwIdx],
                'Y2value': metricAllNets_Y2[subIdx, netwIdx] if not Y2_is_gradient else x_positions[netwIdx],
                'group_by': network_labels[netwIdx],
                'x_pos': x_positions[netwIdx],
                'color': network_colors[netwIdx]
            })
    
    df = pd.DataFrame(df_list)
    
    # Sort by x_position for proper ordering
    df_sorted = df.sort_values('x_pos')
    order = df_sorted['group_by'].unique()
    x_pos_dict = dict(zip(network_labels, x_positions))
    color_dict = dict(zip(network_labels, network_colors))
    
    # Create plot with matching figure size
    fig, ax = plt.subplots(figsize=(plotwidth, 5))
    
    # Determine x-axis range
    if Y2_is_gradient:
        x_min = np.min(Y2[:, pc_idx])
        x_max = np.max(Y2[:, pc_idx])
    else:
        x_min = np.min(Y2)
        x_max = np.max(Y2)
    
    full_range = x_max - x_min
    box_width = full_range / 40
    
    # Box plots at computed positions
    bp = ax.boxplot(
        [df[df['group_by'] == net]['Y1value'].values for net in order],
        positions=[x_pos_dict[net] for net in order],
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        whis=[2.5, 97.5]
    )
    
    # Color boxes and set line widths
    for patch, net in zip(bp['boxes'], order):
        patch.set_facecolor(color_dict[net])
        patch.set_alpha(0.6)  # Match scatter plot alpha
        patch.set_linewidth(2)
    
    # Thicker whiskers, caps, and medians
    for whisker in bp['whiskers']:
        whisker.set_linewidth(2)
    for cap in bp['caps']:
        cap.set_linewidth(2)
    for median in bp['medians']:
        median.set_linewidth(2.5)
        median.set_color('black')
    
    # Add strip plot with matching scatter style
    for net in order:
        net_data = df[df['group_by'] == net]
        jitter_amount = box_width * 0.3
        x_jitter = np.random.normal(x_pos_dict[net], jitter_amount, len(net_data))
        ax.scatter(x_jitter, net_data['Y1value'], 
                  alpha=0.6, s=70, color=color_dict[net], zorder=0, edgecolor='none')
    
    # Calculate means
    means_df = df.groupby('group_by', sort=False)['Y1value'].mean().reindex(order)
    mean_x = [x_pos_dict[net] for net in order]
        
    # Styling to match customScatterPlot
    plt.xlabel(xlabel, fontsize=28)
    plt.ylabel(ylabel, fontsize=28)
    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=3)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    
    # Set axis limits
    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)
    else:
        padding = full_range / 20
        ax.set_xlim(np.min(mean_x) - padding, np.max(mean_x) + padding)
    
    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)
        
    # Format x-tick labels to 2 decimal places
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    ax.xaxis.set_tick_params(width=4)
    ax.yaxis.set_tick_params(width=4)
    
    # Match despine and tight_layout order
    sns.despine()
    plt.tight_layout()
    
    # Create legend with network names and colors
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w', 
                                  markerfacecolor=color_dict[net], markersize=10, 
                                  label=net, markeredgewidth=0) 
                      for net in order]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
              frameon=False, fontsize=12)
    
    plt.savefig(figoutdir + ylabel + '_vs_' + xlabel + '_AllNetworksBoxPlot.pdf', 
                transparent=True, bbox_inches='tight')
    
    return fig, ax, df


def returnNetworkAverages(metric,networklist): 

    metricAllNets = np.zeros((nSub,len(networklist)))
    for subIdx in range(nSub):
        for netwIdx,netw in enumerate(networklist):
            # if netw == 2:
            #     metricAllNets[subIdx,netwIdx] = np.mean(metric[subIdx,np.where((networkdef==1) |(networkdef==2))[0]])
            # else:
            metricAllNets[subIdx,netwIdx] = np.mean(metric[subIdx,np.where(networkdef==netw)[0]])

    return metricAllNets


def create_network_legend(network_names, network_colors, **kwargs):
    """
    Create a standalone legend with colored dots and labels.
    
    Parameters:
    -----------
    network_names : list or array
        Labels for each network
    network_colors : list or array
        Matplotlib color names corresponding to each network
    **kwargs : optional
        Additional arguments passed to plt.legend()
        Common options: ncol, fontsize, frameon, loc, bbox_to_anchor
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Create legend handles
    handles = [mpatches.Patch(color=color, label=name) 
               for name, color in zip(network_names, network_colors)]
    
    # Create figure with legend only
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    
    # Default legend parameters (can be overridden by kwargs)
    legend_params = {
        'ncol': 2,
        'fontsize': 10,
        'frameon': True,
        'loc': 'center'
    }
    legend_params.update(kwargs)
    
    ax.legend(handles=handles, **legend_params)
    
    return fig, ax