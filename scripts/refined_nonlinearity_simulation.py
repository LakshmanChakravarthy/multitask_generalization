import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def get_connectivity_dim(data):
    """
    singular value participation ratio
    """
    U, S, V_T = np.linalg.svd(data)
    
    dimensionality_nom = 0
    dimensionality_denom = 0
    for sv in S:
        dimensionality_nom += np.real(sv)
        dimensionality_denom += np.real(sv)**2
    dimensionality = dimensionality_nom**2/dimensionality_denom
    return dimensionality

def getDimensionality(data):
    """
    data is the cosine similarity matrix of activity shape: (n_tasks,n_tasks)
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

def generate_conn(source_size=10, target_size=10, conn_dim=10):
    """
    Generate low-rank connectivity matrix
    
    Parameters:
    -----------
    source_size : int
        Number of source vertices
    target_size : int
        Number of target vertices
    conn_dim : int
        Connectivity dimensionality (rank)
    
    Returns:
    --------
    conn : np.ndarray, shape (source_size, target_size)
        Connectivity matrix
    """
    U = np.random.randn(source_size, conn_dim)
    V = np.random.randn(conn_dim, target_size)
    conn = U @ V
    
    return conn

def apply_leaky_relu(activity, bias, alpha=0.01, task_noise_std=50, seed=None):
    """
    Apply leaky ReLU non-linearity to activity with bias term
    
    Parameters:
    -----------
    activity : np.ndarray, shape (n_tasks, n_vertices)
        Activity patterns
    bias : np.ndarray, shape (n_vertices,)
        Bias term for each vertex (region property)
    alpha : float
        Slope for negative part (leaky coefficient)
        Default 0.01 for strong attenuation
    task_noise_std : float
        Standard deviation of task-dependent noise to add to bias (default 50)
    seed : int or None
        Random seed for generating task noise
    
    Returns:
    --------
    transformed : np.ndarray
        Activity after leaky ReLU
    fraction_thresholded : float
        Fraction of units that were below threshold (averaged across tasks)
    """
    n_tasks, n_vertices = activity.shape
    
    # Add task-dependent noise to bias
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    # Generate task-specific noise: different for each task
    # Independent N(0, task_noise_std)
    task_noise = rng.randn(n_tasks, n_vertices) * task_noise_std
    effective_bias = bias[np.newaxis, :] + task_noise
    
    # Shift activity by bias
    shifted = activity - effective_bias
    
    # Count units below threshold
    below_threshold = shifted < 0
    fraction_thresholded = np.mean(below_threshold)
    
    # Apply leaky ReLU
    transformed = np.where(shifted > 0, shifted, alpha * shifted)
    
    return transformed, fraction_thresholded

def compute_rep_dim_from_activity(activity):
    """
    Compute representational dimensionality from activity matrix
    activity: (n_tasks, n_vertices)
    """
    # Compute cosine similarity across tasks
    cosine_sim = cosine_similarity(activity)
    # Compute dimensionality from similarity matrix
    return getDimensionality(cosine_sim)

def run_nonlinearity_test(source_size=100, target_size=100, n_tasks=100, 
                          conn_dim_range=None, n_seeds=10, leaky_alpha=0.01,
                          bias_mean=300, bias_std=50, task_noise_factor=0.5):
    """
    Test how leaky ReLU non-linearity at target affects the relationship between 
    connectivity dimensionality and change in representational dimensionality.
    
    Tests 2 conditions:
    1. Linear (baseline) - no non-linearity
    2. Leaky ReLU at target - with region-specific bias terms + task-dependent threshold variability
    
    Parameters:
    -----------
    source_size : int
        Number of source vertices
    target_size : int
        Number of target vertices
    n_tasks : int
        Number of task conditions (default 100)
    conn_dim_range : array-like or None
        Range of connectivity dimensions to test
    n_seeds : int
        Number of random seeds
    leaky_alpha : float
        Slope for negative part of leaky ReLU (default 0.01 for strong attenuation)
    bias_mean : float
        Mean of bias distribution (default 300)
    bias_std : float
        Standard deviation of bias distribution (default 50)
    task_noise_factor : float
        Task-dependent threshold noise as fraction of bias_std (default 0.5)
    """
    
    if conn_dim_range is None:
        conn_dim_max = int(np.min([source_size, target_size]))
        conn_dim_range = np.arange(10, conn_dim_max+1, 10)
    
    # Calculate task noise std from bias std
    task_noise_std = task_noise_factor * bias_std
    
    print(f"Bias distribution: N({bias_mean}, {bias_std})")
    print(f"Task-dependent threshold noise: N(0, {task_noise_std}) = N(0, {task_noise_factor} * {bias_std})")
    
    # Store results
    results = []
    threshold_fractions = []  # Track actual thresholding
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        
        # Generate bias terms from Gaussian N(bias_mean, bias_std)
        # Fixed per region, shared across tasks and conn_dims
        bias = np.random.normal(bias_mean, bias_std, size=target_size)
        
        for conn_dim in conn_dim_range:
            # Generate connectivity
            conn = generate_conn(source_size=source_size, 
                               target_size=target_size, 
                               conn_dim=conn_dim)
            
            # Measure connectivity dimensionality (same for all conditions)
            conn_dimensionality = get_connectivity_dim(conn)
            
            # Generate source activity (clean)
            source_activity = np.random.randn(n_tasks, source_size)
            
            # Compute source representational dimensionality
            source_rep_dim = compute_rep_dim_from_activity(source_activity)
            
            # Condition 1: Linear (baseline)
            target_activity_linear = source_activity @ conn
            target_rep_dim_linear = compute_rep_dim_from_activity(target_activity_linear)
            delta_rep_dim_linear = target_rep_dim_linear - source_rep_dim
            
            results.append({
                'seed': seed,
                'conn_dim': conn_dim,
                'conn_dimensionality': conn_dimensionality,
                'condition': 'linear',
                'source_rep_dim': source_rep_dim,
                'target_rep_dim': target_rep_dim_linear,
                'delta_rep_dim': delta_rep_dim_linear,
                'fraction_thresholded': np.nan
            })
            
            # Condition 2: Leaky ReLU at target with task-dependent threshold variability
            # Threshold variability represents region's task-dependent sensitivity
            target_activity_prerelu = source_activity @ conn
            target_activity_relu, frac_thresh = apply_leaky_relu(
                target_activity_prerelu, bias, alpha=leaky_alpha,
                task_noise_std=task_noise_std, seed=seed
            )
            target_rep_dim_relu = compute_rep_dim_from_activity(target_activity_relu)
            delta_rep_dim_relu = target_rep_dim_relu - source_rep_dim
            
            threshold_fractions.append(frac_thresh)
            
            results.append({
                'seed': seed,
                'conn_dim': conn_dim,
                'conn_dimensionality': conn_dimensionality,
                'condition': 'leaky_relu',
                'source_rep_dim': source_rep_dim,
                'target_rep_dim': target_rep_dim_relu,
                'delta_rep_dim': delta_rep_dim_relu,
                'fraction_thresholded': frac_thresh
            })
    
    # Report actual thresholding statistics
    mean_thresh = np.mean(threshold_fractions)
    std_thresh = np.std(threshold_fractions)
    print(f"\nActual thresholding: {mean_thresh:.3f} ± {std_thresh:.3f} (mean ± std)")
    
    return pd.DataFrame(results), mean_thresh

def analyze_correlations(results_df):
    """
    Analyze correlations between connectivity dimensionality and 
    change in representational dimensionality for each condition.
    """
    conditions = results_df['condition'].unique()
    
    correlation_results = []
    
    for condition in conditions:
        condition_data = results_df[results_df['condition'] == condition]
        
        # Compute correlation
        r, p = stats.pearsonr(condition_data['conn_dimensionality'], 
                              condition_data['delta_rep_dim'])
        
        correlation_results.append({
            'condition': condition,
            'correlation': r,
            'p_value': p,
            'n_points': len(condition_data)
        })
    
    corr_df = pd.DataFrame(correlation_results)
    
    # Sort by correlation strength
    corr_df = corr_df.sort_values('correlation', ascending=False)
    
    return corr_df

def compare_conditions(results_df):
    """
    Compare leaky_relu condition's correlation to the linear baseline
    using Fisher's z-transformation.
    """
    baseline_data = results_df[results_df['condition'] == 'linear']
    r_baseline, _ = stats.pearsonr(baseline_data['conn_dimensionality'],
                                    baseline_data['delta_rep_dim'])
    n_baseline = len(baseline_data)
    
    relu_data = results_df[results_df['condition'] == 'leaky_relu']
    r_relu, _ = stats.pearsonr(relu_data['conn_dimensionality'],
                               relu_data['delta_rep_dim'])
    n_relu = len(relu_data)
    
    # Fisher's z-transformation
    z_baseline = np.arctanh(r_baseline)
    z_relu = np.arctanh(r_relu)
    
    # Standard error
    se = np.sqrt(1/(n_baseline-3) + 1/(n_relu-3))
    
    # Z-statistic
    z_stat = (z_baseline - z_relu) / se
    
    # Two-tailed p-value
    p_diff = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    
    comparison = {
        'r_linear': r_baseline,
        'r_leaky_relu': r_relu,
        'r_difference': r_baseline - r_relu,
        'z_statistic': z_stat,
        'p_value': p_diff,
        'significant': p_diff < 0.05
    }
    
    return pd.DataFrame([comparison])

def plot_correlations(results_df, output_filename='correlation_comparison.png'):
    """
    Plot correlations for both conditions side by side
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe with both conditions
    output_filename : str
        Filename to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    sns.set_style('ticks')
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    conditions = ['linear', 'leaky_relu']
    titles = ['Linear', 'Leaky ReLU']
    
    for idx, (condition, title) in enumerate(zip(conditions, titles)):
        ax = axes[idx]
        
        # Get data for this condition
        condition_data = results_df[results_df['condition'] == condition]
        
        # Compute correlation for title
        r, p = stats.pearsonr(condition_data['conn_dimensionality'],
                              condition_data['delta_rep_dim'])
        
        # Create regression plot
        sns.regplot(
            data=condition_data,
            x='conn_dimensionality',
            y='delta_rep_dim',
            ax=ax,
            scatter_kws={'s': 80, 'alpha': 0.6},
            line_kws={'linewidth': 2.5}
        )
        
        # Set labels and title
        ax.set_xlabel('Connectivity Dimensionality', fontsize=16, fontweight='bold')
        ax.set_ylabel('Δ Representational Dimensionality', fontsize=16, fontweight='bold')
        ax.set_title(f'{title}\nr = {r:.3f}, p = {p:.3f}', fontsize=18, fontweight='bold')
        
        # Adjust tick parameters
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
        
        # Set number of ticks
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='y', nbins=4)
        
        # Make spines thicker
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        # Despine
        sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_filename}")
    plt.close()
    
    return fig

# Run the analysis
if __name__ == "__main__":
    print("="*60)
    print("LEAKY RELU NON-LINEARITY SIMULATION")
    print("="*60)
    
    results, mean_thresh = run_nonlinearity_test(
        source_size=100,
        target_size=100,
        n_tasks=100,
        n_seeds=10,
        leaky_alpha=0.01,
        bias_mean=400,
        bias_std=50,
        task_noise_factor=0.5
    )
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    correlations = analyze_correlations(results)
    print(correlations.to_string(index=False))
    
    print("\n" + "="*60)
    print("COMPARISON: LINEAR vs LEAKY RELU")
    print("="*60)
    comparison = compare_conditions(results)
    print(comparison.to_string(index=False))
    
    # Create visualization
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION")
    print("="*60)
    plot_correlations(results, output_filename='correlation_comparison.png')
    
    # Save results
    results.to_csv('leaky_relu_results.csv', index=False)
    correlations.to_csv('leaky_relu_correlations.csv', index=False)
    comparison.to_csv('leaky_relu_comparison.csv', index=False)
    
    # Save metadata
    metadata = {
        'bias_distribution': 'N(300, 50)',
        'task_noise_factor': '0.5 (threshold noise = 0.5 * bias_std)',
        'task_noise_std': '25',
        'n_tasks': '100',
        'mean_fraction_thresholded': f'{mean_thresh:.3f}',
        'leaky_alpha': 0.01
    }
    with open('leaky_relu_metadata.txt', 'w') as f:
        for key, value in metadata.items():
            f.write(f'{key}: {value}\n')
    
    print("\n" + "="*60)
    print("Results saved to CSV files and metadata.txt")
    print("="*60)
