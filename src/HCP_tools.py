import pandas as pd 
import numpy as np
from scipy import stats

def get_composite_scores(subject_list, behavior_filepath):
    """
    Compute composite intelligence scores and individual measures from behavioral data
    
    Args:
        subject_list (list): List of subject IDs as strings
        behavior_filepath (str): Path to behavioral CSV file
    
    Returns:
        tuple: (composite_scores, fluid_comp_scores, pmat_scores) as numpy arrays
               aligned with input subject list order
    """
    behavior_df = pd.read_csv(behavior_filepath)
    df_subset = behavior_df[['Subject', 'CogFluidComp_Unadj', 'PMAT24_A_CR']]
    
    subject_list_int = [int(x) for x in subject_list]
    df_filtered = df_subset[df_subset['Subject'].isin(subject_list_int)]
    
    # Compute z-scores and average
    cols = ['CogFluidComp_Unadj', 'PMAT24_A_CR']
    z_scores = df_filtered[cols].apply(stats.zscore)
    composite_scores = z_scores.mean(axis=1)
    
    # Create and order results
    result = pd.DataFrame({
        'Subject': df_filtered['Subject'],
        'composite_score': composite_scores,
        'CogFluidComp_Unadj': df_filtered['CogFluidComp_Unadj'],
        'PMAT24_A_CR': df_filtered['PMAT24_A_CR']
    })
    
    # Reindex to match input subject order
    result = result.set_index('Subject').reindex(subject_list_int).reset_index()
    
    return (result['composite_score'].to_numpy(),
            result['CogFluidComp_Unadj'].to_numpy(),
            result['PMAT24_A_CR'].to_numpy())