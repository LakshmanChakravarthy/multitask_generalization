# Network geometry shapes multi-task representational transformations across human cortex

## Overview

This repository includes cleaned, modular code for analyzing functional connectivity (FC) and task-evoked representations in the brain, as described in Chakravarthula et al. (2025). The analysis spans resting-state FC estimation, task GLM modeling, representational similarity analysis, and activity flow modeling.

**Preprint Citation**: Chakravarthula, L. N., Ito, T., Tzalavras, A., & Cole, M. W. (2025). Network geometry shapes multi-task representational transformations across human cortex. *bioRxiv*, 2025-03. https://www.biorxiv.org/content/10.1101/2025.03.14.643366v1.full

**Contacts**: 
- lakastro@gmail.com
- michael.cole@rutgers.edu

**Last Updated**: Feb 26, 2026

---

## Data Repository

This code analyzes the publicly available **Multi-Domain Task Battery (MDTB)** dataset:

**OpenNeuro Repository**: [https://openneuro.org/datasets/ds002105](https://openneuro.org/datasets/ds002105/versions/1.1.0)

**Dataset Citation**: King, M., Hernandez-Castillo, C.R., Poldrack, R.A., Ivry, R.B., Diedrichsen, J., 2019. Functional boundaries in the human cerebellum revealed by a multi-domain task battery. *Nature Neuroscience* 22, 1371–1378. https://doi.org/10.1038/s41593-019-0436-x

**Preprocessing**: QuNex (version 0.61.17; https://qunex.yale.edu/)

---


## Core Modules

### 1. Resting-State FC Estimation

#### `graphical_lasso_cv.py`
**Purpose**: Sparse parcel-level FC estimation using graphical lasso with cross-validation

**Key Functions**:
- `graphicalLassoCV()`: Main CV function with L1 regularization
- `graphicalLasso()`: Core estimation with single L1 value
- `activityPrediction()`: Validates FC via activity prediction

**Input**: Resting-state fMRI (parcellated, n_parcels × n_timepoints)

**Output**: Sparse 360×360 FC matrix per subject

**Dependencies**: gglasso, sklearn

---

#### `vertexwise_fc_pcr.py`
**Purpose**: High-resolution vertex-wise FC using PC regression with CV-optimized hyperparameters

**Key Functions**:
- `process_subject()`: Complete pipeline for one subject
- `compute_vertexwise_fc_with_cv()`: Two-round CV for optimal nPCs
- `compute_pcr_fc()`: PC regression-based FC estimation

**Input**: 
- Vertex-level fMRI (59,412 vertices)
- Parcel-level FC skeleton (from graphical lasso)
- Glasser parcellation

**Output**: 
- `vertFC_dict`: FC matrices for each target region
- Optimal nPCs (subject-specific)
- R² values

**Key Innovation**: Data-driven nPC selection via averaged cross-validation across all regions (avoids overfitting while being subject-specific)

---

#### `data_utils.py`
**Purpose**: Standardized data loading utilities

**Key Functions**:
- `load_rsfmri()`: Load resting-state data (vertex or parcellated)
- `load_glasser_parcellation()`: Load Glasser atlas
- `load_all_subjects_parcel_fc()`: Load parcel FC for all subjects

---

### 2. Task GLM

#### `task_glm.py`
**Purpose**: Estimate task condition betas using ridge regression with nuisance regression

**Key Functions**:
- `compute_task_betas_single_run()`: Complete GLM for one run
- `load_task_timing_betas()`: Create HRF-convolved design matrix
- `ridge_regression_cv()`: CV-optimized ridge regression

**Input**: 
- Task fMRI (QuNex preprocessed)
- Task timing information

**Output**: 
- Task betas (conditions × parcels/vertices)
- Residual timeseries
- Task condition names

**Pipeline**:
1. Load fMRI data
2. Skip first 5 TRs, detrend
3. Create task regressors (HRF-convolved)
4. Load 32 nuisance regressors (QuNex model: 24 motion + 8 physiological, NO global signal)
5. Z-score data and regressors
6. Ridge regression with CV
7. Save betas

**Dataset**: MDTB (24 subjects, 4 sessions, 8 runs/session, ~45 conditions/run)

---

### 3. Representational Similarity Analysis

#### `rsm_computation.py`
**Purpose**: Compute cross-validated RSMs and representational dimensionality

**Key Functions**:
- `process_subject()`: Complete RSM pipeline for one subject
- `compute_rsm_single_region()`: Efficient cosine similarity via normalize + dot product
- `get_dimensionality()`: Participation ratio of eigenvalues
- `compute_dimensionality_all_regions()`: Dimensionality for all parcels

**Input**: 
- Task betas (from task_glm.py)
- Grouped by sessions: a-sessions (a1+a2) vs b-sessions (b1+b2)
- Task subset: 96 active visual conditions

**Output**:
- RSMs: (360, 96, 96) per subject
- Dimensionality: (360,) per subject

**Key Approach**: 
- Normalize beta patterns (L2)
- Cross-validated cosine similarity between session groups
- Participation ratio: (Σλ)² / Σλ²

---

#### `fc_dimensionality.py`
**Purpose**: Compute FC dimensionality using singular value participation ratio (SVPR)

**Key Functions**:
- `get_svpr()`: SVPR from FC matrix
- `compute_fc_dimensionality_all_regions()`: Dimensionality for all targets
- `compute_fc_dimensionality_all_subjects()`: Batch processing

**Input**: Vertex-wise FC (from vertexwise_fc_pcr.py)

**Output**: FC dimensionality (n_subjects, 360)

**Formula**: (Σs)² / Σs² where s = singular values

---

#### `fc_gradients.py`
**Purpose**: Compute principal gradients of FC organization

**Key Functions**:
- `load_and_compute_fc_gradients()`: Complete pipeline
- `compute_fc_gradients()`: PCA on thresholded FC
- `threshold_top_percentile()`: Keep top 80% of connections

**Input**: Parcel-level FC (360×360) averaged across subjects

**Output**: 
- Gradients: (360, 2) - PC loadings for each parcel
- Explained variance
- Mean FC matrix

**Pipeline**:
1. Average FC across subjects
2. Threshold: keep top 80% of connections
3. PCA on thresholded matrix
4. Extract first 2 principal components

**Usage**: Reveals dominant axes of connectivity variation across brain regions

---

### 4. Activity Flow Modeling

#### `actflow_prediction.py`
**Purpose**: Predict target region activations from source activations weighted by FC

**Key Functions**:
- `process_subject_actflow()`: Standard predictions
- `process_subject_actflow_double_cv()`: Double cross-validated predictions
- `load_mdtb_task_betas()`: Load task betas (same approach as RSM computation)
- `predict_target_betas_actflow()`: Core equation: predicted = source_betas × FC^T

**Input**:
- Task betas (2 session groups, 96 conditions, n_vertices)
- Vertex-wise FC
- Parcel-level FC skeleton

**Output**:
- Predicted betas
- Predicted RSMs (reuses `rsm_computation.py`)
- Predicted dimensionality (reuses `rsm_computation.py`)
- Double-CV RSMs (2 combinations kept separate)
- Double-CV dimensionality (averaged)

**Double Cross-Validation**:
- RSM_1: Observed-Session0 × Predicted-Session1
- RSM_2: Observed-Session1 × Predicted-Session0
- Ensures patterns are shared between observed AND predicted, not just session artifacts

---

#### `actflow_metrics.py`
**Purpose**: Evaluate transformation and prediction quality using distance metrics

**Key Functions**:
- `compute_transformation_distance()`: d_trans
- `compute_predicted_transformation_distance()`: d_trans_hat
- `compute_prediction_distance()`: d_pred
- `compute_transformation_evidence()`: d_trans_hat - d_pred

**Three Distance Metrics**:

1. **d_trans** (Transformation Distance):
   - Cosine distance: observed target RSM ↔ observed source RSMs
   - Measures: How much has target transformed from inputs?

2. **d_trans_hat** (Predicted Transformation Distance):
   - Cosine distance: predicted target RSM ↔ observed source RSMs
   - Measures: Does model predict similar transformation?

3. **d_pred** (Prediction Distance):
   - Cosine distance: predicted target RSM ↔ observed target RSM
   - Measures: Prediction accuracy

**Transformation Evidence**:
- If `d_trans_hat > d_pred`: Evidence that connectivity does meaningful transformation (predicted target closer to observed target than to sources)
- If `d_trans_hat < d_pred`: Model fails to capture transformation

---

#### `actflow_permutation.py`
**Purpose**: Generate null distributions via connectivity permutation

**Key Functions**:
- `permutation_test_single_subject()`: Complete permutation pipeline
- `permute_fc_matrix()`: Two permutation modes
- `compute_permuted_rsms()`: RSMs from permuted predictions
- `compute_permuted_transformation_distance()`: Permuted d_trans_hat

**Two Permutation Modes**:

1. **Full** (`permute_mode='full'`):
   - Shuffles ALL FC values
   - Null: Any connectivity pattern is equally likely
   - Tests: Does connectivity structure matter?

2. **Column-wise** (`permute_mode='columns'`):
   - Shuffles within each source (column)
   - Preserves: Target vertex in-degree and distribution
   - Null: Specific source→target arrangement doesn't matter
   - Tests: Does specific wiring pattern matter, controlling for target properties?

**Output**:
- Permuted RSMs (per parcel)
- Permuted dimensionality (360, n_permutations)
- Permuted d_trans_hat (360, n_permutations)
- P-values from permutation distributions

---

## Example Scripts

Each module has corresponding example scripts demonstrating usage:

- `example_single_subject.py`: Vertex-wise FC for one subject
- `example_task_glm.py`: Task GLM for one run
- `example_rsm.py`: RSM computation and dimensionality
- `example_fc_dimensionality.py`: FC dimensionality analysis
- `example_fc_gradients.py`: FC gradient computation with visualization
- `example_actflow.py`: Complete activity flow pipeline
- `example_actflow_metrics.py`: Distance metrics with interpretation
- `example_permutation_modes.py`: Comparison of permutation strategies

---

## Data Flow

### Resting-State Path:
```
Raw fMRI → Parcel FC (graphical_lasso_cv.py) → FC Gradients (fc_gradients.py)
                ↓
        Vertex-wise FC (vertexwise_fc_pcr.py)
                ↓
        FC Dimensionality (fc_dimensionality.py)
                
```

### Task Path:
```
Raw fMRI → Task GLM (task_glm.py) → Task Betas
                                        ↓
                                    Observed RSMs (rsm_computation.py)
                                        ↓
                                    Observed Dimensionality
```

### Activity Flow Path:
```
Task Betas + Vertex-wise FC → Activity Flow (actflow_prediction.py)
                                        ↓
                                Predicted Betas
                                        ↓
                                Predicted RSMs (reuses rsm_computation.py) → Distance Metrics (actflow_metrics.py)
                                        ↓                                       
                                Predicted Dimensionality                        ↓
                                        ↓
                                      Permutation Tests (actflow_permutation.py)
```

---


## Dependencies

### Python Packages:
- **Core**: numpy, scipy, pickle, h5py
- **Neuroimaging**: nibabel
- **Machine Learning**: scikit-learn
- **Optimization**: gglasso (external: https://github.com/GGLasso/GGLasso)
- **Utilities**: pandas, tqdm
- **Visualization** (examples): matplotlib, seaborn

### External Code:
- Huth lab ridge regression (`/home/ln275/f_mc1689_1/MDTB/huth_ridge`)

### Data:
- Glasser parcellation (360 parcels)
- Dilated parcel masks (10mm)
- Task timing info (`allSubTaskConditionInfo.pkl`)

---

## File Organization

### Current Repository Structure:
```
project/
├── src/                          # Core analysis modules
│   ├── graphical_lasso_cv.py
│   ├── vertexwise_fc_pcr.py
│   ├── data_utils.py
│   ├── task_glm.py
│   ├── rsm_computation.py
│   ├── fc_dimensionality.py
│   ├── fc_gradients.py
│   ├── actflow_prediction.py
│   ├── actflow_metrics.py
│   └── actflow_permutation.py
│
├── notebooks/                    # Basic figure generation & analyses
│   └── [Jupyter notebooks for generating figures and supplementary analyses]
│
├── scripts/                      # Addiitional analysis scripts
│   └── [Python scripts for processing data and generating derivatives]
│
├── examples/                     # Usage examples (optional)
│   ├── example_single_subject.py
│   ├── example_task_glm.py
│   ├── example_rsm.py
│   ├── example_fc_dimensionality.py
│   ├── example_fc_gradients.py
│   ├── example_actflow.py
│   ├── example_actflow_metrics.py
│   └── example_permutation_modes.py
│
├── data/                         # Data directory
│   ├── derivatives/
│   │   ├── FC_new/              # FC outputs
│   │   ├── betasByTaskCondition/ # Task GLM outputs
│   │   └── RSM_ActFlow/          # RSM outputs
│   └── ...
│
├── docs/                         # Documentation
│   └── files/                    # Helper files
│
└── README.md                     # This file
```

---

## Typical Analysis Workflow

1. **Estimate FC**:
   ```python
   # Parcel-level FC
   parcel_fc = graphical_lasso_cv.graphicalLassoCV(data, L1s)
   
   # Vertex-wise FC
   vertFC_dict = vertexwise_fc_pcr.process_subject(...)
   ```

2. **Compute FC Properties**:
   ```python
   # FC dimensionality
   fc_dims = fc_dimensionality.compute_fc_dimensionality_all_subjects(...)
   
   # FC gradients
   gradients = fc_gradients.load_and_compute_fc_gradients(...)
   ```

3. **Task Analysis**:
   ```python
   # Task GLM
   task_glm.compute_task_betas_single_run(...)
   
   # Observed RSMs
   rsm_computation.process_subject(...)
   
   # Observed dimensionality
   obs_dim = rsm_computation.compute_dimensionality_all_subjects(...)
   ```

4. **Activity Flow Modeling**:
   ```python
   # Predictions
   results = actflow_prediction.process_subject_actflow_double_cv(...)
   
   # Distance metrics
   distances = actflow_metrics.process_subject_distances(...)
   
   # Permutation tests
   perm_results = actflow_permutation.permutation_test_single_subject(...)
   ```

5. **Analysis**:
   - Correlate observed vs predicted dimensionality
   - Test transformation evidence
   - Map patterns along FC gradients
   - Compute permutation p-values

---

## Citations

If using this code, please cite:

**Main Paper**:
- Chakravarthula, L. N., Ito, T., Tzalavras, A., & Cole, M. W. (2025). Network geometry shapes multi-task representational transformations across human cortex. *bioRxiv*, 2025-03. https://www.biorxiv.org/content/10.1101/2025.03.14.643366v1.full

**MDTB Dataset**: 
- King, M., Hernandez-Castillo, C.R., Poldrack, R.A., Ivry, R.B., Diedrichsen, J. (2019). Functional boundaries in the human cerebellum revealed by a multi-domain task battery. *Nature Neuroscience*, 22(8), 1371–1378. https://doi.org/10.1038/s41593-019-0436-x

**Methods**:
- **Graphical Lasso**: Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics*, 9(3), 432-441.
- **Activity Flow**: Cole, M. W., Ito, T., Bassett, D. S., & Schultz, D. H. (2016). Activity flow over resting-state networks shapes cognitive task activations. *Nature Neuroscience*, 19(12), 1718-1726.

---

## System Requirements

**Neuroimaging Libraries**:
- nibabel (v5.3.2)
- Connectome Workbench (v1.5.0)

**Analysis Libraries**:
- numpy (v1.18.5+)
- scipy (v1.6.0+)
- scikit-learn (v1.6.1+)
- pandas
- h5py
- tqdm

**Visualization Libraries** (for examples):
- matplotlib (v3.10.0+)
- seaborn (v0.13.2+)

**External Dependencies**:
- gglasso: https://github.com/GGLasso/GGLasso
- Huth lab ridge regression (included in repository)

**Installation Time**: 1-2 minutes on a standard CPU

---

## Repository Organization

### Core Analysis Modules (`src/`)
This directory contains the core, reusable modules that implement the analysis pipeline (see **Core Modules** section below for detailed documentation).

### Figure Generation & Additional Analyses (`notebooks/` and `scripts/`)
- **`notebooks/`**: Jupyter notebooks that generate figure panels from processed derivative data and perform additional analyses discussed in the main text and supplement
- **`scripts/`**: Python scripts for running analyses on processed data and generating derivative outputs

**Note**: Not all derivative data for supplementary figures is included in the repository due to file size constraints. However, all code required to generate derivative data from preprocessed fMRI is included.


---
