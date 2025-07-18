# multitask_generalization
#### Demo code repository for Chakravarthula et al. (2025)
#### Contacts: lakastro@gmail.com, michael.cole@rutgers.edu
#### Last updated: 7/18/2025

Preprint citation: Chakravarthula, L. N., Ito, T., Tzalavras, A., & Cole, M. W. (2025). Network geometry shapes multi-task representational transformations across human cortex. bioRxiv, 2025-03. https://www.biorxiv.org/content/10.1101/2025.03.14.643366v1.full

## Overview

This repository includes all raw code that was used for analyses and modeling on the publicly available Multi-Domain Task Battery dataset. Some of the code is written for the Rutgers Amarel cluster (see "slurm_...py"), but in principle, the python and shell scripts can be adapted for servers/clusters.

Link to the Multi-Domain Task Battery dataset: [https://openneuro.org/datasets/ds002105](https://openneuro.org/datasets/ds002105)
Citation for the Multi-Domain Task Battery dataset:
King, M., Hernandez-Castillo, C.R., Poldrack, R.A., Ivry, R.B., Diedrichsen, J., 2019. Functional boundaries in the human cerebellum revealed by a multi-domain task battery. Nature Neuroscience 22, 1371–1378. [https://doi.org/10.1038/s41593-019-0436-x](https://doi.org/10.1038/s41593-019-0436-x)

## System requirements

Standard neuroimaging libraries:
*   nibabel (v 5.3.2)
*   workbench (v 1.5.0)
Analysis lbraries 
*   numpy (v 1.18.5) 
*   scipy (v 1.6.0)
*   sklearn (v 1.6.1)
Plotting libraries 
*   matplotlib (3.10.0)
*   seaborn (0.13.2)

QuNex (version 0.61.17) was used to preprocess the raw data.
Typical install time is 1-2 minutes on a standard CPU.

## Description/Organization of code repository

`src/`: contains both Jupyter Notebooks that generate Figure panels from derivative processed data. Not all derivative data for supplementary figure panels is included (due to file size constraints). However, all the code required to generate the derivative data is included (in `*.py` files). Note that `*.py` files are included, and generate derivative data from preprocessed fMRI data.

`notebooks/`: contains jupyter notebooks with figure panels and other analyses discussed in text.

`processed_data/`: link to derivatives data will be shared soon.

Preprocessing was performed using QuNex (version 0.61.17; [https://qunex.yale.edu/](https://qunex.yale.edu/)). Postprocessing (i.e., task activation estimation) can be found in `src/postproc_betasByTaskCondition.py`.
