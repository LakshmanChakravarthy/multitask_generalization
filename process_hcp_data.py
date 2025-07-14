import numpy as np
import h5py
import nibabel as nib
from pathlib import Path
import time

def compute_RSM_crossvalidated(betas_run1, betas_run2):
    '''
    Computes cross-validated RSM using cosine similarities between conditions from two independent runs
    
    Parameters:
    betas_run1: array of shape (n_vertices, n_conditions) from first run
    betas_run2: array of shape (n_vertices, n_conditions) from second run
    
    Returns:
    similarity_matrix: array of shape (n_conditions, n_conditions)
    where similarity_matrix[i,j] is the cosine similarity between
    condition i from run 1 and condition j from run 2
    '''
    # Normalize the vectors from each run
    betas1_normalized = betas_run1 / np.linalg.norm(betas_run1, axis=0)
    betas2_normalized = betas_run2 / np.linalg.norm(betas_run2, axis=0)
    
    # Compute cross-run similarities
    similarity_matrix = np.dot(betas1_normalized.T, betas2_normalized)
    
    return similarity_matrix

def getDimensionality(data):
    """
    Computes dimensionality from a square, symmetric matrix
    """
    eigenvalues, _ = np.linalg.eig(data)
    dimensionality_nom = 0
    dimensionality_denom = 0
    for eig in eigenvalues:
        dimensionality_nom += np.real(eig)
        dimensionality_denom += np.real(eig)**2

    dimensionality = dimensionality_nom**2/dimensionality_denom
    return dimensionality

def load_parcellation(helpfiles_dir):
    """
    Load Glasser parcellation scheme
    """
    glasserfilename = helpfiles_dir + 'Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
    glasser = np.squeeze(nib.load(glasserfilename).get_fdata())
    return glasser

def process_subject(subject_id, base_path, keyname_prefix, keyname_suffix, taskNames, glasser):
    """
    Process a single subject's data, computing both whole-brain and parcel-wise metrics
    """
    filename = "{}_glmOutput_64k_data.h5".format(subject_id)
    datafile = base_path / filename
    
    try:
        with h5py.File(datafile, 'r') as h5f:
            # Process RL run
            betas_RL = []
            # Process LR run
            betas_LR = []
            
            dataset = h5f['taskRegression']
            for task in taskNames:
                # Get RL run betas
                keyname_RL = keyname_prefix + task + '_RL' + keyname_suffix
                betas_RL.extend(dataset[keyname_RL][:,1:].T)
                
                # Get LR run betas
                keyname_LR = keyname_prefix + task + '_LR' + keyname_suffix
                betas_LR.extend(dataset[keyname_LR][:,1:].T)

            # Convert to numpy arrays
            betas_RL = np.asarray(betas_RL).T  # Shape: (n_vertices, n_conditions)
            betas_LR = np.asarray(betas_LR).T
            
            # Whole-brain computations
            whole_brain_rsm = compute_RSM_crossvalidated(betas_RL, betas_LR)
            whole_brain_dim = getDimensionality(whole_brain_rsm)
            
            # Parcel-wise computations
            n_parcels = 360
            parcel_dims = np.zeros(n_parcels)
            parcel_rsms = np.zeros((n_parcels, betas_RL.shape[1], betas_RL.shape[1]))
            
            for roi in range(1, n_parcels + 1):
                roi_idx = np.where(glasser == roi)[0]
                if len(roi_idx) > 0:  # Check if parcel exists
                    betas_RL_roi = betas_RL[roi_idx, :]
                    betas_LR_roi = betas_LR[roi_idx, :]
                    rsm_roi = compute_RSM_crossvalidated(betas_RL_roi, betas_LR_roi)
                    parcel_rsms[roi - 1] = rsm_roi
                    parcel_dims[roi - 1] = getDimensionality(rsm_roi)
            
            return whole_brain_rsm, whole_brain_dim, parcel_rsms, parcel_dims
            
    except Exception as e:
        print("Error processing subject {}: {}".format(subject_id, str(e)))
        return None, None, None, None

def main():
    # Path configurations
    base_path = Path('/home/ln275/f_mc1689_1/HCP352Data/data/hcppreprocessedmsmall/vertexWise/')
    helpfiles_dir = '/home/ln275/f_mc1689_1/multitask_generalization/docs/files/'
    keyname_prefix = 'tfMRI_'
    keyname_suffix = '_24pXaCompCorXVolterra_taskReg_betas_canonical'
    
    # Load parcellation
    print("Loading parcellation...")
    glasser = load_parcellation(helpfiles_dir)
    
    # Task configurations
    taskNames = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
    
    # Subject lists
    discovery_subjects = ['100206','108020','117930','126325','133928','143224','153934','164636','174437',
                          '183034','194443','204521','212823','268749','322224','385450','463040','529953',
                          '587664','656253','731140','814548','877269','978578','100408','108222','118124',
                          '126426','134021','144832','154229','164939','175338','185139','194645','204622',
                          '213017','268850','329844','389357','467351','530635','588565','657659','737960',
                          '816653','878877','987074','101006','110007','118225','127933','134324','146331',
                          '154532','165638','175742','185341','195445','205119','213421','274542','341834',
                          '393247','479762','545345','597869','664757','742549','820745','887373','989987',
                          '102311','111009','118831','128632','135528','146432','154936','167036','176441',
                          '186141','196144','205725','213522','285345','342129','394956','480141','552241',
                          '598568','671855','744553','826454','896879','990366','102513','112516','118932',
                          '129028','135629','146533','156031','167440','176845','187850','196346','205826',
                          '214423','285446','348545','395756','481042','553344','599671','675661','749058',
                          '832651','899885','991267','102614','112920','119126','129129','135932','147636',
                          '157336','168745','177645','188145','198350','208226','214726','286347','349244',
                          '406432','486759','555651','604537','679568','749361','835657','901442','992774',
                          '103111','113316','120212','130013','136227','148133','157437','169545','178748',
                          '188549','198451','208327','217429','290136','352738','414229','497865','559457',
                          '615744','679770','753150','837560','907656','993675','103414','113619','120414',
                          '130114','136833','150726','157942','171330']
    
    replication_subjects = ['178950','189450','199453','209228','220721','298455','356948','419239','499566',
                            '561444','618952','680452','757764','841349','908860','103818','113922','121618',
                            '130619','137229','151829','158035','171633','179346','190031','200008','210112',
                            '221319','299154','361234','424939','500222','570243','622236','687163','769064',
                            '845458','911849','104416','114217','122317','130720','137532','151930','159744',
                            '172029','180230','191235','200614','211316','228434','300618','361941','432332',
                            '513130','571144','623844','692964','773257','857263','926862','105014','114419',
                            '122822','130821','137633','152427','160123','172938','180432','192035','200917',
                            '211417','239944','303119','365343','436239','513736','579665','638049','702133',
                            '774663','865363','930449','106521','114823','123521','130922','137936','152831',
                            '160729','173334','180533','192136','201111','211619','249947','305830','366042',
                            '436845','516742','580650','645450','715041','782561','871762','942658','106824',
                            '117021','123925','131823','138332','153025','162026','173536','180735','192439',
                            '201414','211821','251833','310621','371843','445543','519950','580751','647858',
                            '720337','800941','871964','955465','107018','117122','125222','132017','138837',
                            '153227','162329','173637','180937','193239','201818','211922','257542','314225',
                            '378857','454140','523032','585862','654350','725751','803240','872562','959574',
                            '107422','117324','125424','133827','142828','153631','164030','173940','182739',
                            '194140','202719','212015','257845','316633','381543','459453','525541','586460',
                            '654754','727553','812746','873968','966975']
    
    # Initialize storage
    n_conditions = 24
    n_parcels = 360
    
    # Discovery dataset arrays
    discovery_whole_brain_rsms = np.zeros((len(discovery_subjects), n_conditions, n_conditions))
    discovery_whole_brain_dims = np.zeros(len(discovery_subjects))
    discovery_parcel_rsms = np.zeros((len(discovery_subjects), n_parcels, n_conditions, n_conditions))
    discovery_parcel_dims = np.zeros((len(discovery_subjects), n_parcels))
    
    # Replication dataset arrays
    replication_whole_brain_rsms = np.zeros((len(replication_subjects), n_conditions, n_conditions))
    replication_whole_brain_dims = np.zeros(len(replication_subjects))
    replication_parcel_rsms = np.zeros((len(replication_subjects), n_parcels, n_conditions, n_conditions))
    replication_parcel_dims = np.zeros((len(replication_subjects), n_parcels))
    
    # Process discovery subjects
    print("\nProcessing discovery subjects...")
    for i, subject in enumerate(discovery_subjects):
        wb_rsm, wb_dim, p_rsms, p_dims = process_subject(
            subject, base_path, keyname_prefix, keyname_suffix, taskNames, glasser
        )
        if wb_rsm is not None:
            discovery_whole_brain_rsms[i] = wb_rsm
            discovery_whole_brain_dims[i] = wb_dim
            discovery_parcel_rsms[i] = p_rsms
            discovery_parcel_dims[i] = p_dims
        print("Processed subject {} ({}/{})".format(subject, i+1, len(discovery_subjects)))
    
    # Process replication subjects
    print("\nProcessing replication subjects...")
    for i, subject in enumerate(replication_subjects):
        wb_rsm, wb_dim, p_rsms, p_dims = process_subject(
            subject, base_path, keyname_prefix, keyname_suffix, taskNames, glasser
        )
        if wb_rsm is not None:
            replication_whole_brain_rsms[i] = wb_rsm
            replication_whole_brain_dims[i] = wb_dim
            replication_parcel_rsms[i] = p_rsms
            replication_parcel_dims[i] = p_dims
        print("Processed subject {} ({}/{})".format(subject, i+1, len(replication_subjects)))
    
    # Save results
    output_path = Path('/home/ln275/f_mc1689_1/multitask_generalization/derivatives/HCP_derivatives/')
    output_path.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save discovery results
    np.save(output_path / 'discovery_whole_brain_rsms_{}.npy'.format(timestamp), discovery_whole_brain_rsms)
    np.save(output_path / 'discovery_whole_brain_dims_{}.npy'.format(timestamp), discovery_whole_brain_dims)
    np.save(output_path / 'discovery_parcel_rsms_{}.npy'.format(timestamp), discovery_parcel_rsms)
    np.save(output_path / 'discovery_parcel_dims_{}.npy'.format(timestamp), discovery_parcel_dims)
    np.save(output_path / 'discovery_subjects_{}.npy'.format(timestamp), discovery_subjects)
    
    # Save replication results
    np.save(output_path / 'replication_whole_brain_rsms_{}.npy'.format(timestamp), replication_whole_brain_rsms)
    np.save(output_path / 'replication_whole_brain_dims_{}.npy'.format(timestamp), replication_whole_brain_dims)
    np.save(output_path / 'replication_parcel_rsms_{}.npy'.format(timestamp), replication_parcel_rsms)
    np.save(output_path / 'replication_parcel_dims_{}.npy'.format(timestamp), replication_parcel_dims)
    np.save(output_path / 'replication_subjects_{}.npy'.format(timestamp), replication_subjects)
    
    print("\nProcessing complete!")
    print("Results saved in {} directory with timestamp {}".format(output_path, timestamp))

if __name__ == "__main__":
    main()