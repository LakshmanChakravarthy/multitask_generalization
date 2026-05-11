

projdir = '/home/ln275/f_mc1689_1/multitask_generalization/'
subProjDir = projdir + 'data/derivatives/RSM_ActFlow/'

## Reading in observed betas

# Task conditions subset (active visual task conditions)

nTaskCond_select = 96 # excluding passive tasks and interval timing (auditory task)

TaskCondIdx_subset = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,36,37,38,39,40,
                      41,42,43,44,45,46,49,50,51,52,53,54,55,56,60,61,62,63,64,65,
                      66,67,68,69,70,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,
                      87,88,89,91,92,93,94,95,96,97,98,99,100,101,102,104,105,106,
                      107,108,109,110,111,112,113,114,115,120,121,122,123,124,125])

with open(subProjDir + 'allSubTaskCondBetas.pkl', 'rb') as f:
    allSubTaskBetas = pickle.load(f)
    
selectsub_selecttaskcond_TaskBetas = allSubTaskBetas[onlyRestSubIdx,:,:,:][:,:,TaskCondIdx_subset,:]
# selectsub_selecttaskcond_TaskBetas.shape is (18, 2, 96, 91282), which is (n_subjects, n_sessions, n_taskcond, n_vertices)