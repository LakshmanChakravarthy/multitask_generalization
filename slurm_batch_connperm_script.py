#!/usr/bin/env python
# slurm_actflow_batch.py - Script to generate and submit SLURM jobs for actflow analysis

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Generate and submit SLURM jobs for actflow analysis')
    
    # Configuration options
    parser.add_argument('--proj-dir', type=str, 
                        default='/home/ln275/f_mc1689_1/multitask_generalization/',
                        help='Project directory path')
    parser.add_argument('--nperm', type=int, default=100,
                        help='Number of permutations to run')
    parser.add_argument('--all-subjects', action='store_true',
                        help='Run for all subjects (default: False)')
    parser.add_argument('--subjects', type=int, nargs='+', 
                        help='Subject indices to process (if not all_subjects)')
    
    # SLURM specific parameters
    parser.add_argument('--time', type=str, default='24:00:00',
                        help='Maximum runtime in format HH:MM:SS')
    parser.add_argument('--mem', type=int, default=128000,
                        help='Memory to request (in MB)')
    parser.add_argument('--email', type=str, default=None,
                        help='Email address for job notifications')
    parser.add_argument('--no-submit', action='store_true',
                        help='Generate job scripts but do not submit to SLURM')
    parser.add_argument('--partition', type=str, default='main',
                        help='SLURM partition to use (default: main)')
    parser.add_argument('--cpus-per-task', type=int, default=4,
                        help='Number of CPU cores per task (default: 4)')
    
    args = parser.parse_args()
    
    # Define subject IDs (same as in your original script)
    subIDs = ['02', '03', '06', '08', '10', '12', '14', '18', '20',
              '22', '24', '25', '26', '27', '28', '29', '30', '31']
    
    # Determine which subjects to process
    if args.all_subjects:
        subject_indices = list(range(len(subIDs)))
    elif args.subjects:
        subject_indices = args.subjects
    else:
        print("Error: Must specify either --all-subjects or --subjects")
        sys.exit(1)
    
    # Create a timestamped directory for this batch of experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(f"./actflow_slurm_batch_{timestamp}")
    batch_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a directory for SLURM scripts and logs
    slurm_dir = batch_dir / "slurm_scripts"
    slurm_dir.mkdir(exist_ok=True)
    
    print(f"Batch scripts will be saved in: {batch_dir}")
    print(f"SLURM scripts will be saved in: {slurm_dir}")
    
    # Track all job IDs
    all_job_ids = []
    
    # Create a master script to track all jobs
    master_script_path = slurm_dir / "master_job_list.txt"
    with open(master_script_path, 'w') as master_file:
        master_file.write(f"# SLURM batch generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        master_file.write(f"# Total jobs: {len(subject_indices)}\n\n")
    
    # Generate a SLURM script for each subject
    for sub_idx in subject_indices:
        # Create a unique job name
        job_name = f"actflow_sub{subIDs[sub_idx]}"
        
        # Create the SLURM script for this subject
        slurm_script_path = slurm_dir / f"{job_name}.sh"
        
        with open(slurm_script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("\n# ---- Resource Allocation ----\n")
            f.write(f"#SBATCH --job-name={job_name}\n")
            f.write("#SBATCH --nodes=1\n")
            f.write(f"#SBATCH --ntasks=1\n")
            f.write(f"#SBATCH --cpus-per-task={args.cpus_per_task}\n")
            f.write(f"#SBATCH --mem={args.mem}\n")
            f.write(f"#SBATCH --time={args.time}\n")
            f.write(f"#SBATCH --partition={args.partition}\n")
            
            # Add email notifications if specified
            if args.email:
                f.write("#SBATCH --mail-type=BEGIN,END,FAIL\n")
                f.write(f"#SBATCH --mail-user={args.email}\n")
            
            # Output and error files
            f.write(f"#SBATCH --output={slurm_dir}/{job_name}.out\n")
            f.write(f"#SBATCH --error={slurm_dir}/{job_name}.err\n")
            f.write("#SBATCH --export=ALL\n\n")
            
            # Environment setup (using base environment as requested)
            f.write("\n# ---- Environment Setup ----\n")
            f.write("# Load necessary modules\n")
            f.write("module purge\n")
            f.write("\n")
            
            # Write the actual command to run the experiment
            f.write("\n# ---- Run Experiment ----\n")
            f.write("echo \"Starting analysis for subject ${subIDs[sub_idx]}\"\n")
            f.write("cd /home/ln275/f_mc1689_1/multitask_generalization/docs/scripts/local_upload/\n\n")        
            f.write(f"python get_actflow_pred_betas_connperm_modified.py \\\n")
            f.write(f"    --sub-idx {sub_idx} \\\n")
            f.write(f"    --nperm {args.nperm} \\\n")
            f.write(f"    --proj-dir {args.proj_dir}\n")
        
        # Make the script executable
        os.chmod(slurm_script_path, 0o755)
        
        # Submit the job if not in no-submit mode
        if not args.no_submit:
            cmd = f"sbatch {slurm_script_path}"
            print(f"Submitting job: {job_name}")
            
            # Get the job ID
            try:
                result = os.popen(cmd).read().strip()
                job_id = result.split()[-1]
                all_job_ids.append(job_id)
                
                # Add to master list
                with open(master_script_path, 'a') as master_file:
                    master_file.write(f"Job ID: {job_id} - {job_name}\n")
                
                print(f"Submitted job ID: {job_id}")
            except Exception as e:
                print(f"Error submitting job: {e}")
        else:
            print(f"Generated script for: {job_name}")
    
    # Create a script to check the status of all jobs
    if all_job_ids and not args.no_submit:
        status_script_path = slurm_dir / "check_job_status.sh"
        with open(status_script_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# Check the status of all jobs in the batch\n")
            f.write("echo 'Checking status of all batch jobs...'\n")
            f.write(f"squeue -u $USER | grep -f <(echo")
            for job_id in all_job_ids:
                f.write(f" {job_id}")
            f.write(")\n")
        
        os.chmod(status_script_path, 0o755)
        print(f"\nCreated job status check script: {status_script_path}")
    
    # Create a script to cancel all jobs if needed
    if all_job_ids and not args.no_submit:
        cancel_script_path = slurm_dir / "cancel_all_jobs.sh"
        with open(cancel_script_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# Cancel all jobs in the batch\n")
            f.write("echo 'Cancelling all batch jobs...'\n")
            f.write("scancel")
            for job_id in all_job_ids:
                f.write(f" {job_id}")
            f.write("\n")
        
        os.chmod(cancel_script_path, 0o755)
        print(f"Created job cancellation script: {cancel_script_path}")
    
    # Final message
    if args.no_submit:
        print(f"\nGenerated {len(subject_indices)} job scripts. To submit, run them individually with 'sbatch' or use:")
        print(f"for script in {slurm_dir}/*.sh; do [ -f \"$script\" ] && sbatch \"$script\"; done")
    else:
        print(f"\nSubmitted {len(all_job_ids)} out of {len(subject_indices)} jobs to SLURM")
        print(f"To check job status: {slurm_dir}/check_job_status.sh")
        print(f"To cancel all jobs: {slurm_dir}/cancel_all_jobs.sh")

if __name__ == "__main__":
    main()
