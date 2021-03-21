#!/bin/bash

####################### BSUB Headers #########################
#BSUB -J train_lit_image_classifier_model_with_pl
#BSUB -P bip198
#BSUB -W 0:10
#BSUB -nnodes 2
#BSUB -q batch-hm
#BSUB -alloc_flags "gpumps"
#BSUB -o /gpfs/alpine/scratch/acmwhb/bip198/Repositories/Personal_Repositories/deep-learning-hpc-project-template/job%J.out
#BSUB -e /gpfs/alpine/scratch/acmwhb/bip198/Repositories/Personal_Repositories/deep-learning-hpc-project-template/job%J.out
###############################################################

# Remote project path
export PROJDIR=$MEMBERWORK/bip198/Repositories/Personal_Repositories/deep-learning-hpc-project-template

# Configure Conda for BSUB script environment
eval "$(conda shell.bash hook)"

# Remote Conda environment
conda activate "$PROJDIR"/venv

# Configure Neptune.ai logger for local configuration storage and proxy access on compute nodes
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=https://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,.ccs.ornl.gov,.ncrc.gov'

# Run training script
cd "$PROJDIR"/project || exit
START=$(date +%s)  # Capture script start time in seconds since Unix epoch
jsrun -bpacked:7 -g6 -a6 -c42 -r1 python lit_image_classifier.py
END=$(date +%s)  # Capture script end time in seconds since Unix epoch

# Calculate and output number of hours elapsed during script execution
((diff=END-START))
((minutes=diff/(60)))
echo "Script took $minutes minutes to execute"