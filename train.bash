#!/bin/bash

####################### BSUB Headers #########################
#BSUB -J train_lit_image_classifier_model_with_pl
#BSUB -P bip198
#BSUB -W 0:10
#BSUB -nnodes 2
#BSUB -q batch-hm
#BSUB -alloc_flags "gpumps NVME"
#BSUB -o /gpfs/alpine/scratch/acmwhb/bip198/Repositories/Personal_Repositories/deep-learning-hpc-project-template/job%J.out
#BSUB -e /gpfs/alpine/scratch/acmwhb/bip198/Repositories/Personal_Repositories/deep-learning-hpc-project-template/job%J.out
###############################################################

# Remote project path
export PROJDIR=$MEMBERWORK/bip198/Repositories/Personal_Repositories/deep-learning-hpc-project-template
export DGLBACKEND=pytorch # Required to override default ~/.dgl config directory which is read-only

# Configure OMP for PyTorch
export OMP_PLACES=threads

# Configure Conda for BSUB script environment
eval "$(conda shell.bash hook)"

# Remote Conda environment
conda activate "$PROJDIR"/venv

# Configure WandB logger for local configuration storage and proxy access on compute nodes
export WANDB_CONFIG_DIR=/gpfs/alpine/scratch/acmwhb/bip198/  # For local reading and writing of WandB files
export WANDB_CACHE_DIR=/gpfs/alpine/scratch/acmwhb/bip198/  # For logging checkpoints as artifacts
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=https://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,.ccs.ornl.gov,.ncrc.gov'

# Run training script
cd "$PROJDIR"/project || exit

START=$(date +%s)  # Capture script start time in seconds since Unix epoch
echo "Script started at $(date)"

# Execute script
jsrun -bpacked:7 -g6 -a6 -c42 -r1 python lit_image_classifier.py --num_epochs 25 --batch_size 8192 --hidden_dim 128 --num_dataloader_workers 28

END=$(date +%s)  # Capture script end time in seconds since Unix epoch
echo "Script finished at $(date)"

# Calculate and output time elapsed during script execution
((diff=END-START))
((seconds=diff))
((minutes=seconds/(60)))
((hours=minutes/(24)))
echo "Script took $seconds second(s) to execute"
echo "Script took $minutes minute(s) to execute"
echo "Script took $hours hour(s) to execute"