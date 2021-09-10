#!/bin/bash

####################### BSUB Headers #########################
#BSUB -J train_lit_image_classifier_model_with_pl
#BSUB -P bif132
#BSUB -W 0:15
#BSUB -nnodes 32
#BSUB -q batch
#BSUB -o job%J.out
#BSUB -e job%J.out
###############################################################

# Remote project path
export PROJDIR="$PWD"

# Remote Conda environment
module load open-ce/1.1.3-py38-0
conda activate DLHPT

# Configure proxy access on compute nodes
export WANDB_INSECURE_DISABLE_SSL=true
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,.ccs.ornl.gov,.ncrc.gov'
export LC_ALL=en_US.utf8

# Run training script
cd "$PROJDIR"/project || exit

# Execute script
date
jsrun -g6 -a6 -c42 -r1 python3 lit_image_classifier.py --logger_name WandB --num_gpus 6 --num_compute_nodes 32 --num_epochs 50 --batch_size 64 --hidden_dim 512 --lr 1e-4 --num_dataloader_workers 28 --log_dir /mnt/bb/"$USER"/tb_logs
date

# Copying leftover items from NVMe drive back to GPFS
echo "Copying log files back to GPFS..."
jsrun -n 1 cp -r /mnt/bb/"$USER"/tb_logs "$PROJDIR"/project/tb_logs
echo "Done copying log files back to GPFS"
