#!/bin/bash

####################### BSUB Headers #########################
#BSUB -J train_lit_image_classifier_model_with_pl
#BSUB -P bif132
#BSUB -W 0:10
#BSUB -nnodes 2
#BSUB -q batch
#BSUB -alloc_flags "gpumps NVME"
#BSUB -o job%J.out
#BSUB -e job%J.out
###############################################################

# Remote project path
export USER=acmwhb
export PROJID=bip198
export DATASET=MNIST
export PROJDIR=$MEMBERWORK/$PROJID/Repositories/Personal_Repositories/deep-learning-hpc-project-template
export DGLBACKEND=pytorch # Required to override default ~/.dgl config directory which is read-only

# Configure OMP for PyTorch
export OMP_PLACES=threads

# Remote Conda environment
conda activate "$PROJDIR"/venv

# Run training script
cd "$PROJDIR"/project || exit

# Execute script
date
jsrun -bpacked:7 -g6 -a6 -c42 -r1 python lit_image_classifier.py --num_epochs 25 --batch_size 16384 --hidden_dim 128 --num_dataloader_workers 28 --tb_log_dir /mnt/bb/$USER/tb_log --ckpt_dir /mnt/bb/$USER/checkpoints
date

# Copying leftover items from NVMe drive back to GPFS
echo "Copying log files and best checkpoint(s) back to GPFS..."
jsrun -n 1 cp -r /mnt/bb/$USER/tb_log /gpfs/alpine/scratch/$USER/$PROJID/Repositories/Personal_Repositories/deep-learning-hpc-project-template/project/tb_log
jsrun -n 1 cp -r /mnt/bb/$USER/checkpoints /gpfs/alpine/scratch/$USER/$PROJID/Repositories/Personal_Repositories/deep-learning-hpc-project-template/project/checkpoints
echo "Done copying log files and best checkpoint(s) back to GPFS"
