#!/bin/bash

####################### BSUB Headers #########################
#BSUB -J train_lit_image_classifier_model_with_pl
#BSUB -P bif132
#BSUB -W 0:10
#BSUB -nnodes 2
#BSUB -q batch
#BSUB -alloc_flags "gpumps"
#BSUB -o job%J.out
#BSUB -e job%J.out
###############################################################

# Remote project path
export USER=acmwhb
export PROJID=bip198
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
jsrun -bpacked:7 -g6 -a6 -c42 -r1 python lit_image_classifier.py --num_epochs 25 --batch_size 16384 --hidden_dim 128 --num_dataloader_workers 28
date
