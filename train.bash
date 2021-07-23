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
export PROJDIR="$PWD"

# Remote Conda environment
module load open-ce/1.1.3-py38-0
conda activate "$PROJDIR"/venv

# Configure OMP for PyTorch
export OMP_PLACES=threads

# Run training script
cd "$PROJDIR"/project || exit

# Execute script
date
jsrun -bpacked:7 -g6 -a6 -c42 -r1 python lit_image_classifier.py --num_epochs 25 --batch_size 16384 --hidden_dim 128 --num_dataloader_workers 28
date
