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
jsrun -bpacked:7 -g6 -a6 -c42 -r1 python lit_image_classifier.py --num_gpus 6 --num_compute_nodes 2 --num_epochs 25 --batch_size 16384 --hidden_dim 128 --num_dataloader_workers 28 --tb_log_dir /mnt/bb/"$USER"/tb_logs --ckpt_dir /mnt/bb/"$USER"/checkpoints
date

# Copying leftover items from NVMe drive back to GPFS
echo "Copying log files and best checkpoint(s) back to GPFS..."
jsrun -n 1 cp -r /mnt/bb/"$USER"/tb_logs "$PROJDIR"/project/tb_logs
jsrun -n 1 cp -r /mnt/bb/"$USER"/checkpoints "$PROJDIR"/project/checkpoints
echo "Done copying log files and best checkpoint(s) back to GPFS"
