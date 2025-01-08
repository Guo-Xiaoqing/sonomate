#!/bin/bash
#SBATCH -J bash
#SBATCH -o ./log/job%j_ours.out
#SBATCH -e ./log/error.err
#SBATCH --gres=gpu:1
#SBATCH -w node04
#SBATCH --partition=standard

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

source /users/engs2527/.bashrc

conda activate pulsept
# nvidia-smi
cd /users/engs2527/ultrasound/ours_iv_vqa/train_videoqa/
python -u ./main_vqa.py
# CUDA_VISIBLE_DEVICES=8 python -u ./main_vqa.py
