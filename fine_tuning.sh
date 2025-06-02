#!/usr/bin/bash
#SBATCH --job-name build
#SBATCH --account carlo
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 10:00:00
#SBATCH --error myJobInference.err
#SBATCH --output myJobInference.out

export CUDA_VISIBLE_DEVICES=0
cd FineTuning/
torchrun --nproc_per_node 1 fine_tuning.py --config configs/r50_nuimg_704x256.py
