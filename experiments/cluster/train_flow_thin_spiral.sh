#!/bin/bash

#SBATCH --mail-user=<horvat@pyl.unibe.ch>
#SBATCH --mail-type=fail,end
#SBATCH --job-name="flow_thin_spiral_train"
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4 #4
#SBATCH --mem=32G  #32

#SBATCH --partition=gpu
#SBATCH --qos=job_gpu

#SBATCH --gres=gpu:gtx1080ti:1  

cd /storage/homefs/ch19g182/Python/Denoising-Normalizing-Flow-master/experiments

nvcc --version
nvidia-smi

python train.py  -c configs/train_flow_thin_spiral.config 