#!/bin/bash

#SBATCH --mail-user=<horvat@pyl.unibe.ch>
#SBATCH --mail-type=fail,end
#SBATCH --job-name="dnf_celeba"
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4 #4
#SBATCH --mem=32G  #32

#SBATCH --partition=gpu
#SBATCH --qos=job_gpu

#SBATCH --gres=gpu:gtx1080ti:1   

cd /storage/homefs/ch19g182/Python/Denoising-Normalizing-Flow-master/experiments

nvcc --version
nvidia-smi

python train.py -c configs/train_dnf_celeba.config