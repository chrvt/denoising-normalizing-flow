#!/bin/bash

#SBATCH --mail-user=<horvat@pyl.unibe.ch>
#SBATCH --mail-type=fail,end
#SBATCH --job-name="dnf_thin_spiral_train_sig0"
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4 #4
#SBATCH --mem=32G  #32

#SBATCH --partition=gpu
#SBATCH --qos=job_gpu

#SBATCH --gres=gpu:gtx1080ti:1 

#SBATCH --array=1-10 

cd /storage/homefs/ch19g182/Python/Denoising-Normalizing-Flow-master/experiments

python train_seeds.py --dir /storage/homefs/ch19g182/Python/Denoising-Normalizing-Flow-master --modelname new_sig0 --dataset thin_spiral --algorithm dnf --modellatentdim 1 --truelatentdim 1 --sig2 0.0 --outertransform rq-coupling --innertransform affine-autoregressive --outerlayers 9 --innerlayers 1 --levels 1 --linlayers 2 --linchannelfactor 1 --lineartransform lu --splinerange 5.0 --splinebins 10 --batchnorm --dropout 0.0 --epochs 100 --batchsize 100 --lr 3.0e-4 --msefactor 1.0 --nllfactor 1. --uvl2reg 0.01 --weightdecay 1.0e-5 --clip 5.0 --validationsplit 0.1