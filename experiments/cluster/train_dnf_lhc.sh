#!/bin/bash

#SBATCH --mail-user=<horvat@pyl.unibe.ch>
#SBATCH --mail-type=fail,end
#SBATCH --job-name="dnf_lhc"
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4 #4
#SBATCH --mem=32G  #32

#SBATCH --partition=gpu
#SBATCH --qos=job_gpu

#SBATCH --gres=gpu:gtx1080ti:1  

cd /storage/homefs/ch19g182/Python/Denoising-Normalizing-Flow-master/experiments

python train.py --dir /storage/homefs/ch19g182/Python/Denoising-Normalizing-Flow-master --modelname original_setting --dataset lhc40d --algorithm dnf --modellatentdim 14 --sig2 0.01 --innertransform rq-coupling --outertransform rq-coupling  --outerlayers 20 --innerlayers 15 --lineartransform lu --splinerange 10.0 --splinebins 11 --dropout 0.0 --epochs 50 --batchsize 100 --lr 3.0e-4 --msefactor 1.0 --nllfactor 1. --uvl2reg 0.01 --weightdecay 1.0e-5 --clip 10.0 --validationsplit 0.25 
