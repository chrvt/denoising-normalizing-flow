#!/bin/bash

#SBATCH --mail-user=<horvat@pyl.unibe.ch>
#SBATCH --mail-type=fail,end
#SBATCH --job-name="mf_gan2d_eval_test"
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4 #4
#SBATCH --mem=32G  #32

#SBATCH --partition=gpu
#SBATCH --qos=job_gpu

#SBATCH --gres=gpu:gtx1080ti:1  

cd /storage/homefs/ch19g182/Python/Denoising-Normalizing-Flow-master/experiments

python evaluate.py --modelname lorenz_setting_epoch --dataset thin_spiral --algorithm mf --evalbatchsize 25 --modellatentdim 1 --truelatentdim 1 --sig2 0 --outerlayers 9 --innerlayers 1 --lineartransform permutation --splinerange 3.0 --splinebins 5 --dir /storage/homefs/ch19g182/Python/Denoising-Normalizing-Flow-master
