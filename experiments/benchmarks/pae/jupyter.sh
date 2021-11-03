#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=jupyter-notebook
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu
#SBATCH --gres=gpu:gtx1080ti:1

# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="ubelix"
port=8889

# print tunneling instructions jupyter-log       
echo -e "
Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@$ submit.unibe.ch  

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
module load cuDNN/7.6.0.64-gcccuda-2019a

# Run Jupyter
jupyter-notebook --no-browser --port=${port} --ip=${node}