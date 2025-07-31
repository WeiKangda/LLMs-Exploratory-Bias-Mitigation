#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=finetune_cda       #Set the job name to "finetune_cda"
#SBATCH --time=6:00:00              #Set the wall clock limit to 6 hours
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=32G                  #Request 32GB per node
#SBATCH --output=finetune_cda.%j      #Send stdout/err to "finetune_cda.[jobID]"
#SBATCH --gres=gpu:h100:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

#First Executable Line
conda activate INL
module load WebProxy/0000
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN_HERE
CUDA_VISIBLE_DEVICES=0 python GenderBiasMitigation/fine_tune_cda.py