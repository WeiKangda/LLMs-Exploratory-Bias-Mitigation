#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=dpo_training       #Set the job name to "dpo_training"
#SBATCH --time=6:00:00              #Set the wall clock limit to 6 hours
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=32G                  #Request 32GB per node
#SBATCH --output=dpo_training.%j      #Send stdout/err to "dpo_training.[jobID]"
#SBATCH --gres=gpu:h100:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

#First Executable Line
conda activate genderbias_env
module load WebProxy/0000
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN_HERE
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python GenderBiasMitigation/dpo.py --model_name meta-llama/Llama-3.1-8B-Instruct --max_examples 125