#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=moral_stance_update       #Set the job name to "moral_stance_update"
#SBATCH --time=6:00:00              #Set the wall clock limit to 6 hours
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=32G                  #Request 32GB per node
#SBATCH --output=moral_stance_update.%j      #Send stdout/err to "moral_stance_update.[jobID]"
#SBATCH --gres=gpu:h100:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

#First Executable Line
conda activate INL
module load WebProxy/0000
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN_HERE
CUDA_VISIBLE_DEVICES=0 python StoryGeneration/moral_stance_update.py --model llama