#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=evaluate_genmo_icl       #Set the job name to "evaluate_genmo_icl"
#SBATCH --time=2:00:00              #Set the wall clock limit to 2 hours
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=32G                  #Request 32GB per node
#SBATCH --output=evaluate_genmo_icl.%j      #Send stdout/err to "evaluate_genmo_icl.[jobID]"
#SBATCH --gres=gpu:h100:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

#First Executable Line
conda activate genderbias_env
module load WebProxy/0000
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN_HERE
CUDA_VISIBLE_DEVICES=0 python Benchmarks/GenMO/evaluate_genmo_icl.py