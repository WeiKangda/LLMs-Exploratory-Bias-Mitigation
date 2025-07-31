#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=mmlu_benchmark       #Set the job name to "mmlu_benchmark"
#SBATCH --time=4:00:00              #Set the wall clock limit to 4 hours
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=32G                  #Request 32GB per node
#SBATCH --output=mmlu_benchmark.%j      #Send stdout/err to "mmlu_benchmark.[jobID]"
#SBATCH --gres=gpu:h100:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

#First Executable Line
conda activate INL
module load WebProxy/0000
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN_HERE
CUDA_VISIBLE_DEVICES=0 python Benchmarks/MMLU/mmlu_benchmark.py --model_name meta-llama/Llama-3.1-8B-Instruct 