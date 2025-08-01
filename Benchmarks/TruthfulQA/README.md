# TruthfulQA Benchmark

## Overview

TruthfulQA [https://aclanthology.org/2022.acl-long.229/] is a benchmark designed to evaluate whether language models can answer questions truthfully and avoid common misconceptions. The dataset consists of 817 questions that test a model's ability to distinguish between true and false statements across various domains including health, science, history, and popular culture.

## Dataset Structure

The dataset (`mc_task.json`) contains multiple-choice questions with three different evaluation formats:

- **MC0**: Binary choice between a correct and incorrect answer
- **MC1**: Multiple choice with one correct answer and several incorrect distractors
- **MC2**: Multiple choice with multiple correct answers and incorrect distractors
- We only evaluted on MC0 and MC1

Each question includes:
- A question text
- Multiple choice options with binary labels (1 = correct, 0 = incorrect)
- Different sets of choices for each evaluation format

### Running the Benchmark

```bash
python truthfulqa_benchmark.py --model_name "meta-llama/Llama-3.1-8B-Instruct"
```

### Command Line Arguments

- `--model_name`: Model to evaluate (default: meta-llama/Llama-3.1-8B-Instruct)
- `--cache_dir`: Directory to cache models (default: ./models)
- `--output_dir`: Output directory for results (default: ./results)
- `--batch_size`: Batch size for evaluation (default: 8)
- `--debug`: Enable debug output

### Example

```bash
python truthfulqa_benchmark.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --batch_size 4 \
    --debug
```

## Output

The benchmark generates:
- **Results**: JSON files with accuracy scores and detailed results
- **Logs**: Timestamped log files with evaluation progress
- **Console output**: Summary of MC0 and MC1 accuracy scores

Results are saved in `./results/truthfulqa_results/{model_name}/` with timestamps.

## Evaluation Metrics

- **MC0 Accuracy**: Binary choice accuracy (correct vs. incorrect)
- **MC1 Accuracy**: Multiple choice accuracy with single correct answer

The benchmark evaluates how well models can:
- Distinguish between factual and false information
- Avoid common misconceptions and urban legends
- Provide truthful answers to questions about science, history, and culture