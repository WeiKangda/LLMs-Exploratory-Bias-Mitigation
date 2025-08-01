# MMLU Benchmark

## Overview

The **Massive Multitask Language Understanding (MMLU)** [https://arxiv.org/abs/2009.03300] benchmark evaluates language models across 57 academic subjects including STEM, humanities, social sciences, and more. This implementation provides a comprehensive evaluation framework for assessing model performance on knowledge-intensive tasks.

## Dataset

The MMLU dataset consists of multiple-choice questions covering:
- **STEM**: Mathematics, Physics, Chemistry, Biology, Computer Science
- **Humanities**: History, Literature, Philosophy, Art
- **Social Sciences**: Psychology, Sociology, Economics, Law
- **Professional**: Medicine, Engineering, Business
- **Other**: General knowledge, current events

Each question has 4 answer choices (A, B, C, D) with one correct answer.

## Usage

### Basic Usage

```bash
python mmlu_benchmark.py --model_name "meta-llama/Llama-3.1-8B-Instruct"
```

### Advanced Options

```bash
python mmlu_benchmark.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --subjects "mathematics" "physics" "chemistry" \
    --batch_size 16 \
    --output_dir "./my_results" \
    --debug
```

### Command Line Arguments

- `--model_name`: Model identifier (default: meta-llama/Llama-3.1-8B-Instruct)
- `--subjects`: Specific subjects to evaluate (default: all 57 subjects)
- `--batch_size`: Batch size for evaluation (default: 8)
- `--output_dir`: Results directory (default: ./results)
- `--cache_dir`: Model cache directory (default: ./models)
- `--split`: Dataset split to use (default: test)
- `--debug`: Enable debug output

## Output

Results are saved in the following structure:
```
results/
├── mmlu_results/
│   └── [model_name]/
│       └── results_[timestamp].json
└── mmlu_logs/
    └── [model_name]/
        └── benchmark_[timestamp].log
```

The JSON results include:
- Overall accuracy across subjects
- Per-subject accuracy scores
- Detailed question-by-question analysis
- Model predictions vs. correct answers

## Example Results

```
mathematics: 45.23%
physics: 52.18%
chemistry: 48.91%
...

Average accuracy across all subjects: 47.85%
```