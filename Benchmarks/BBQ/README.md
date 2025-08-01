# BBQ (Bias Benchmark for Question Answering) Evaluation

## Overview

The BBQ (Bias Benchmark for Question Answering) [https://aclanthology.org/2022.findings-acl.165/] is a comprehensive benchmark designed to evaluate language models for social biases across multiple demographic categories. This benchmark consists of multiple-choice questions that test for biases related to age, disability status, gender identity, nationality, physical appearance, race/ethnicity, religion, socioeconomic status (SES), and sexual orientation.

## Dataset Description

The BBQ dataset contains 9 categories of bias evaluation, each focusing on a specific demographic dimension:

- **Age**: Questions testing age-related biases
- **Disability_status**: Questions testing biases against people with disabilities
- **Gender_identity**: Questions testing gender identity biases
- **Nationality**: Questions testing nationality-based biases
- **Physical_appearance**: Questions testing appearance-based biases
- **Race_ethnicity**: Questions testing racial and ethnic biases
- **Religion**: Questions testing religious biases
- **SES**: Questions testing socioeconomic status biases
- **Sexual_orientation**: Questions testing sexual orientation biases

Each category contains multiple-choice questions with context, a question, and typically 3 answer choices (A, B, C). The dataset is stored in JSONL format with one example per line.

## Dataset Format

Each example in the dataset follows this structure:

```json
{
  "context": "Context information for the question",
  "question": "The question to be answered",
  "ans0": "First answer choice",
  "ans1": "Second answer choice", 
  "ans2": "Third answer choice",
  "label": 0
}
```

Where:
- `context`: Provides background information for the question
- `question`: The actual question to be answered
- `ans0`, `ans1`, `ans2`: The multiple choice options
- `label`: The index of the correct answer (0, 1, or 2)

## Usage

### Basic Usage

Run the BBQ benchmark evaluation with default settings:

```bash
python bbq_benchmark.py --model_name "meta-llama/Llama-3.1-8B-Instruct"
```

### Command Line Arguments

- `--model_name`: Name or path of the model to evaluate (default: "meta-llama/Llama-3.1-8B-Instruct")
- `--cache_dir`: Directory to cache models (default: "./models")
- `--data_dir`: Directory containing BBQ dataset files (default: "./Benchmarks/BBQ")
- `--output_dir`: Output directory for results (default: "./results")
- `--batch_size`: Batch size for evaluation (default: 8)
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 20)

**Evaluate a local model:**
```bash
python bbq_benchmark.py \
    --model_name "./local_model_path" \
    --cache_dir "./local_cache"
```

## Output and Results

### Results Structure

The evaluation generates results in the following directory structure:

```
results/
├── bbq_results/
│   └── {model_name}/
│       └── results_YYYYMMDD_HHMMSS.json
└── bbq_logs/
    └── {model_name}/
        └── benchmark_YYYYMMDD_HHMMSS.log
```

### Results File Format

The main results file (`results_YYYYMMDD_HHMMSS.json`) contains:

```json
{
  "model": "model_name",
  "average_accuracy": 0.75,
  "total_questions": 1000,
  "category_results": {
    "Age": {
      "accuracy": 0.80,
      "total_questions": 100,
      "results": [...]
    },
    "Gender_identity": {
      "accuracy": 0.70,
      "total_questions": 100,
      "results": [...]
    }
    // ... other categories
  }
}
```