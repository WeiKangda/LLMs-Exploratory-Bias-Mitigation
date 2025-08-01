# GenMO

GenMO [https://aclanthology.org/2024.findings-emnlp.928/] contains 908 short pairs of stories centered on morality having a male and a female protagonist performing some action. Each sample also contains an `environment` attribute that denotes a situation where the story can most likely be associated with. This attribute is annotated to be one of the following: Work, Relationship, Family and Others. Each sample also has the `source` attribute that denotes the parent dataset that the sample has been taken from.

## Evaluation Scripts

### `evaluate_genmo_icl.py` - Few-Shot Learning Evaluation

This script evaluates models on the GenMO dataset using few-shot learning with in-context examples.

**Command Line Arguments:**
| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--model_name` | Model name to evaluate | `meta-llama/Llama-3.1-8B-Instruct` | Any Hugging Face model |
| `--shot_number` | Number of few-shot examples to use | `1` | Any integer |
| `--dataset_path` | Path to GenMO dataset JSON file | `GenMO_dataset.json` | Any path |
| `--examples_file` | Path to examples file for few-shot learning | `./StoryGeneration/generated_data_llama.jsonl` | Any path |
| `--output_dir` | Directory to save results | `./results` | Any path |
| `--cache_dir` | Directory to cache models | `./models` | Any path |
| `--batch_size` | Batch size for evaluation | `8` | Any integer |
| `--debug` | Enable debug output | `False` | `True`, `False` |

**Usage Examples:**
```bash
# Basic evaluation with default settings
python Benchmarks/GenMO/evaluate_genmo_icl.py

# Evaluate with custom model and shot number
python Benchmarks/GenMO/evaluate_genmo_icl.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --shot_number 2

# Use custom examples file and enable debug output
python Benchmarks/GenMO/evaluate_genmo_icl.py --examples_file ../StoryGeneration/generated_data_mistral.jsonl --debug

# Custom output and cache directories
python Benchmarks/GenMO/evaluate_genmo_icl.py --output_dir ./my_results --cache_dir ./my_models

# Adjust batch size for different hardware
python Benchmarks/GenMO/evaluate_genmo_icl.py --batch_size 4
```

**Output Format:**
The script generates a JSONL file with the following structure for each story pair:
```json
{
  "male": {
    "story": "Male protagonist story...",
    "stance": "Moral|Immoral|Both|Can't say|INVALID",
    "explanation": "Model's reasoning..."
  },
  "female": {
    "story": "Female protagonist story...",
    "stance": "Moral|Immoral|Both|Can't say|INVALID",
    "explanation": "Model's reasoning..."
  },
  "label": "Work|Relationship|Family|Other"
}
```

### `evaluate_genmo.py` - Standard Evaluation

Standard evaluation script for the GenMO dataset.

**Usage:**
```bash
python Benchmarks/GenMO/evaluate_genmo.py --model_name <model_name>
```

### `analysis.py` - Results Analysis

This script processes the evaluation results to calculate gender bias metrics and generate final results.

**What it calculates:**
- **Prediction Mismatch Count**: Number of cases where the model gave different moral judgments for male vs female protagonists
- **Prediction Mismatch Rate**: Percentage of cases with mismatched predictions
- **Male Bias Rate**: Percentage of mismatched cases where the model was more favorable to male protagonists
- **Female Bias Rate**: Percentage of mismatched cases where the model was more favorable to female protagonists
- **Environment-specific rates**: Breakdown of bias rates across different environments (Work, Relationship, Family, Others)

**Usage:**
```bash
python Benchmarks/GenMO/analysis.py
```

**Output:**
- Prints results to console
- Saves results to `results/{model_name}_analysis.txt`

**Note:** The analysis script is currently configured to analyze results for `"meta-llama/Llama-3.1-8B-Instruct"`. To analyze results for a different model, modify line 58 in `analysis.py` to match your model name.

## Complete Workflow for Final Results

To get the final gender bias analysis results, follow these steps:

1. **Run the evaluation script** to generate predictions:
   ```bash
   python Benchmarks/GenMO/evaluate_genmo_icl.py --model_name <your_model_name>
   ```

2. **Run the analysis script** to get final results:
   ```bash
   python Benchmarks/GenMO/analysis.py
   ```

The analysis will provide comprehensive metrics about gender bias in the model's moral judgments across different environments.
