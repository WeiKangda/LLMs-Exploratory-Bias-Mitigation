# Gender Bias in Language Models Research

This repository contains research on gender bias in language models, including bias detection, mitigation techniques, and evaluation across multiple benchmarks.

## Project Structure

```
├── Benchmarks/           # Evaluation benchmarks
│   ├── BBQ/             # BBQ benchmark for bias evaluation
│   ├── GenMO/           # GenMO moral reasoning dataset
│   ├── MMLU/            # Massive Multitask Language Understanding
│   ├── TruthfulQA/      # TruthfulQA benchmark
│   └── WinoBias/        # WinoBias benchmark
├── GenderBiasMitigation/ # Bias mitigation techniques
│   ├── dpo.py           # Direct Preference Optimization
│   ├── fine_tune.py     # Standard fine-tuning
│   ├── fine_tune_cda.py # Counterfactual Data Augmentation
│   └── resolution_analysis.py
├── StoryGeneration/     # Story generation and analysis
│   ├── story_generation.py
│   ├── story_generation_mistral.py
│   ├── swap_and_rewrite_explanations.py
│   ├── calculate_story_similarity.py
│   ├── moral_stance_update.py
│   ├── moral_stance_update_mistral.py
│   ├── prompts.py
│   ├── utils.py
│   ├── filter.py
│   ├── generated_data.jsonl
│   ├── generated_data_mistral.jsonl
│   ├── swapped_explanations.jsonl
│   └── swapped_explanations_mistral.jsonl
└── script/              # Slurm scripts for HPC
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/WeiKangda/LLMs-Exploratory-Bias-Mitigation.git
cd LLMs-Exploratory-Bias-Mitigation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Important**: If you plan to use the Slurm scripts, you'll need to:
   - Get a Hugging Face access token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Replace `YOUR_HUGGINGFACE_TOKEN_HERE` in the slurm script files with your actual token

## Usage

### Story Generation

#### Generate Stories
```bash
python StoryGeneration/story_generation.py --model_name <model_name>
```

#### Swap and Rewrite Explanations
```bash
python StoryGeneration/swap_and_rewrite_explanations.py --input_file <input_file>
```

#### Calculate Story Similarity
```bash
python StoryGeneration/calculate_story_similarity.py --input_file <input_file>
```

### Bias Mitigation

#### Direct Preference Optimization (DPO)
```bash
python GenderBiasMitigation/dpo.py --model_name <model_name> --output_dir <output_dir>
```

#### Fine-tuning
```bash
python GenderBiasMitigation/fine_tune.py --model_name <model_name> --output_dir <output_dir>
```

#### Counterfactual Data Augmentation (CDA)
```bash
python GenderBiasMitigation/fine_tune_cda.py --model_name <model_name> --output_dir <output_dir>
```

### Benchmark Evaluation

#### WinoBias
```bash
python Benchmarks/WinoBias/winobias_benchmark.py --model_name <model_name>
```

#### MMLU
```bash
python Benchmarks/MMLU/mmlu_benchmark.py --model_name <model_name>
```

#### TruthfulQA
```bash
python Benchmarks/TruthfulQA/truthfulqa_benchmark.py --model_name <model_name>
```

#### BBQ
```bash
python Benchmarks/BBQ/bbq_benchmark.py --model_name <model_name>
```

#### GenMO
```bash
python Benchmarks/GenMO/evaluate_genmo.py --model_name <model_name>
```

## Data

This repository includes the following data files:
- `StoryGeneration/generated_data.jsonl` - Generated stories with male protagonists
- `StoryGeneration/generated_data_mistral.jsonl` - Generated stories with male protagonists (Mistral model)
- `StoryGeneration/swapped_explanations.jsonl` - Swapped explanations for gender bias analysis
- `StoryGeneration/swapped_explanations_mistral.jsonl` - Swapped explanations (Mistral model)

The data files contain the research datasets used for gender bias analysis and story generation experiments.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{wei2025mitigatinggenderbiasfostering,
      title={Mitigating Gender Bias via Fostering Exploratory Thinking in LLMs}, 
      author={Kangda Wei and Hasnat Md Abdullah and Ruihong Huang},
      year={2025},
      eprint={2505.17217},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.17217}, 
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub or contact the authors at kangda@tamu.edu. 