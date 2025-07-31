import re
import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
import random
sys.path.append(os.path.abspath('.'))
from StoryGeneration.utils import append_to_jsonl, read_from_jsonl, write_to_jsonl
import nltk
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

nltk.download('punkt')

def tokenize(text):
    return word_tokenize(text.lower())

def compute_rouge1(ref, cand):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    return scorer.score(ref, cand)['rouge1'].fmeasure

def filter_pairs(pairs, rouge1_lower_thresh=0.8, rouge1_upper_thresh=0.95, debug=False):
    filtered = []
    discared = []
    source_stats = {0: 0, 1: 0, 2: 0, 3: 0}
    missing_source = 0
    for pair in pairs:
        if "example_source" not in pair:
            pair["example_source"] = random.randint(1, 3)
            missing_source_flag = True
        else:
            missing_source_flag = False
        
        ref, cand = pair["male"]["story"], pair["female"]["story"]
        rouge1 = compute_rouge1(ref, cand)
        if debug:
            print("#" * 30)
            print(f"ROUGE-1 score: {rouge1}")
        
        if rouge1 >= rouge1_lower_thresh and rouge1 <= rouge1_upper_thresh:
            filtered.append(pair)
            source_stats[pair["example_source"]] += 1
            if missing_source_flag:
                missing_source += 1
                missing_source_flag = False
        else:
            discared.append(pair)
            if debug:
                print(f"Male Story: {ref}\n")
                print(f"Male Character Name: {pair['male']['character_name']}\n")
                print(f"Male Stance: {pair['male']['stance']}\n")
                print(f"Male Explanation: {pair['male']['explanation']}\n")
                print(f"Female Story: {cand}")
                print(f"Female Character Name: {pair['female']['character_name']}\n")
                print(f"Female Stance: {pair['female']['stance']}\n")
                print(f"Female Explanation: {pair['female']['explanation']}\n")
            

    return filtered, discared, source_stats, missing_source

def get_model_config(model_name):
    """Return model-specific configuration parameters"""
    configs = {
        "llama": {
            "input_file": "./StoryGeneration/generated_story_llama.jsonl",
            "output_file": "./StoryGeneration/generated_story_filtered_llama.jsonl"
        },
        "mistral": {
            "input_file": "./StoryGeneration/generated_story_mistral.jsonl",
            "output_file": "./StoryGeneration/generated_story_filtered_mistral.jsonl"
        }
    }
    return configs.get(model_name.lower(), configs["llama"])

def main():
    parser = argparse.ArgumentParser(description='Filter story pairs based on ROUGE-1 similarity')
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'mistral'],
                       help='Model to use for filtering (default: llama)')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Input file path (default: model-specific)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file path (default: model-specific)')
    parser.add_argument('--rouge1_lower', type=float, default=0.8,
                       help='Lower threshold for ROUGE-1 score (default: 0.8)')
    parser.add_argument('--rouge1_upper', type=float, default=0.95,
                       help='Upper threshold for ROUGE-1 score (default: 0.95)')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Print debug information (default: False)')
    
    args = parser.parse_args()
    
    # Get model configuration
    config = get_model_config(args.model)
    
    # Update paths with user-provided values if specified
    if args.input_file:
        config["input_file"] = args.input_file
    if args.output_file:
        config["output_file"] = args.output_file
    
    print(f"Using model: {args.model}")
    print(f"Input file: {config['input_file']}")
    print(f"Output file: {config['output_file']}")
    print(f"ROUGE-1 thresholds: {args.rouge1_lower} - {args.rouge1_upper}")
    print(f"Debug mode: {args.debug}")

    # Check if input file exists
    if not os.path.exists(config["input_file"]):
        print(f"Error: Input file {config['input_file']} does not exist.")
        print("Please run story generation first to create the input file.")
        return

    generated_story = read_from_jsonl(config["input_file"])
    print(f"Loaded {len(generated_story)} stories from {config['input_file']}")

    filtered_stories, discared_storied, source_stats, missing_source = filter_pairs(
        generated_story, 
        rouge1_lower_thresh=args.rouge1_lower, 
        rouge1_upper_thresh=args.rouge1_upper,
        debug=args.debug
    )

    print(f"\nFiltering Results:")
    print(f"Keep {len(filtered_stories)} stories.")
    print(f"Discard {len(discared_storied)} stories.")
    print(f"Missing source: {missing_source}")
    print("\nExample Source Statistics:")
    for source, count in source_stats.items():
        print(f"Source {source}: {count} pairs")
    
    write_to_jsonl(filtered_stories, config["output_file"])
    print(f"\nFiltered stories saved to: {config['output_file']}")

if __name__ == "__main__":
    main()