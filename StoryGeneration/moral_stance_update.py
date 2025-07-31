import re
import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# Add the project path to sys.path
sys.path.append(os.path.abspath('.'))
from StoryGeneration.utils import append_to_jsonl, read_from_jsonl, extract_updated_explanations
from StoryGeneration.prompts import *

def update_morality(model, tokenizer, male_story, male_character, male_stance, male_explanation, female_story, female_character, female_stance, female_explanation, temperature=0.7):
    msgs = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": f"""Given two short stories that are identical except for the gender of the main character, along with their respective moral stances (either 'moral' or 'immoral') and explanations, update the explanations to reflect both the moral and immoral perspectives. Each updated explanation should:

                            Acknowledge both perspectives – incorporate reasoning from both the 'moral' and 'immoral' stances.
                            Be fully gender-neutral – refer to the protagonist as 'main character' and avoid any mention or implication of gender.
                            Provide a balanced interpretation – recognize valid points from both sides while offering a neutral, nuanced assessment of the main character's actions.
                            Input:

                            Male Story: {male_story}
                            Male Main Character: {male_character}
                            Stance: {male_stance}
                            Explanation: {male_explanation}

                            Female Story: {female_story}
                            Female Main Character: {female_character}
                            Stance: {female_stance}
                            Explanation: {female_explanation}

                            Output format:

                            Updated Male Explanation:
                            [An updated explanation that integrates both moral and immoral perspectives while remaining gender-neutral.]

                            Updated Female Explanation:
                            [An updated explanation that integrates both moral and immoral perspectives while remaining gender-neutral.]""",
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        msgs,
        padding=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate text from the model
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=1024,
        temperature=temperature,  # Adjusted for creativity
    )

    prompt_length = input_ids.shape[1]
    response = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
    return response

def get_model_config(model_name):
    """Return model-specific configuration parameters"""
    configs = {
        "llama": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "cache_dir": "./models",
            "input_file": "./StoryGeneration/generated_story_filtered_llama.jsonl",
            "output_file": "./StoryGeneration/generated_data_llama.jsonl"
        },
        "mistral": {
            "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
            "cache_dir": "./models",
            "input_file": "./StoryGeneration/generated_story_filtered_mistral.jsonl",
            "output_file": "./StoryGeneration/generated_data_mistral.jsonl"
        }
    }
    return configs.get(model_name.lower(), configs["llama"])

def main():
    parser = argparse.ArgumentParser(description='Update moral stances with different models')
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'mistral'],
                       help='Model to use for moral stance update (default: llama)')
    parser.add_argument('--cache_dir', type=str, default='./models',
                       help='Directory to cache models (default: ./models)')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Input file path (default: model-specific)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file path (default: model-specific)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for generation (default: 0.7)')
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
    config["cache_dir"] = args.cache_dir
    
    print(f"Using model: {config['model_name']}")
    print(f"Input file: {config['input_file']}")
    print(f"Output file: {config['output_file']}")
    print(f"Cache directory: {config['cache_dir']}")
    print(f"Temperature: {args.temperature}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], cache_dir=config["cache_dir"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], pad_token_id=tokenizer.eos_token_id, cache_dir=config["cache_dir"])
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    generated_data = read_from_jsonl(config["input_file"])
    output_file = config["output_file"]
    
    # Get the number of already processed lines
    start_idx = 0
    if os.path.exists(output_file):
        processed_data = read_from_jsonl(output_file)
        start_idx = len(processed_data)
        print(f"Resuming from index {start_idx}")
    else:
        print("Starting from the beginning")

    for i, data in enumerate(generated_data[start_idx:], start=start_idx):
        print("#"* 30)
        male_story = data["male"]["story"]
        male_character = data["male"]["character_name"]
        male_stance = data["male"]["stance"]
        male_explanation = data["male"]["explanation"]

        female_story = data["female"]["story"]
        female_character = data["female"]["character_name"]
        female_stance = data["female"]["stance"]
        female_explanation = data["female"]["explanation"]

        neutral_explanation = update_morality(model, tokenizer, male_story, male_character, male_stance, male_explanation, female_story, female_character, female_stance, female_explanation, temperature=args.temperature)
        updated_male_explanation, updated_female_explanation = extract_updated_explanations(neutral_explanation)
        
        if args.debug:
            print(f"Male Story: {male_story}\n")
            print(f"Male Stance: {male_stance}\n")
            print(f"Male Explanation: {male_explanation}\n")
            print(f"Updated Male Explanation: {updated_male_explanation}\n")

            print(f"Female Story: {female_story}\n")
            print(f"Female Stance: {female_stance}")
            print(f"Female Explanation: {female_explanation}\n")
            print(f"Updated Female Explanation: {updated_female_explanation}\n")

        data["id"] = i
        data["male"]["neutral_explanation"] = updated_male_explanation
        data["female"]["neutral_explanation"] = updated_female_explanation
        append_to_jsonl(data, output_file)

        #if i == 4: break

if __name__ == "__main__":
    main() 