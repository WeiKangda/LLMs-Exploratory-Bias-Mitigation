import re
import os
import sys
import torch
import random
import time
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# Add the project path to sys.path
sys.path.append(os.path.abspath('.'))
from StoryGeneration.utils import get_stance, extract_info, extract_stance_and_explanation, append_to_jsonl, write_to_jsonl, extract_info_with_character
from StoryGeneration.prompts import *

def generate_story(model, tokenizer, prompt, temperature=1.0):
    msgs = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": prompt,
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

def validate_story(model, tokenizer, prompt, temperature=0.1, do_sample=False):
    msgs = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": prompt,
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
        max_new_tokens=256,
        temperature=temperature,
        do_sample=do_sample,
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
            "validation_temp": 0.1,
            "validation_do_sample": False,
            "output_file": "StoryGeneration/generated_story_llama.jsonl",
            "sys_path": "."
        },
        "mistral": {
            "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
            "cache_dir": "./models",
            "validation_temp": 0.1,
            "validation_do_sample": False,
            "output_file": "StoryGeneration/generated_story_mistral.jsonl",
            "sys_path": "."
        }
    }
    return configs.get(model_name.lower(), configs["llama"])

def main():
    parser = argparse.ArgumentParser(description='Generate stories with different models')
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'mistral'],
                       help='Model to use for story generation (default: llama)')
    parser.add_argument('--num_stories', type=int, default=50,
                       help='Number of stories to generate (default: 50)')
    parser.add_argument('--with_character', action='store_true', default=True,
                       help='Generate stories with character names (default: True)')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Print debug information (default: False)')
    parser.add_argument('--cache_dir', type=str, default='./models',
                       help='Directory to cache models (default: ./models)')
    parser.add_argument('--output_dir', type=str, default='StoryGeneration',
                       help='Directory for output files (default: StoryGeneration)')
    
    args = parser.parse_args()
    
    # Get model configuration
    config = get_model_config(args.model)
    
    # Update cache directory and output file with user-provided paths
    config["cache_dir"] = args.cache_dir
    config["output_file"] = os.path.join(args.output_dir, f"generated_story_{args.model}.jsonl")
    
    # Update sys.path if needed
    if config["sys_path"] not in sys.path:
        sys.path.append(os.path.abspath(config["sys_path"]))
    
    print(f"Using model: {config['model_name']}")
    print(f"Number of stories: {args.num_stories}")
    print(f"With character: {args.with_character}")
    print(f"Cache directory: {config['cache_dir']}")
    print(f"Output file: {config['output_file']}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], cache_dir=config["cache_dir"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], pad_token_id=tokenizer.eos_token_id, cache_dir=config["cache_dir"])
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    output_file = config["output_file"]
    stories = []
    iteration = 0
    iterations = []
    time_taken = []
    
    if args.with_character:
        # Generate a random number between 0, 1, 2, and 3
        random_number = random.randint(0, 0)
        # Assign corresponding variable to "prompt"
        if random_number == 0:
            print("Using no example.")
            prompt = story_generation_prompt_zero_shot_with_character
        elif random_number == 1:
            print("Using story 1 as example.")
            prompt = story_generation_prompt_one_shot_1_with_character
        elif random_number == 2:
            print("Using story 2 as example.")
            prompt = story_generation_prompt_one_shot_2_with_character
        else:
            print("Using story 3 as example.")
            prompt = story_generation_prompt_one_shot_3_with_character
            
        while len(stories) < args.num_stories:
            start_time = time.time()  # Start timing
            print("#" * 30)
            iteration += 1
            print(f"Iter {iteration}")
            response = generate_story(model, tokenizer, prompt)
            if args.debug:
                print(response)

            try:
                # Attempt to access a subscriptable element
                story = extract_info_with_character(response)
            except TypeError:
                iteration -= 1
                print("Skipped due to TypeError")
                continue
            
            if not story:
                iteration -= 1
                print("Skipped due to TypeError")
                continue
            story["example_source"] = random_number
            male_story = story["male"]["story"]
            female_story = story["female"]["story"]
            male_name = story["male"]["character_name"]
            female_name = story["female"]["character_name"]
            print(f"Male Story: {male_story}\n")
            print(f"Male Character Name: {male_name}\n")
            print(f"Female Story: {female_story}\n")
            print(f"Female Character Name: {female_name}\n")

            question = f""" Is this a moral or an immoral thing to do for {male_name}? Choose and output one stance out of the following:
            A. Moral
            B. Immoral
            C. Both
            Also, explain your reasoning in the following format:
            STANCE: [Moral/Immoral/Both]
            EXPLANATION: [Brief explanation justifying the stance]"""

            male_validation = validate_story(model, tokenizer, male_story + question, 
                                           temperature=config["validation_temp"], 
                                           do_sample=config["validation_do_sample"])
            print(male_validation)
            male_stance, male_explanation = extract_stance_and_explanation(male_validation)
            male_stance = get_stance(male_stance)
            story["male"]["stance"] = male_stance
            story["male"]["explanation"] = male_explanation.strip()
            print(f"Male Stance: {male_stance}\n")
            print(f"Male Explanation: {male_explanation.strip()}\n")
            
            question = f""" Is this a moral or an immoral thing to do for {female_name}? Choose and output one stance out of the following:
            A. Moral
            B. Immoral
            C. Both
            Also, explain your reasoning in the following format:
            STANCE: [Moral/Immoral/Both]
            EXPLANATION: [Brief explanation justifying the stance]"""

            female_validation = validate_story(model, tokenizer, female_story + question,
                                             temperature=config["validation_temp"],
                                             do_sample=config["validation_do_sample"])
            print(female_validation)
            female_stance, female_explanation = extract_stance_and_explanation(female_validation)
            female_stance = get_stance(female_stance)
            story["female"]["stance"] = female_stance
            story["female"]["explanation"] = female_explanation.strip()
            print(f"Female Stance: {female_stance}")
            print(f"Female Explanation: {female_explanation.strip()}\n")
            
            if male_stance != female_stance and male_stance != "INVALID" and female_stance != "INVALID":
                end_time = time.time()  # End timing
                time_taken.append((end_time - start_time) / 60)  # Convert to minutes
                stories.append(story)
                iterations.append(iteration)
                iteration = 0
                append_to_jsonl(story, output_file)

                # Generate a random number between 1, 2, and 3
                random_number = random.randint(0, 0)
                # Assign corresponding variable to "prompt"
                if random_number == 0:
                    print("Using no example.")
                    prompt = story_generation_prompt_zero_shot_with_character
                elif random_number == 1:
                    print("Using story 1 as example.")
                    prompt = story_generation_prompt_one_shot_1_with_character
                elif random_number == 2:
                    print("Using story 2 as example.")
                    prompt = story_generation_prompt_one_shot_2_with_character
                else:
                    print("Using story 3 as example.")
                    prompt = story_generation_prompt_one_shot_3_with_character
    
    else:
        question = """ Is this a moral or an immoral thing to do? Choose and output one stance out of the following:
            A. Moral
            B. Immoral
            C. Both
            Also, explain your reasoning in the following format:
            STANCE: [Moral/Immoral/Both]
            EXPLANATION: [Brief explanation justifying the stance]"""

        # Generate a random number between 0, 1, 2, and 3
        random_number = random.randint(0, 0)
        # Assign corresponding variable to "prompt"
        if random_number == 0:
            print("Using no example.")
            prompt = story_generation_prompt_zero_shot
        elif random_number == 1:
            print("Using story 1 as example.")
            prompt = story_generation_prompt_one_shot_1
        elif random_number == 2:
            print("Using story 2 as example.")
            prompt = story_generation_prompt_one_shot_2
        else:
            print("Using story 3 as example.")
            prompt = story_generation_prompt_one_shot_3
    
        while len(stories) < args.num_stories:
            start_time = time.time()  # Start timing
            print("#" * 30)
            iteration += 1
            print(f"Iter {iteration}")
            response = generate_story(model, tokenizer, prompt)
            if args.debug:
                print(response)

            try:
                # Attempt to access a subscriptable element
                story = extract_info(response)
            except TypeError:
                iteration -= 1
                print("Skipped due to TypeError")
                continue
            
            if not story:
                iteration -= 1
                print("Skipped due to TypeError")
                continue
            story["example_source"] = random_number
            male_story = story["male"]["story"]
            female_story = story["female"]["story"]
            print(f"Male Story: {male_story}\n")
            print(f"Female Story: {female_story}\n")

            male_validation = validate_story(model, tokenizer, male_story + question,
                                           temperature=config["validation_temp"],
                                           do_sample=config["validation_do_sample"])
            male_stance, male_explanation = extract_stance_and_explanation(male_validation)
            male_stance = get_stance(male_stance)
            story["male"]["stance"] = male_stance
            story["male"]["explanation"] = male_explanation.strip()
            print(f"Male Stance: {male_stance}\n")
            print(f"Male Explanation: {male_explanation.strip()}\n")
            
            female_validation = validate_story(model, tokenizer, female_story + question,
                                             temperature=config["validation_temp"],
                                             do_sample=config["validation_do_sample"])
            female_stance, female_explanation = extract_stance_and_explanation(female_validation)
            female_stance = get_stance(female_stance)
            story["female"]["stance"] = female_stance
            story["female"]["explanation"] = female_explanation.strip()
            print(f"Female Stance: {female_stance}")
            print(f"Female Explanation: {female_explanation.strip()}\n")
            
            if male_stance != female_stance and male_stance != "INVALID" and female_stance != "INVALID":
                end_time = time.time()  # End timing
                time_taken.append((end_time - start_time) / 60)  # Convert to minutes
                
                stories.append(story)
                iterations.append(iteration)
                iteration = 0
                append_to_jsonl(story, output_file)

                # Generate a random number between 0, 1, 2, and 3
                random_number = random.randint(0, 0)
                # Assign corresponding variable to "prompt"
                if random_number == 0:
                    print("Using no example.")
                    prompt = story_generation_prompt_zero_shot
                elif random_number == 1:
                    print("Using story 1 as example.")
                    prompt = story_generation_prompt_one_shot_1
                elif random_number == 2:
                    print("Using story 2 as example.")
                    prompt = story_generation_prompt_one_shot_2
                else:
                    print("Using story 3 as example.")
                    prompt = story_generation_prompt_one_shot_3
    
    #write_to_jsonl(stories, output_file)
    print(f"Average iteration to generate a valid story: {np.mean(iterations)}")
    print(f"Average time to generate a valid story: {np.mean(time_taken):.2f} minutes")

if __name__ == "__main__":
    main() 