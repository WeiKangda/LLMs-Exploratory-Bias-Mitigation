import re
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
sys.path.append(os.path.abspath('/scratch/user/u.kw178339/GenderBias'))
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

if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/scratch/user/u.kw178339/huggingface_models")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id, cache_dir="/scratch/user/u.kw178339/huggingface_models")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    generated_data = read_from_jsonl("./StoryGeneration/generated_story_filtered_mistral.jsonl")
    output_file = "./StoryGeneration/generated_data_mistral.jsonl"
    
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

        neutral_explanation = update_morality(model, tokenizer, male_story, male_character, male_stance, male_explanation, female_story, female_character, female_stance, female_explanation)
        updated_male_explanation, updated_female_explanation = extract_updated_explanations(neutral_explanation)
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

