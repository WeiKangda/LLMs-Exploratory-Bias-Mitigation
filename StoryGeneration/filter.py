import re
import os
import sys
import numpy as np
from tqdm import tqdm
import random
sys.path.append(os.path.abspath('/scratch/user/u.kw178339/GenderBias'))
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

def filter_pairs(pairs, rouge1_lower_thresh=0.8, rouge1_upper_thresh=0.95):
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
            print(f"Male Story: {ref}\n")
            print(f"Male Character Name: {pair["male"]["character_name"]}\n")
            print(f"Male Stance: {pair["male"]["stance"]}\n")
            print(f"Male Explanation: {pair["male"]["explanation"]}\n")
            print(f"Female Story: {cand}")
            print(f"Female Character Name: {pair["female"]["character_name"]}\n")
            print(f"Female Stance: {pair["female"]["stance"]}\n")
            print(f"Female Explanation: {pair["female"]["explanation"]}\n")
            

    return filtered, discared, source_stats, missing_source

if __name__ == "__main__":

    generated_story = read_from_jsonl("./StoryGeneration/generated_story_mistral.jsonl")
    output_file = "./StoryGeneration/generated_story_filtered_mistral.jsonl"

    filtered_stories, discared_storied, source_stats, missing_source = filter_pairs(generated_story)

    print(f"Keep {len(filtered_stories)} stories.")
    print(f"Discard {len(discared_storied)} stories.")
    print(f"Missing source: {missing_source}")
    print("\nExample Source Statistics:")
    for source, count in source_stats.items():
        print(f"Source {source}: {count} pairs")
    write_to_jsonl(filtered_stories, output_file)