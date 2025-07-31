import json
import os
from collections import defaultdict
from tqdm import tqdm
from utils import read_from_jsonl

def analyze_stories(file_path):
    # Read the stories
    stories = read_from_jsonl(file_path)
    
    # Initialize counters
    total_stories = len(stories)
    #stance_counts = defaultdict(int)
    source_counts = defaultdict(int)
    
    # Analyze each story
    for story in tqdm(stories, desc="Analyzing stories"):
        # Count stances
        #male_stance = story["male"]["stance"]
        #female_stance = story["female"]["stance"]
        #stance_counts[f"male_{male_stance}"] += 1
        #stance_counts[f"female_{female_stance}"] += 1
        
        # Count sources if available
        if "example_source" in story:
            source_counts[story["example_source"]] += 1
    
    # Print statistics
    print("\nStory Analysis Statistics")
    print("=" * 30)
    print(f"Total number of stories: {total_stories}")
    
    #print("\nStance Distribution:")
    #print("-" * 20)
    #for stance, count in stance_counts.items():
    #    print(f"{stance}: {count} ({count/total_stories*100:.1f}%)")
    
    if source_counts:
        print("\nSource Distribution:")
        print("-" * 20)
        for source, count in source_counts.items():
            print(f"Source {source}: {count} ({count/total_stories*100:.1f}%)")

if __name__ == "__main__":
    file_path = "StoryGeneration/generated_data.jsonl"
    analyze_stories(file_path) 