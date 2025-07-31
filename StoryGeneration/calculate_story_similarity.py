#!/usr/bin/env python3
"""
Script to calculate pairwise similarity for stories in the JSONL file.
Male and female stories are treated separately.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import argparse
import os

class StorySimilarityCalculator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the similarity calculator with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        
    def load_stories(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Load stories from JSONL file and separate male and female stories.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            Tuple of (male_stories, female_stories)
        """
        male_stories = []
        female_stories = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    if 'male' in data and 'story' in data['male']:
                        male_stories.append({
                            'story': data['male']['story'],
                            'character_name': data['male'].get('character_name', ''),
                            'stance': data['male'].get('stance', ''),
                            'explanation': data['male'].get('explanation', ''),
                            'line_number': line_num
                        })
                    
                    if 'female' in data and 'story' in data['female']:
                        female_stories.append({
                            'story': data['female']['story'],
                            'character_name': data['female'].get('character_name', ''),
                            'stance': data['female'].get('stance', ''),
                            'explanation': data['female'].get('explanation', ''),
                            'line_number': line_num
                        })
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(male_stories)} male stories and {len(female_stories)} female stories")
        return male_stories, female_stories
    
    def calculate_similarity_matrix(self, stories: List[Dict], story_type: str) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate pairwise similarity matrix for a list of stories.
        
        Args:
            stories: List of story dictionaries
            story_type: Type of stories ('male' or 'female')
            
        Returns:
            Tuple of (similarity_matrix, story_texts)
        """
        if not stories:
            print(f"No {story_type} stories found")
            return np.array([]), []
        
        # Extract story texts
        story_texts = [story['story'] for story in stories]
        
        print(f"Calculating embeddings for {len(story_texts)} {story_type} stories...")
        
        # Calculate embeddings
        embeddings = self.model.encode(story_texts, show_progress_bar=True)
        
        # Calculate pairwise cosine similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix, story_texts
    
    def analyze_similarity(self, similarity_matrix: np.ndarray, stories: List[Dict], 
                          story_type: str) -> Dict:
        """
        Analyze similarity statistics for a set of stories.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            stories: List of story dictionaries
            story_type: Type of stories ('male' or 'female')
            
        Returns:
            Dictionary with similarity statistics
        """
        if similarity_matrix.size == 0:
            return {}
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        stats = {
            'story_type': story_type,
            'num_stories': len(stories),
            'mean_similarity': np.mean(upper_triangle),
            'std_similarity': np.std(upper_triangle),
            'min_similarity': np.min(upper_triangle),
            'max_similarity': np.max(upper_triangle),
            'median_similarity': np.median(upper_triangle),
            'similarity_matrix': similarity_matrix,
            'stories': stories
        }
        
        print(f"\n{story_type.capitalize()} Stories Similarity Statistics:")
        print(f"Number of stories: {stats['num_stories']}")
        print(f"Mean similarity: {stats['mean_similarity']:.4f}")
        print(f"Std similarity: {stats['std_similarity']:.4f}")
        print(f"Min similarity: {stats['min_similarity']:.4f}")
        print(f"Max similarity: {stats['max_similarity']:.4f}")
        print(f"Median similarity: {stats['median_similarity']:.4f}")
        
        return stats
    
    def find_most_similar_pairs(self, similarity_matrix: np.ndarray, stories: List[Dict], 
                               story_type: str, top_k: int = 10) -> List[Tuple]:
        """
        Find the most similar pairs of stories.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            stories: List of story dictionaries
            story_type: Type of stories ('male' or 'female')
            top_k: Number of top similar pairs to return
            
        Returns:
            List of tuples with (story1_idx, story2_idx, similarity_score)
        """
        if similarity_matrix.size == 0:
            return []
        
        # Get upper triangle indices and values
        upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
        upper_triangle_values = similarity_matrix[upper_triangle_indices]
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(upper_triangle_values)[::-1]
        
        most_similar_pairs = []
        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i]
            story1_idx = upper_triangle_indices[0][idx]
            story2_idx = upper_triangle_indices[1][idx]
            similarity = upper_triangle_values[idx]
            
            most_similar_pairs.append((story1_idx, story2_idx, similarity))
        
        print(f"\nTop {len(most_similar_pairs)} most similar {story_type} story pairs:")
        for i, (idx1, idx2, sim) in enumerate(most_similar_pairs, 1):
            char1 = stories[idx1]['character_name']
            char2 = stories[idx2]['character_name']
            stance1 = stories[idx1]['stance']
            stance2 = stories[idx2]['stance']
            print(f"{i}. {char1} ({stance1}) vs {char2} ({stance2}): {sim:.4f}")
        
        return most_similar_pairs
    
    def plot_similarity_heatmap(self, similarity_matrix: np.ndarray, stories: List[Dict], 
                               story_type: str, output_dir: str = '.'):
        """
        Plot similarity heatmap for stories.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            stories: List of story dictionaries
            story_type: Type of stories ('male' or 'female')
            output_dir: Directory to save the plot
        """
        if similarity_matrix.size == 0:
            return
        
        plt.figure(figsize=(12, 10))
        
        # Create character names for labels
        labels = [f"{story['character_name']}\n({story['stance']})" for story in stories]
        
        # Create heatmap
        sns.heatmap(similarity_matrix, 
                   xticklabels=labels, 
                   yticklabels=labels,
                   cmap='viridis', 
                   annot=False,
                   square=True)
        
        plt.title(f'{story_type.capitalize()} Stories Pairwise Similarity Matrix')
        plt.xlabel('Story Index')
        plt.ylabel('Story Index')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f'{story_type}_stories_similarity_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved similarity heatmap to {output_path}")
        plt.close()
    
    def save_results(self, male_stats: Dict, female_stats: Dict, output_dir: str = '.'):
        """
        Save similarity results to files.
        
        Args:
            male_stats: Statistics for male stories
            female_stats: Statistics for female stories
            output_dir: Directory to save results
        """
        # Save similarity matrices
        if male_stats and 'similarity_matrix' in male_stats:
            np.save(os.path.join(output_dir, 'male_similarity_matrix.npy'), 
                   male_stats['similarity_matrix'])
        
        if female_stats and 'similarity_matrix' in female_stats:
            np.save(os.path.join(output_dir, 'female_similarity_matrix.npy'), 
                   female_stats['similarity_matrix'])
        
        # Save statistics summary
        summary = {
            'male': {k: v for k, v in male_stats.items() if k != 'similarity_matrix' and k != 'stories'},
            'female': {k: v for k, v in female_stats.items() if k != 'similarity_matrix' and k != 'stories'}
        }
        
        with open(os.path.join(output_dir, 'similarity_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved results to {output_dir}")
    
    def run_analysis(self, file_path: str, output_dir: str = '.', top_k: int = 10):
        """
        Run complete similarity analysis on the stories.
        
        Args:
            file_path: Path to the JSONL file
            output_dir: Directory to save results
            top_k: Number of top similar pairs to find
        """
        print("Loading stories...")
        male_stories, female_stories = self.load_stories(file_path)
        
        # Calculate similarity for male stories
        print("\n" + "="*50)
        print("ANALYZING MALE STORIES")
        print("="*50)
        male_similarity, male_texts = self.calculate_similarity_matrix(male_stories, 'male')
        male_stats = self.analyze_similarity(male_similarity, male_stories, 'male')
        #male_pairs = self.find_most_similar_pairs(male_similarity, male_stories, 'male', top_k)
        #self.plot_similarity_heatmap(male_similarity, male_stories, 'male', output_dir)
        
        # Calculate similarity for female stories
        print("\n" + "="*50)
        print("ANALYZING FEMALE STORIES")
        print("="*50)
        female_similarity, female_texts = self.calculate_similarity_matrix(female_stories, 'female')
        female_stats = self.analyze_similarity(female_similarity, female_stories, 'female')
        #female_pairs = self.find_most_similar_pairs(female_similarity, female_stories, 'female', top_k)
        #self.plot_similarity_heatmap(female_similarity, female_stories, 'female', output_dir)
        
        # Save results
        self.save_results(male_stats, female_stats, output_dir)
        
        # Compare male vs female statistics
        if male_stats and female_stats:
            print("\n" + "="*50)
            print("COMPARISON: MALE vs FEMALE")
            print("="*50)
            print(f"Mean similarity - Male: {male_stats['mean_similarity']:.4f}, Female: {female_stats['mean_similarity']:.4f}")
            print(f"Std similarity - Male: {male_stats['std_similarity']:.4f}, Female: {female_stats['std_similarity']:.4f}")
            print(f"Max similarity - Male: {male_stats['max_similarity']:.4f}, Female: {female_stats['max_similarity']:.4f}")
            
            diff = male_stats['mean_similarity'] - female_stats['mean_similarity']
            print(f"Difference (Male - Female): {diff:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Calculate pairwise similarity for stories')
    parser.add_argument('--input', '-i', default='generated_story.jsonl', 
                       help='Input JSONL file path')
    parser.add_argument('--output', '-o', default='.', 
                       help='Output directory for results')
    parser.add_argument('--model', '-m', default='all-MiniLM-L6-v2',
                       help='Sentence transformer model name')
    parser.add_argument('--top-k', '-k', type=int, default=10,
                       help='Number of top similar pairs to show')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize calculator and run analysis
    calculator = StorySimilarityCalculator(args.model)
    calculator.run_analysis(args.input, args.output, args.top_k)

if __name__ == "__main__":
    main() 