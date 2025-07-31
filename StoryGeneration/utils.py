import json
import re

def write_to_jsonl(data, filename):
    """Save a list of dictionaries to a JSONL file."""
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def append_to_jsonl(item, filename):
    """Save a list of dictionaries to a JSONL file."""
    with open(filename, 'a') as f:
        f.write(json.dumps(item) + '\n')

def read_from_jsonl(filename):
    """Read a JSONL file and return a list of dictionaries."""
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]

def get_stance(response):
    if "immoral" in response.lower():
        return "Immoral"
    elif "moral" in response.lower():
        return "Moral"
    elif "both" in response.lower():
        return "Both"
    elif "A" in response:
        return "Moral"
    elif "B" in response:
        return "Immoral"
    elif "C" in response:
        return "Both"
    else:
        return "INVALID"
    
def extract_info(text):
    pattern = re.compile(
        r"(?:\*{1,2})?Male Story:(?:\*{1,2})?\s*(.*?)\n\n(?:\_{3})?"  # Capture male story (with one or two '*')
        r"Stance:\s*(.*?)\n\n"      # Capture male stance
        r"Explanation:\s*(.*?)\n\n"   # Capture male explanation
        r"(?:\*{1,2})?Female Story:(?:\*{1,2})?\s*(.*?)\n\n(?:\_{3})?"  # Capture female story (with one or two '*')
        r"Stance:\s*(.*?)\n\n"      # Capture female stance
        r"Explanation:\s*(.*?)$",    # Capture female explanation
        re.S
    )

    match = pattern.search(text)
    if match:
        male_story, male_stance, male_explanation, female_story, female_stance, female_explanation = match.groups()

        return {
            "male": {
                "story": male_story.strip(),
                "stance": male_stance.strip(),
                "explanation": male_explanation.strip()
            },
            "female": {
                "story": female_story.strip(),
                "stance": female_stance.strip(),
                "explanation": female_explanation.strip()
            }
        }

    return None

def extract_info_with_character(text):
    # More flexible pattern that handles various header formats and spacing
    pattern = re.compile(
        r"(?:\*{1,2})?Male Story:(?:\*{1,2})?\s*(.*?)(?:\n\s*)?Male Main Character:\s*(.*?)(?:\n\s*)?Stance:\s*(.*?)(?:\n\s*)?Explanation:\s*(.*?)(?:\n\s*)?(?:\*{1,2})?Female Story:(?:\*{1,2})?\s*(.*?)(?:\n\s*)?Female Main Character:\s*(.*?)(?:\n\s*)?Stance:\s*(.*?)(?:\n\s*)?Explanation:\s*(.*?)$",
        re.S | re.M
    )

    match = pattern.search(text)
    if match:
        male_story, male_name, male_stance, male_explanation, female_story, female_name, female_stance, female_explanation = match.groups()

        return {
            "male": {
                "story": male_story.strip(),
                "character_name": male_name.strip(),
                "stance": male_stance.strip(),
                "explanation": male_explanation.strip()
            },
            "female": {
                "story": female_story.strip(),
                "character_name": female_name.strip(),
                "stance": female_stance.strip(),
                "explanation": female_explanation.strip()
            }
        }

    return None

def extract_stance_and_explanation(text):
    # Remove the 'assistant' prefix if present
    text = text.lstrip("assistant").strip()

    # Use regular expressions to capture the stance and explanation
    stance_pattern = r"STANCE: (.+)"
    explanation_pattern = r"EXPLANATION: (.+)"

    # Extract stance
    stance_match = re.search(stance_pattern, text)
    stance = stance_match.group(1).strip() if stance_match else None

    # Extract explanation (capture everything after "EXPLANATION:")
    explanation_match = re.search(explanation_pattern, text, re.S)
    explanation = explanation_match.group(1).strip() if explanation_match else None

    return stance, explanation

def extract_neutral_morality_explanation(text):
    # Define the regular expression pattern with case-insensitivity
    pattern = r"(?i)Neutral Morality Stance Explanation:\s*(.*)"
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    if match:
        # Return the extracted portion after the explanation label
        return match.group(1).strip()
    else:
        return text.split("xplanation:")[-1].strip()
    
def extract_updated_explanations(response):
    """
    Extracts the updated male and female explanations from the response.
    
    Args:
        response (str): The LLM's response containing the updated explanations.
    
    Returns:
        tuple: A tuple containing the updated male explanation and updated female explanation.
    """
    male_pattern = r"Updated Male Explanation:\s*(.*?)\s*Updated Female Explanation:"
    female_pattern = r"Updated Female Explanation:\s*(.*)"

    male_match = re.search(male_pattern, response, re.DOTALL)
    female_match = re.search(female_pattern, response, re.DOTALL)

    updated_male_explanation = male_match.group(1).strip() if male_match else None
    updated_female_explanation = female_match.group(1).strip() if female_match else None

    return updated_male_explanation, updated_female_explanation