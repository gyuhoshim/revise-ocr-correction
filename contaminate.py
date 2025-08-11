"""
Text Contamination Tool for OCR Error Simulation

This module provides functions to simulate various types of OCR (Optical Character Recognition) 
errors in text documents, including character-level, word-level, and layout-level contamination.

Main contamination types:
- Deletion: Remove characters, words, or sentences
- Insertion: Add random characters
- Segmentation: Add/remove spaces (over/under segmentation)
- Transposition: Swap adjacent elements
- Substitution: Replace characters with similar-looking ones
- Column layout: Simulate multi-column document reading errors

Author: Gyuho Shim
"""

import random
import string
import re
import json
import time
import textwrap
import jsonlines
from typing import Dict, List, Tuple

# Configuration constants
CONTAMINATION_RATES = {
    'delete': {
        'char': 0.07,
        'word': 0.02,
    },
    'insert': {
        'char': 0.05
    },
    'segment': {
        'over': 0.05,
        'under': 0.05,
    },
    'transpose': {
        'char': 0.05,
        'word': 0.02,
    },
    'substitute': {
        'char': 0.05
    }
}

DEFAULT_MAX_LINE_LENGTH = 30
MAPPING_FILE = 'mapping.json'

# Set random seed using nanosecond timestamp for reproducible randomness
# Remove this line if you want different results each time
random.seed(time.time_ns())


def random_variable(category: str) -> Tuple[str, float]:
    """
    Get random contamination parameters for a given category.
    
    Args:
        category: One of 'delete', 'insert', 'segment', 'transpose', 'substitute'
        
    Returns:
        Tuple of (level, base_rate) where level is the contamination level
        and base_rate is the probability of applying contamination
        
    Raises:
        KeyError: If category is not supported
    """
    if category not in CONTAMINATION_RATES:
        raise KeyError(f"Unsupported category: {category}")
    
    level = random.choice(list(CONTAMINATION_RATES[category].keys()))
    base_rate = CONTAMINATION_RATES[category][level]
    return level, base_rate


def get_random_char() -> str:
    """
    Generate a random character (letters, digits, or punctuation).
    
    Returns:
        A randomly selected character
    """
    choices = string.ascii_letters + string.digits + string.punctuation
    return random.choice(choices)


def get_mapping_dict() -> Dict[str, List[str]]:
    """
    Load character substitution mapping from JSON file.
    
    Expected mapping.json structure:
    {
      "a": ["@","ą","à"],
      "b": ["8","ß"]
    }
    
    Returns:
        Dictionary mapping characters to their substitution candidates
        
    Raises:
        FileNotFoundError: If mapping.json is not found
        json.JSONDecodeError: If mapping.json is invalid
    """
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        char_map = json.load(f)
    return char_map


def deletion(text: str) -> str:
    """
    Apply deletion contamination to text.
    
    Randomly removes characters, words, or sentences based on contamination rates.
    
    Args:
        text: Input text to contaminate
        
    Returns:
        Text with random deletions applied
    """
    level, base_rate = random_variable('delete')
    print(f"[deletion] level={level}, base_rate={base_rate}")

    if level == 'char':
        output = []
        for ch in text:
            if ch.isspace():
                output.append(ch)
            else:
                if random.random() < base_rate:
                    # Delete character
                    continue
                output.append(ch)
        return ''.join(output)

    elif level == 'word':
        tokens = text.split()
        output_tokens = []
        for token in tokens:
            if random.random() < base_rate:
                # Delete word
                continue
            output_tokens.append(token)
        return ' '.join(output_tokens)

    elif level == 'sentence':
        sentences = re.split(r'(?<=[.!?])\s+', text)
        output_sentences = []
        for sent in sentences:
            if random.random() < base_rate:
                # Delete sentence
                continue
            output_sentences.append(sent)
        return ' '.join(output_sentences)


def insertion(text: str) -> str:
    """
    Apply insertion contamination to text.
    
    Randomly inserts characters at character level.
    
    Args:
        text: Input text to contaminate
        
    Returns:
        Text with random insertions applied
    """
    level, base_rate = random_variable('insert')
    print(f"[insertion] level={level}, base_rate={base_rate}")

    output = []
    for ch in text:
        output.append(ch)
        if not ch.isspace() and random.random() < base_rate:
            output.append(get_random_char())
    return ''.join(output)


def segmentation(text: str) -> str:
    """
    Apply segmentation contamination to text.
    
    Adds spaces (over-segmentation) or removes spaces (under-segmentation).
    
    Args:
        text: Input text to contaminate
        
    Returns:
        Text with segmentation errors applied
    """
    level, base_rate = random_variable('segment')
    print(f"[segmentation] level={level}, base_rate={base_rate}")

    if level == 'over':
        # Over-segmentation: add more spaces to split words
        output = []
        for i, ch in enumerate(text):
            output.append(ch)
            if ch != ' ' and random.random() < base_rate:
                if i + 1 < len(text) and text[i + 1] != ' ':
                    output.append(' ')
        return ''.join(output)

    elif level == 'under':
        # Under-segmentation: remove spaces to merge words
        output = []
        for ch in text:
            if ch.isspace() and random.random() < base_rate:
                # Skip space
                continue
            output.append(ch)
        return ''.join(output)


def transposition(text: str) -> str:
    """
    Apply transposition contamination to text.
    
    Swaps adjacent characters, words, or sentences.
    
    Args:
        text: Input text to contaminate
        
    Returns:
        Text with transposition errors applied
    """
    level, base_rate = random_variable('transpose')
    print(f"[transposition] level={level}, base_rate={base_rate}")

    if level == 'char':
        chars = list(text)
        i = 0
        while i < len(chars) - 1:
            if random.random() < base_rate:
                if chars[i] != " " and chars[i+1] != " ":
                    chars[i], chars[i+1] = chars[i+1], chars[i]
                    i += 2
                    continue
            i += 1
        return ''.join(chars)

    elif level == 'word':
        words = text.split()
        i = 0
        while i < len(words) - 1:
            if random.random() < base_rate:
                words[i], words[i+1] = words[i+1], words[i]
                i += 2
                continue
            i += 1
        return ' '.join(words)

    elif level == 'sentence':
        sentences = re.split(r'(?<=[.!?])\s+', text)
        i = 0
        while i < len(sentences) - 1:
            if random.random() < base_rate:
                sentences[i], sentences[i+1] = sentences[i+1], sentences[i]
                i += 2
                continue
            i += 1
        return ' '.join(sentences)


def substitution(text: str) -> str:
    """
    Apply substitution contamination to text.
    
    Replaces characters with visually similar alternatives from mapping file.
    
    Args:
        text: Input text to contaminate
        
    Returns:
        Text with character substitutions applied
    """
    char_map = get_mapping_dict()
    level, base_rate = random_variable('substitute')
    print(f"[substitution] level={level}, base_rate={base_rate}")

    output = []
    for ch in text:
        if random.random() < base_rate and ch in char_map:
            candidates = char_map[ch]
            if candidates:
                substituted = random.choice(candidates)
                output.append(substituted)
            else:
                output.append(ch)
        else:
            output.append(ch)
    return ''.join(output)


def ordering():
    """
    Apply layout contamination for multi-column documents.
    
    TODO: Implement multi-column, multi-row layout contamination
    that considers various document formats for realistic text reconstruction.
    
    Implementation approach:
        1. Split document into paragraphs
        2. Randomly shuffle paragraphs
        3. Mix two paragraphs together
    """
    pass


def split_text_by_length(text: str, max_length: int) -> List[str]:
    """
    Split continuous text into chunks of specified maximum length.
    
    Args:
        text: Input text to split
        max_length: Maximum length of each chunk
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If max_length is less than 1
    """
    if max_length < 1:
        raise ValueError("max_length must be an integer >= 1")
    
    return [
        text[i : i + max_length]
        for i in range(0, len(text), max_length)
    ]


def build_perfect_doc(raw_text: str, max_line_length: int = DEFAULT_MAX_LINE_LENGTH) -> str:
    """
    Format raw text into a document with fixed line lengths.
    
    Args:
        raw_text: Input text to format
        max_line_length: Maximum characters per line
        
    Returns:
        Formatted document with newlines
    """
    lines = split_text_by_length(raw_text, max_line_length)
    return "\n".join(lines)


def contaminate_section(lines: List[str], num_cols: int) -> str:
    """
    Contaminate text section by simulating multi-column OCR reading order.
    
    Divides lines into columns and reads vertically then horizontally,
    simulating how OCR might misread multi-column documents.
    
    Example with num_cols=2:
        Original lines: [A1, A2, A3, B1, B2, B3]
        Result:         [A1 B1, A2 B2, A3 B3]
    
    Args:
        lines: List of text lines to contaminate
        num_cols: Number of columns to simulate
        
    Returns:
        Contaminated text as string with newlines
    """
    if num_cols <= 1:
        return "\n".join(lines)
    
    total_lines = len(lines)
    base_count, remainder = divmod(total_lines, num_cols)
    
    # Divide lines into columns
    columns = []
    start_idx = 0
    for col_idx in range(num_cols):
        column_size = base_count + (1 if col_idx < remainder else 0)
        columns.append(lines[start_idx : start_idx + column_size])
        start_idx += column_size
    
    # Read vertically (by row) across columns
    max_rows = max(len(col) for col in columns)
    contaminated_lines = []
    for row_idx in range(max_rows):
        row_texts = [
            col[row_idx].strip() for col in columns if row_idx < len(col)
        ]
        contaminated_lines.append(" ".join(row_texts))
    
    return "\n".join(contaminated_lines)


def contaminate_document_by_sections(perfect_doc: str, sections: List[Tuple[int, int, int]], 
                                   max_line_length: int = DEFAULT_MAX_LINE_LENGTH) -> str:
    """
    Apply column-level contamination to document sections.
    
    Args:
        perfect_doc: Document with lines already split by max_line_length
        sections: List of (start_idx, end_idx, num_cols) tuples
        max_line_length: Maximum characters per line
        
    Returns:
        Document with column contamination applied
        
    Raises:
        ValueError: If section format is invalid
    """
    all_lines = [line for line in perfect_doc.splitlines() if line.strip()]
    
    results = []
    for section in sections:
        if len(section) != 3:
            raise ValueError("sections must be tuples of (start_idx, end_idx, num_cols)")
        
        start_idx, end_idx, num_cols = section
        section_lines = all_lines[max(0, start_idx) : min(end_idx, len(all_lines))]
        
        # Set column width for multi-column processing
        if num_cols <= 1:
            column_width = max_line_length
        else:
            column_width = max_line_length // num_cols
        
        # Combine section lines and rewrap for column processing
        combined_text = "".join(section_lines)
        wrapped_lines = textwrap.wrap(combined_text, width=column_width, break_long_words=True)
        
        contaminated = contaminate_section(wrapped_lines, num_cols)
        results.append(contaminated)
    
    return "\n".join(results)


def set_contamination_parameters(error_type: str) -> Dict[str, float]:
    """
    Set contamination parameters based on error type.
    
    Args:
        error_type: Either "Type I" (column-focused) or "Type II" (char/word-focused)
        
    Returns:
        Dictionary of contamination levels for each error type
        
    Raises:
        ValueError: If error_type is not supported
    """
    if error_type == "Type I":
        # Column-level contamination is significant
        return {
            'delete': 0.03,
            'insert': 0.03,
            'segment': 0.02,
            'transpose': 0.03,
            'substitute': 0.03
        }
    elif error_type == "Type II":
        # Character/word/sentence-level contamination is significant
        return {
            'delete': 0.12,
            'insert': 0.12,
            'segment': 0.08,
            'transpose': 0.12,
            'substitute': 0.08
        }
    else:
        raise ValueError("Unknown error type. Use 'Type I' or 'Type II'.")


def generate_section_params(total_lines: int, error_type: str) -> List[Tuple[int, int, int]]:
    """
    Dynamically generate section parameters for column-level contamination.
    
    Args:
        total_lines: Total number of lines in the document
        error_type: Type of error to simulate ('Type I' or 'Type II')
        
    Returns:
        List of tuples representing (start_idx, end_idx, num_cols)
        
    Raises:
        ValueError: If error_type is not supported
    """
    if error_type == "Type I":
        # More column-level contamination
        sections = []
        split_points = [0]
        while len(split_points) < random.randint(1, 2):
            print("split_points: ", split_points)
            split_point = random.randint(1, total_lines - 1)
            if split_point not in split_points:
                split_points.append(split_point)
        
        split_points.sort()
        
        # Generate section parameters with varying column counts
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            num_cols = random.choice([2, 3, 4])  # Multi-column focus
            sections.append((start_idx, end_idx, num_cols))
        
        # Add remaining part if any
        if split_points[-1] < total_lines:
            sections.append((split_points[-1], total_lines, random.choice([2, 3, 4])))

    elif error_type == "Type II":
        # Less column-level contamination
        sections = [(0, total_lines, random.choice([2, 3]))]
    
    elif error_type == "Type III":
        # Single column only
        sections = [(0, total_lines, 1)]
    else:
        raise ValueError("Unknown error type. Use 'Type I', 'Type II', or 'Type III'.")
    
    return sections


def apply_contamination(text: str, error_type: str) -> str:
    """
    Apply comprehensive contamination to text based on error type.
    
    Args:
        text: Input text to contaminate
        error_type: Type of contamination to apply
        
    Returns:
        Contaminated text
    """
    # Prepare document
    perfect_doc = build_perfect_doc(text)
    total_lines = len(perfect_doc.splitlines())
    
    # Dynamically generate section parameters
    section_params = generate_section_params(total_lines, error_type)
    
    # Apply column-level contamination
    column_contaminated = contaminate_document_by_sections(perfect_doc, section_params)
    
    return column_contaminated


def process_jsonl_file(input_file: str, output_file: str, total_data_lines: int = 30000):
    """
    Process JSONL file and apply various contamination types.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        total_data_lines: Maximum number of lines to process
    """
    breakpoints = [0, 3000, 6000, 9000, 12000, 15000, 27000]
    num_line = 0
    
    with jsonlines.open(input_file) as reader:
        with jsonlines.open(output_file, 'w') as writer:
            for line in reader:
                num_line += 1
                if num_line > total_data_lines:
                    break
                elif num_line >= breakpoints[0] and num_line < breakpoints[1]: 
                    line['contaminated_text'] = segmentation(line['text'])
                elif num_line >= breakpoints[1] and num_line < breakpoints[2]:
                    line['contaminated_text'] = deletion(line['text'])
                elif num_line >= breakpoints[2] and num_line < breakpoints[3]:
                    line['contaminated_text'] = insertion(line['text'])
                elif num_line >= breakpoints[3] and num_line < breakpoints[4]:
                    line['contaminated_text'] = transposition(line['text'])
                elif num_line >= breakpoints[4] and num_line < breakpoints[5]:
                    line['contaminated_text'] = substitution(line['text'])
                elif num_line >= breakpoints[5] and num_line < breakpoints[6]:
                    contaminated_text = apply_contamination(line['text'], "Type II")
                    line['contaminated_text'] = contaminated_text
                else:
                    line['contaminated_text'] = apply_contamination(line['text'], "Type III")
                
                writer.write(line)


def main():
    """Example usage of the contamination functions."""
    sample_text = "This is a sample text for demonstration purposes. It contains multiple sentences."
    
    print("Original text:", sample_text)
    print("\nContamination examples:")
    print("Deletion:", deletion(sample_text))
    print("Insertion:", insertion(sample_text))
    print("Segmentation:", segmentation(sample_text))
    print("Transposition:", transposition(sample_text))
    
    # Note: substitution requires mapping.json file
    try:
        print("Substitution:", substitution(sample_text))
    except FileNotFoundError:
        print("Substitution: (requires mapping.json file)")


if __name__ == "__main__":
    # Example usage
    input_file = './new_data/cut_wikidata_en.jsonl'
    output_file = './data/meta_final.jsonl'
    
    # Uncomment to process files
    # process_jsonl_file(input_file, output_file)
    
    # Run example demonstrations
    main()
