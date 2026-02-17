#!/usr/bin/env python3
"""
Extract features from algorithm descriptions by stripping template prefix and suffix.

This script:
1. Reads template from template_results.json (or uses defaults)
2. Strips the [combine] verb prefix (Leverage, Incorporate, Fuse, etc.)
3. Strips the "to [goal]..." suffix (to optimize, maximize, enhance, etc.)
4. Keeps the middle part containing features and mechanisms
5. Counts word frequencies and common phrases

Arguments:
    path                  Checkpoint .pkl file or folder containing checkpoints.
    --template            Path to template_results.json from templates.py.
    --output, -o          Output directory for results, default auto generated.
    --top, -t             Number of top items to display, default 50.

Usage:
    python extract_features.py checkpoint.pkl --template template_results.json
    python extract_features.py ./checkpoints/run_folder/
    python extract_features.py checkpoint.pkl --top 100

Dependencies:
    pip install numpy
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import find_latest_checkpoint, load_checkpoint, extract_descriptions


# Default verbs (used if no template file provided)
DEFAULT_COMBINE_VERBS = [
    'leverage', 'incorporate', 'fuse', 'combine', 'integrate',
    'leverages', 'incorporates', 'fuses', 'combines', 'integrates',
    'use', 'uses', 'apply', 'applies', 'employ', 'employs',
]

DEFAULT_GOAL_VERBS = [
    'optimize', 'maximize', 'enhance', 'prioritize', 'improve',
    'guide', 'balance', 'refine', 'create', 'identify',
    'select', 'construct', 'build', 'find', 'achieve',
]


def parse_template_response(response: str) -> dict:
    """
    Parse the LLM template response to extract verb lists.

    Looks for patterns like:
        [combine]: verbs like leverage, incorporate, fuse
        [goal]: optimize/improve/maximize/enhance
    """
    combine_verbs = []
    goal_verbs = []

    # Look for [combine] slot and extract words
    # Matches words after "like" or between asterisks or in comma lists
    combine_match = re.search(r'\[combine\][^:]*:([^\n]+)', response, re.IGNORECASE)
    if combine_match:
        text = combine_match.group(1)
        # Extract words between asterisks: *word*
        asterisk_words = re.findall(r'\*(\w+)\*', text)
        combine_verbs.extend(asterisk_words)
        # Extract words after "like" in comma separated format
        like_match = re.search(r'like\s+([^(]+)', text, re.IGNORECASE)
        if like_match:
            words = re.findall(r'(\w+)', like_match.group(1))
            combine_verbs.extend(words)

    # Look for [goal] slot and extract words
    goal_match = re.search(r'\[goal\][^:]*:([^\n]+)', response, re.IGNORECASE)
    if goal_match:
        text = goal_match.group(1)
        # Extract words separated by / like "optimize/improve/maximize"
        slash_words = re.findall(r'(\w+)(?:/|$)', text)
        goal_verbs.extend(slash_words)
        # Also look for asterisk words
        asterisk_words = re.findall(r'\*(\w+)\*', text)
        goal_verbs.extend(asterisk_words)

    # Clean up: lowercase, remove duplicates, filter short words
    combine_verbs = list(set(w.lower() for w in combine_verbs if len(w) > 2))
    goal_verbs = list(set(w.lower() for w in goal_verbs if len(w) > 2))

    # Add common conjugations
    combine_expanded = []
    for v in combine_verbs:
        combine_expanded.append(v)
        if not v.endswith('s'):
            combine_expanded.append(v + 's')
        if v.endswith('e'):
            combine_expanded.append(v[:-1] + 'ing')

    return {
        'combine_verbs': combine_expanded if combine_expanded else None,
        'goal_verbs': goal_verbs if goal_verbs else None,
    }


def load_template(template_path: Path) -> dict:
    """Load and parse template_results.json."""
    with open(template_path) as f:
        data = json.load(f)

    response = data.get('response', '')
    parsed = parse_template_response(response)

    return {
        'combine_verbs': parsed['combine_verbs'] or DEFAULT_COMBINE_VERBS,
        'goal_verbs': parsed['goal_verbs'] or DEFAULT_GOAL_VERBS,
        'raw_response': response,
    }


# Global verb lists (set by main or defaults)
COMBINE_VERBS = DEFAULT_COMBINE_VERBS.copy()
GOAL_VERBS = DEFAULT_GOAL_VERBS.copy()

STOP_WORDS = ENGLISH_STOP_WORDS


def extract_middle(description: str) -> str:
    """
    Extract the middle part of a description by stripping prefix and suffix.

    Returns the content between [combine] verb and "to [goal]..." suffix.
    """
    text = description.strip()

    # Step 1: Strip [combine] verb prefix
    pattern_prefix = r'^(' + '|'.join(COMBINE_VERBS) + r')\s+'
    text = re.sub(pattern_prefix, '', text, flags=re.IGNORECASE)

    # Step 2: Strip "to [goal]..." suffix
    # Match "to [goal_verb]" and everything after
    pattern_suffix = r'\s+to\s+(' + '|'.join(GOAL_VERBS) + r')\b.*$'
    text = re.sub(pattern_suffix, '', text, flags=re.IGNORECASE)

    # Clean up trailing punctuation
    text = text.rstrip('.,;:')

    return text.strip()


def tokenize(text: str) -> list:
    """Tokenize text into lowercase words."""
    words = re.findall(r'[a-z]+', text.lower())
    return [w for w in words if len(w) > 2 and w not in STOP_WORDS]


def extract_bigrams(text: str) -> list:
    """Extract bigrams (two-word phrases) from text."""
    words = re.findall(r'[a-z]+', text.lower())
    words = [w for w in words if len(w) > 2]
    bigrams = []
    for i in range(len(words) - 1):
        if words[i] not in STOP_WORDS or words[i+1] not in STOP_WORDS:
            bigrams.append(f"{words[i]} {words[i+1]}")
    return bigrams


def analyze_descriptions(descriptions: list, top_n: int = 50) -> dict:
    """
    Analyze all descriptions and extract features.

    Returns dict with extracted middles, word counts, and bigram counts.
    """
    middles = []
    word_counter = Counter()
    bigram_counter = Counter()

    for desc in descriptions:
        middle = extract_middle(desc)
        middles.append(middle)

        # Count words
        words = tokenize(middle)
        word_counter.update(words)

        # Count bigrams
        bigrams = extract_bigrams(middle)
        bigram_counter.update(bigrams)

    return {
        'middles': middles,
        'word_counts': word_counter.most_common(top_n * 2),
        'bigram_counts': bigram_counter.most_common(top_n * 2),
    }


def main():
    global COMBINE_VERBS, GOAL_VERBS

    parser = argparse.ArgumentParser(
        description='Extract features from algorithm descriptions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('path', help='Checkpoint .pkl file or folder')
    parser.add_argument('--template', default=None,
                        help='Path to template_results.json from templates.py')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory for results')
    parser.add_argument('--top', '-t', type=int, default=50,
                        help='Number of top items to display (default: 50)')

    args = parser.parse_args()

    # Load template if provided
    if args.template:
        template_path = Path(args.template)
        if template_path.exists():
            print(f"Loading template: {template_path}")
            template = load_template(template_path)
            COMBINE_VERBS = template['combine_verbs']
            GOAL_VERBS = template['goal_verbs']
            print(f"  Combine verbs: {', '.join(COMBINE_VERBS[:10])}...")
            print(f"  Goal verbs: {', '.join(GOAL_VERBS[:10])}...")
            print()
        else:
            print(f"Warning: template file not found: {template_path}")
            print("Using default verb lists")
            print()

    # Find checkpoint
    path = Path(args.path)
    checkpoint_path = find_latest_checkpoint(path)

    print(f"Loading: {checkpoint_path}")
    checkpoint = load_checkpoint(str(checkpoint_path))

    descriptions = extract_descriptions(checkpoint)
    n = len(descriptions)
    print(f"Total descriptions: {n:,}")
    print()

    # Analyze
    print("Extracting features...")
    results = analyze_descriptions(descriptions, top_n=args.top)
    print("Done")
    print()

    # Show examples of extracted middles
    print("=" * 70)
    print("Example extractions (first 10)")
    print("=" * 70)
    for i, (orig, middle) in enumerate(zip(descriptions[:10], results['middles'][:10])):
        print(f"\nOriginal:  {orig[:100]}...")
        print(f"Extracted: {middle[:100]}...")
    print()

    # Show top words
    print("=" * 70)
    print(f"Top {args.top} words")
    print("=" * 70)
    for i, (word, count) in enumerate(results['word_counts'][:args.top], 1):
        pct = count / n * 100
        bar = '#' * int(pct / 2)
        print(f"{i:3}. {word:25} {count:6,} ({pct:5.1f}%) {bar}")
    print()

    # Show top bigrams
    print("=" * 70)
    print(f"Top {args.top} bigrams (two-word phrases)")
    print("=" * 70)
    for i, (bigram, count) in enumerate(results['bigram_counts'][:args.top], 1):
        pct = count / n * 100
        bar = '#' * int(pct / 2)
        print(f"{i:3}. {bigram:35} {count:6,} ({pct:5.1f}%) {bar}")
    print()

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./features_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results JSON
    results_file = output_dir / "feature_analysis.json"
    with open(results_file, 'w') as f:
        json.dump({
            'checkpoint': str(checkpoint_path),
            'total_descriptions': n,
            'generated': datetime.now().isoformat(),
            'top_words': results['word_counts'][:200],
            'top_bigrams': results['bigram_counts'][:200],
        }, f, indent=2)
    print(f"Saved: {results_file}")

    # Save all extracted middles
    middles_file = output_dir / "extracted_middles.txt"
    with open(middles_file, 'w') as f:
        for middle in results['middles']:
            f.write(middle + '\n')
    print(f"Saved: {middles_file}")

    # Save word frequencies as simple text
    words_file = output_dir / "word_frequencies.txt"
    with open(words_file, 'w') as f:
        f.write(f"# Word frequencies from {n:,} descriptions\n")
        f.write(f"# Checkpoint: {checkpoint_path.name}\n\n")
        for word, count in results['word_counts']:
            pct = count / n * 100
            f.write(f"{word:30} {count:6,} ({pct:5.1f}%)\n")
    print(f"Saved: {words_file}")

    # Save bigram frequencies
    bigrams_file = output_dir / "bigram_frequencies.txt"
    with open(bigrams_file, 'w') as f:
        f.write(f"# Bigram frequencies from {n:,} descriptions\n")
        f.write(f"# Checkpoint: {checkpoint_path.name}\n\n")
        for bigram, count in results['bigram_counts']:
            pct = count / n * 100
            f.write(f"{bigram:40} {count:6,} ({pct:5.1f}%)\n")
    print(f"Saved: {bigrams_file}")

    print()
    print(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
