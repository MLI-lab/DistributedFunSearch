#!/usr/bin/env python3
"""
Extract noun phrases/concepts from extracted middles using spaCy.

Usage:
    python extract_concepts.py features_20260209_194749/extracted_middles.txt
    python extract_concepts.py features_20260209_194749/extracted_middles.txt --top 100

Dependencies:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


def extract_noun_phrases(nlp, texts: list[str], batch_size: int = 1000) -> Counter:
    """Extract noun phrases from texts using spaCy."""
    phrase_counter = Counter()

    # Process in batches for efficiency
    for doc in nlp.pipe(texts, batch_size=batch_size, disable=["ner"]):
        for chunk in doc.noun_chunks:
            # Clean up the phrase
            phrase = chunk.text.lower().strip()
            # Skip very short or very long phrases
            if 2 <= len(phrase.split()) <= 5 and len(phrase) > 5:
                phrase_counter[phrase] += 1

    return phrase_counter


def main():
    parser = argparse.ArgumentParser(
        description="Extract noun phrases from extracted middles"
    )
    parser.add_argument("input", help="Path to extracted_middles.txt")
    parser.add_argument("--top", "-t", type=int, default=50,
                        help="Number of top phrases to display (default: 50)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file for results")

    args = parser.parse_args()

    if not SPACY_AVAILABLE:
        print("Error: spacy is required. Install with:")
        print("  pip install spacy")
        print("  python -m spacy download en_core_web_sm")
        return 1

    # Load spaCy model
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Error: spaCy model not found. Download with:")
        print("  python -m spacy download en_core_web_sm")
        return 1

    # Read input file
    input_path = Path(args.input)
    print(f"Reading: {input_path}")
    with open(input_path) as f:
        texts = [line.strip() for line in f if line.strip()]

    n = len(texts)
    print(f"Total texts: {n:,}")
    print()

    # Extract noun phrases
    print("Extracting noun phrases (this may take a minute)...")
    phrase_counts = extract_noun_phrases(nlp, texts)
    print(f"Found {len(phrase_counts):,} unique phrases")
    print()

    # Display top phrases
    print("=" * 70)
    print(f"Top {args.top} noun phrases")
    print("=" * 70)
    for i, (phrase, count) in enumerate(phrase_counts.most_common(args.top), 1):
        pct = count / n * 100
        bar = '#' * int(pct / 2)
        print(f"{i:3}. {phrase:45} {count:6,} ({pct:5.1f}%) {bar}")

    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write(f"# Noun phrases from {n:,} descriptions\n\n")
            for phrase, count in phrase_counts.most_common(200):
                pct = count / n * 100
                f.write(f"{phrase:50} {count:6,} ({pct:5.1f}%)\n")
        print()
        print(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
