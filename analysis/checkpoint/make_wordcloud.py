#!/usr/bin/env python3
"""
Generate word clouds from checkpoint descriptions using n-grams (phrases).

N-gram extraction: extracts ALL phrases from 2 to 5 words and combines them.
The most frequent ones appear in the wordcloud, regardless of length.
So "greedy selection" (2 words) and "long term independence" (3 words)
both show up based on frequency.

Common/obvious phrases like "independent set", "greedy selection", "prioritize nodes"
are excluded by default. Use --no-exclude-phrases to show all phrases (useful for
comparing wordclouds from two runs to see if they generate similar things).

Dependencies: pip install wordcloud matplotlib

Options:
    --min-n N           Minimum n-gram length (default: 2)
    --max-n N           Maximum n-gram length (default: 5)
    --single-words      Use single words instead of n-grams
    --no-exclude-phrases  Show all phrases, don't exclude common ones
    --exclusions FILE   Extra words to exclude (one per line)
    --each              One wordcloud per checkpoint in folder
    --combined          Single combined wordcloud from all checkpoints
    --top N             Show top N phrases in console (default: 30)

Usage:
    python make_wordcloud.py checkpoint.pkl           # Specific checkpoint
    python make_wordcloud.py ./checkpoints/           # Auto-picks latest checkpoint in folder
    python make_wordcloud.py ./checkpoints/ --each    # One wordcloud per checkpoint
    python make_wordcloud.py ./checkpoints/ --combined  # Merge all checkpoints
    python make_wordcloud.py checkpoint.pkl --min-n 3 --max-n 4
    python make_wordcloud.py checkpoint.pkl --no-exclude-phrases
    python make_wordcloud.py checkpoint.pkl --single-words
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import find_checkpoints, load_checkpoint, extract_descriptions

# Default exclusion list: common English words and generic terms.
# Add terms here that appear frequently but are not meaningful.
DEFAULT_EXCLUSIONS = {
    # Articles, pronouns, prepositions
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    'not', 'only', 'same', 'than', 'too', 'very', 'just', 'also',
    'this', 'that', 'these', 'those', 'it', 'its', 'which', 'who',
    'i', 'we', 'you', 'he', 'she', 'they', 'me', 'us', 'him', 'her', 'them',
    'my', 'our', 'your', 'his', 'their', 'mine', 'ours', 'yours', 'theirs',
    'what', 'where', 'when', 'why', 'how', 'all', 'each', 'every', 'any',
    'some', 'no', 'none', 'one', 'two', 'three', 'if', 'then', 'else',
    'more', 'most', 'less', 'least', 'much', 'many', 'few', 'little',
    'other', 'another', 'such', 'own', 'about', 'between', 'under', 'over',

    # Generic function/code terms (add more as you discover noise)
    'function', 'priority', 'improves', 'upon', 'first', 'second',
    'using', 'based', 'better', 'approach', 'method', 'value', 'values',
    'number', 'returns', 'return', 'given', 'takes', 'use', 'uses',
    'new', 'higher', 'lower', 'high', 'low', 'like', 'way', 'make',
    'helps', 'help', 'allows', 'allow', 'works', 'work', 'instead',
    'rather', 'well', 'good', 'best', 'even', 'still', 'also',
    'thus', 'therefore', 'however', 'while', 'since', 'because',
    'whether', 'although', 'though', 'unless', 'until', 'once', 'leverage', 'deletion', 'fuse', 'incorporate'
}

# Default phrase exclusions: common/obvious phrases that don't add information.
# Add phrases here that appear frequently but are not informative.
DEFAULT_PHRASE_EXCLUSIONS = {
    # Goal statements (what we're trying to do, not how)
    'independent set',
    'independent sets',
    'independent set size',
    'independent set growth',
    'independent set construction',
    'greedy independent set',
    'greedy independent set construction',
    'maximal independent',
    'maximal independent set',
    'maximize independent',
    'maximize independent set',
    'maximizing independent',
    'maximizing independent set',
    'maximizing independent set size',
    'optimal independent',
    'optimal independent sets',
    'optimize independent',
    'optimize independent set',
    'long term independence',
    'term independence',
    'long term independent',
    'term independent',
    'long term independent set',
    'term independent set',
    'long term independent set growth',
    'term independent set growth',

    # Generic action phrases
    'prioritize nodes',
    'nodes maximize',
    'set construction',
    'priority function',
    'greedy selection',
    'enhance greedy',
    'greedy independent',
    'greedy construction',
    'greedy algorithm',
    'greedy independence',
    'optimize greedy selection', 
    'enhance greedy selection', 
    'optimize greedy selection', 
    'robustness against deletions', 
    'greedy selection efficiency', 
    'robustness against deletions', 
    'enhance greedy selection efficiency',
    'guide greedy selection',
    'term independence minimizing',
    'long term independence minimizing',
    'prioritize nodes maximize long term independence',


    # Partial/filler phrases
    'long term',
    'maximize long',
    'maximize long term',
    'maximize long term independence',
    'nodes maximize long',
    'nodes maximize long term',
    'nodes maximize long term independence',
    'prioritize nodes maximize',
    'prioritize nodes maximize long',
    'prioritize nodes maximize long term',
    'prioritize nodes maximize future',
    'gains long',
    'gains long term',
    'immediate gains long',
    'immediate gains long term',
    'nodes offering',
    'offering maximal',
    'nodes offering maximal',
    'prioritize nodes offering',
    'prioritize nodes offering maximal',
    'features including',
    'set size',
    'set growth',

    # Domain obvious phrases
    'correcting codes',
    'correction graphs',
}


def load_exclusions(exclusions_path: Path) -> Set[str]:
    """Load exclusion words from file (one word per line)."""
    exclusions = set()
    if exclusions_path.exists():
        with open(exclusions_path) as f:
            for line in f:
                word = line.strip().lower()
                if word and not word.startswith('#'):
                    exclusions.add(word)
    return exclusions


def extract_words(texts: List[str], exclusions: Set[str]) -> Counter:
    """Extract word frequencies from texts, excluding stopwords."""
    word_counts = Counter()

    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        for word in words:
            if word not in exclusions:
                word_counts[word] += 1

    return word_counts


def extract_ngrams(texts: List[str], exclusions: Set[str], n: int = 2) -> Counter:
    """Extract n-gram frequencies from texts, filtering out stopwords."""
    ngram_counts = Counter()

    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Filter out exclusions from word list
        filtered = [w for w in words if w not in exclusions]

        # Extract n-grams
        for i in range(len(filtered) - n + 1):
            ngram = ' '.join(filtered[i:i+n])
            ngram_counts[ngram] += 1

    return ngram_counts


def extract_mixed_ngrams(
    texts: List[str],
    exclusions: Set[str],
    phrase_exclusions: Set[str] = None,
    min_n: int = 2,
    max_n: int = 5,
    min_count: int = 3
) -> Counter:
    """Extract n-grams from min_n to max_n words, return combined counts."""
    if phrase_exclusions is None:
        phrase_exclusions = DEFAULT_PHRASE_EXCLUSIONS

    # Extract n-grams of specified lengths
    all_ngrams = []
    for n in range(min_n, max_n + 1):
        all_ngrams.append(extract_ngrams(texts, exclusions, n=n))

    # Combine, filter by minimum count and excluded phrases
    combined = Counter()
    for ngram_counter in all_ngrams:
        for gram, count in ngram_counter.items():
            if count >= min_count and gram not in phrase_exclusions:
                combined[gram] = count

    return combined


def generate_wordcloud_image(
    word_counts: Counter,
    output_path: Path,
    title: str = "",
    width: int = 1200,
    height: int = 600,
):
    """Generate and save word cloud image."""
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Error: wordcloud and matplotlib required")
        print("  Install: pip install wordcloud matplotlib")
        return False

    if not word_counts:
        print("  No words to display")
        return False

    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        max_words=150,
        colormap='viridis',
        relative_scaling=0.5,
    ).generate_from_frequencies(word_counts)

    plt.figure(figsize=(width/100, height/100))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title, fontsize=14, pad=10)
    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return True


def process_checkpoint(
    checkpoint_path: Path,
    exclusions: Set[str],
    output_dir: Path,
    prefix: str = "",
    use_ngrams: bool = True,
    min_n: int = 2,
    max_n: int = 5,
    no_exclude_phrases: bool = False,
):
    """Process a single checkpoint and generate word cloud."""
    print(f"Loading: {checkpoint_path.name}")
    checkpoint = load_checkpoint(str(checkpoint_path))

    descriptions = extract_descriptions(checkpoint)
    print(f"  Extracted {len(descriptions)} descriptions")

    if not descriptions:
        print("  No descriptions found, skipping")
        return None

    if use_ngrams:
        phrase_excl = set() if no_exclude_phrases else None
        word_counts = extract_mixed_ngrams(
            descriptions, exclusions,
            phrase_exclusions=phrase_excl,
            min_n=min_n, max_n=max_n
        )
        print(f"  Unique phrases (after exclusions): {len(word_counts)}")
    else:
        word_counts = extract_words(descriptions, exclusions)
        print(f"  Unique words (after exclusions): {len(word_counts)}")

    # Generate output filenames
    stem = checkpoint_path.stem
    if prefix:
        base_name = f"{prefix}_{stem}"
    else:
        base_name = stem

    # Always save frequencies (works without wordcloud library)
    freq_file = output_dir / f"{base_name}_frequencies.json"
    with open(freq_file, 'w') as f:
        json.dump(dict(word_counts.most_common(200)), f, indent=2)
    print(f"  Saved: {freq_file}")

    # Generate word cloud image (requires wordcloud library)
    image_path = output_dir / f"{base_name}_wordcloud.png"
    if generate_wordcloud_image(word_counts, image_path, title=checkpoint_path.name):
        print(f"  Saved: {image_path}")

    # Return counts for potential combination
    return word_counts


def main():
    parser = argparse.ArgumentParser(
        description='Generate word clouds from checkpoint descriptions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('paths', nargs='+',
                        help='Checkpoint .pkl file(s) or folder(s). If folder given without --each/--combined, uses latest checkpoint.')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: ./wordclouds_<timestamp>/)')
    parser.add_argument('--each', action='store_true',
                        help='Generate separate word cloud for each checkpoint in folder')
    parser.add_argument('--combined', action='store_true',
                        help='Generate single combined word cloud from all checkpoints')
    parser.add_argument('--exclusions', '-e', default=None,
                        help='Extra words to exclude (one per line), added to defaults')
    parser.add_argument('--single-words', action='store_true',
                        help='Use single words instead of n-grams (default: n-grams)')
    parser.add_argument('--no-exclude-phrases', action='store_true',
                        help='Show all phrases, do not exclude common ones')
    parser.add_argument('--min-n', type=int, default=2,
                        help='Minimum n-gram length (default: 2)')
    parser.add_argument('--max-n', type=int, default=5,
                        help='Maximum n-gram length (default: 5)')
    parser.add_argument('--top', '-t', type=int, default=30,
                        help='Show top N phrases in console (default: 30)')

    args = parser.parse_args()

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./wordclouds_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load exclusions (defaults always applied, custom file adds to them)
    exclusions = DEFAULT_EXCLUSIONS.copy()
    if args.exclusions:
        custom = load_exclusions(Path(args.exclusions))
        exclusions.update(custom)
        print(f"Loaded {len(custom)} extra exclusions from {args.exclusions}")

    # Collect checkpoints
    # If --each or --combined: collect all checkpoints from each path
    # Otherwise: pick only the latest checkpoint from each directory
    all_checkpoints = []
    for p in args.paths:
        path = Path(p)
        checkpoints = find_checkpoints(path)
        if not checkpoints:
            print(f"Warning: no checkpoints found in {path}")
            continue

        if args.each or args.combined:
            # Use all checkpoints
            all_checkpoints.extend(checkpoints)
        else:
            # Use only the latest checkpoint from this path
            if path.is_dir() and len(checkpoints) > 1:
                latest = checkpoints[-1]  # Already sorted by mtime, last is newest
                print(f"  {path.name}: {len(checkpoints)} checkpoints, using latest: {latest.name}")
                all_checkpoints.append(latest)
            else:
                all_checkpoints.extend(checkpoints)

    if not all_checkpoints:
        print("No checkpoints found")
        return 1

    print(f"Processing {len(all_checkpoints)} checkpoint(s)")
    print()

    # Determine mode
    # Default is n-grams (phrases), use --single-words to switch
    use_ngrams = not args.single_words
    mode_label = "phrases" if use_ngrams else "words"

    if args.combined:
        # Combined mode: merge all descriptions
        print("=" * 60)
        print(f"Combined Word Cloud ({mode_label})")
        print("=" * 60)

        all_descriptions = []
        for ckpt_path in all_checkpoints:
            print(f"  Loading: {ckpt_path.name}")
            checkpoint = load_checkpoint(str(ckpt_path))
            descriptions = extract_descriptions(checkpoint)
            all_descriptions.extend(descriptions)
            print(f"    {len(descriptions)} descriptions")

        print()
        print(f"Total descriptions: {len(all_descriptions)}")

        if use_ngrams:
            phrase_excl = set() if args.no_exclude_phrases else None
            word_counts = extract_mixed_ngrams(
                all_descriptions, exclusions,
                phrase_exclusions=phrase_excl,
                min_n=args.min_n, max_n=args.max_n
            )
        else:
            word_counts = extract_words(all_descriptions, exclusions)
        print(f"Unique {mode_label}: {len(word_counts)}")

        # Show top items
        print()
        print(f"Top {args.top} {mode_label}:")
        for word, count in word_counts.most_common(args.top):
            bar = "█" * min(count // 5, 40)
            print(f"  {word:30s} {count:5d}  {bar}")

        # Generate combined word cloud
        output_path = output_dir / "combined_wordcloud.png"
        if generate_wordcloud_image(word_counts, output_path, title="Combined"):
            print(f"\nSaved: {output_path}")

        # Save frequencies
        freq_file = output_dir / "combined_frequencies.json"
        with open(freq_file, 'w') as f:
            json.dump(dict(word_counts.most_common(200)), f, indent=2)
        print(f"Saved: {freq_file}")

    else:
        # Individual mode: one word cloud per checkpoint
        for i, ckpt_path in enumerate(all_checkpoints):
            print("=" * 60)
            print(f"Checkpoint {i+1}/{len(all_checkpoints)}")
            print("=" * 60)

            word_counts = process_checkpoint(
                ckpt_path, exclusions, output_dir,
                prefix=f"{i+1:02d}" if len(all_checkpoints) > 1 else "",
                use_ngrams=use_ngrams,
                min_n=args.min_n,
                max_n=args.max_n,
                no_exclude_phrases=args.no_exclude_phrases
            )

            if word_counts:
                print()
                print(f"Top {args.top} {mode_label}:")
                for word, count in word_counts.most_common(args.top):
                    print(f"  {word:40s} {count:5d}")
            print()

    print()
    print(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
