"""
DEPRECATED - This file has been split into two separate tools:

1. checkpoint_inspector.py - For analyzing a SINGLE checkpoint file
   Usage: python checkpoint_inspector.py <checkpoint_path> [options]

   Examples:
       python checkpoint_inspector.py checkpoint.pkl
       python checkpoint_inspector.py checkpoint.pkl --island 0
       python checkpoint_inspector.py checkpoint.pkl --list-signatures
       python checkpoint_inspector.py checkpoint.pkl --search-signature "{(7,2,2): 4}"

2. checkpoint_searcher.py - For searching across MULTIPLE checkpoints
   Usage: python checkpoint_searcher.py <checkpoint_dir> <command> [argument]

   Commands:
       first-signature SIGNATURE   Find first occurrence of a signature
       by-prompts COUNT            Find checkpoint closest to prompt count
       by-programs COUNT           Find checkpoint closest to program count
       first-score-above SCORE     Find first checkpoint exceeding score
       compare                     Compare all checkpoints (progress table)
       list-signatures             List all unique signatures found

   Examples:
       python checkpoint_searcher.py Checkpoints/ first-signature "{(7,2,2): 4, (8,2,2): 5}"
       python checkpoint_searcher.py Checkpoints/ by-prompts 50000
       python checkpoint_searcher.py Checkpoints/ compare
"""

import sys

def main():
    print(__doc__)
    print("=" * 80)
    print("Please use one of the new tools listed above.")
    print("=" * 80)
    return 1

if __name__ == "__main__":
    sys.exit(main())
