# Checkpoint Analysis Tools

Tools for analyzing DistributedFunSearch checkpoint files.

## Tools

| Script | Purpose |
|--------|---------|
| `descriptions.py` | Cluster descriptions using MinHash LSH + Louvain community detection |
| `make_wordcloud.py` | Generate word clouds from descriptions with customizable exclusions |
| `inspector.py` | View checkpoint stats, islands, clusters, best programs |
| `duplicates.py` | Analyze exact/near-duplicate descriptions |
| `clusters.py` | Alternative clustering using Union-Find (simpler, faster) |
| `similarity.py` | Code similarity functions (library, used by other scripts) |

## `descriptions.py`

Cluster descriptions using a sophisticated pipeline:

```
Descriptions
    ↓
Tokenize + remove stop words (sklearn ENGLISH_STOP_WORDS + domain terms)
    ↓
Analyze token length distribution
    ↓
Choose similarity metric based on CV (coefficient of variation):
    • CV < 0.4 → Jaccard (similar lengths)
    • CV ≥ 0.4 → Overlap coefficient (varying lengths)
    ↓
MinHash signatures (datasketch, num_perm=128)
    ↓
LSH banding to find candidate pairs (datasketch MinHashLSH)
    ↓
Compute actual similarity for candidates, filter by threshold
    ↓
Build similarity graph (NetworkX)
    ↓
Louvain community detection
    ↓
TF-IDF across clusters → top-10 words per cluster as labels
```

### Usage

```bash
# Single checkpoint
python descriptions.py checkpoint.pkl

# Folder (uses latest checkpoint)
python descriptions.py ./checkpoints/

# Custom threshold (lower = bigger clusters)
python descriptions.py checkpoint.pkl --threshold 0.4

# Custom CV threshold for metric selection
python descriptions.py checkpoint.pkl --cv-threshold 0.5

# Specify output directory
python descriptions.py checkpoint.pkl --output ./my_analysis/
```

### Output files

- `cluster_analysis.json` - Full results with all parameters and cluster data
- `cluster_summaries.txt` - Human-readable cluster summaries with examples
- `cluster_assignments.json` - Maps each description to its cluster

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | 0.5 | LSH/similarity threshold for clustering |
| `--num-perm` | 128 | Number of MinHash permutations |
| `--cv-threshold` | 0.4 | CV threshold for Jaccard vs Overlap selection |
| `--top` | 20 | Number of top clusters to display |
| `--output` | auto | Output directory |

### Dependencies

```bash
pip install datasketch networkx scikit-learn numpy
```

## `make_wordcloud.py`

Generate word clouds from checkpoint descriptions using n-grams (2-5 word phrases).

```bash
# Single checkpoint (default: n-grams)
python make_wordcloud.py checkpoint.pkl

# Multiple checkpoints (combined by default)
python make_wordcloud.py run1/ckpt.pkl run2/ckpt.pkl

# Folder, one word cloud per checkpoint
python make_wordcloud.py ./checkpoints/ --each

# Single words instead of phrases
python make_wordcloud.py checkpoint.pkl --single-words

# Add extra words to exclude
python make_wordcloud.py checkpoint.pkl --exclusions extra_words.txt
```

Output: `*_wordcloud.png`, `*_frequencies.json`

Requires: `pip install wordcloud matplotlib`

## `inspector.py`

View detailed stats for a single checkpoint.

```bash
python inspector.py checkpoint.pkl
python inspector.py checkpoint.pkl --island 0
python inspector.py checkpoint.pkl --export best_programs.txt
python inspector.py checkpoint.pkl --list-signatures
```

Shows: resource usage (CPU/GPU time, tokens), program statistics, island summaries, best programs.

## `duplicates.py`

Analyze description duplicates within a single run.

```bash
python duplicates.py checkpoint.pkl
python duplicates.py checkpoint.pkl --sample 2000
python duplicates.py checkpoint.pkl --top 20
```

Computes: exact duplicates (hash-based), near-duplicate distribution (sampled Jaccard), top duplicate clusters.

## `clusters.py`

Alternative clustering using hand-rolled MinHash/LSH + Union-Find. Simpler and faster than `descriptions.py` but less sophisticated (no Louvain, no TF-IDF labels).

```bash
python clusters.py checkpoint.pkl
python clusters.py checkpoint.pkl --threshold 0.6
python clusters.py checkpoint.pkl --jaccard  # Use Jaccard instead of overlap
```

## `similarity.py`

Library of code similarity functions used by other scripts.

```python
from checkpoint.similarity import compare_one_code_similarity_with_protection

similarity = compare_one_code_similarity_with_protection(
    code1, code2,
    similarity_type="all",
    protected_vars=['node', 'G', 'n', 's']
)
```

Similarity measures: string similarity, bag of AST nodes, tree edit distance.
