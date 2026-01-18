# Programs Database

Stores priority functions across multiple islands, constructs few-shot prompts by sampling from clusters, and tracks evolutionary metrics.

## Flow

**get_prompt()**: Pick random island, sample clusters weighted by score, sample one function per cluster, publish prompt with island_id.

**process_message()**: Receive result from evaluator. If execution failed, trigger new prompt. If success, check for duplicate hash, register if unique, trigger new prompt.

## Islands and Clusters

**Islands**: Independent populations that evolve separately, maintaining diversity.

**Clusters**: Priority functions grouped by identical `scores_per_test`. Each cluster stores:
- `score`: Aggregate score (used for sampling weight)
- `scores_per_test`: Per-test-case scores, e.g., `{(6,1): 10, (7,1): 16}`
- `programs`: List of functions with identical `scores_per_test`

## Sampling

### Cluster Selection

Clusters sampled using temperature-scaled softmax over scores:

```
probs = softmax(scores / temperature)
```

Temperature decays cyclically from `cluster_sampling_temperature_init` toward a minimum floor over `cluster_sampling_temperature_period` functions, then resets:
- High temp (start): explores diverse clusters
- Low temp (end): exploits best clusters

### Function Selection

Within each cluster, shorter functions are preferred:

```
probs = softmax(-normalized_lengths)
```

### Edge Cases

| Case | Handling |
|------|----------|
| No clusters | Skip prompt |
| Fewer clusters than needed | Use all available |
| Softmax fails | Uniform fallback |
| Duplicate few-shot examples | Flag and count |

## Deduplication

Functions with identical output hashes are discarded:

```python
if hash_value in island['hash_set']:
    duplicates_discarded += 1
    return
```

## Metrics

### Progress
- `iterations`: Completed evolutionary cycles (prompt generated, sampled, evaluated, result processed)
- `total_stored_programs`: Functions that executed successfully and passed deduplication
- `database_version`: Increments on each store

### Rejections
- `execution_failed`: Execution failed or timed out
- `duplicates_discarded`: Identical output hash
- `version_mismatch_discarded`: Outdated island version

### Parallel vs Sequential

Tracks if database changed between consecutive prompts:
- `parallel_prompts`: Same database state as previous prompt
- `sequential_prompts`: New functions were stored since previous prompt

### Resources
- `cumulative_evaluator_cpu_time`: Total CPU seconds evaluating
- `cumulative_sampler_gpu_time`: Total GPU seconds generating
- `cumulative_input_tokens` / `cumulative_output_tokens`: LLM token counts

## Island Reset

Weak islands are periodically reset by copying the best performing function from a randomly sampled surviving island.

## Configuration

```python
DatabaseConfig(
    num_islands=10,
    reset_programs=1200,
    cluster_sampling_temperature_init=0.1,
    cluster_sampling_temperature_period=30_000,
    no_deduplication=False,
    save_lineage=False,  # Track parent-child relationships between functions
    batch_size=10,  # Number of messages to process per batch
    batch_timeout=0.01,  # Max seconds to wait for batch to fill
)
```
