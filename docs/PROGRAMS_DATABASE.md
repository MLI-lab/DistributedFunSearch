# Programs Database

Stores priority functions and their metadata across multiple islands, constructs few-shot prompts by sampling from clusters, and tracks evolutionary metrics.

## Islands and Clusters

**Islands**: Independent populations that evolve separately, maintaining diversity. Each island contains multiple clusters.

**Clusters**: Within each island, priority functions are grouped by identical `scores_per_test`. Each cluster stores:
- `score`: Aggregate score (used for sampling weight)
- `scores_per_test`: Per-test-case scores, e.g., `{(6,1): 10, (7,1): 16}`
- `programs`: List of priority functions with optional metadata (e.g., thought/reasoning)

## Sampling

### Cluster Selection

Clusters are sampled with probability proportional to their scores using softmax. The temperature controls exploration vs exploitation: high temperature samples more uniformly across clusters, low temperature favors the best-scoring clusters.

Temperature decays cyclically over `cluster_sampling_temperature_period` stored functions, then resets. This creates alternating phases of exploration (early) and exploitation (late).

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

A hash is computed from the priority values the function assigns to each input. Two functions that produce identical priority mappings are considered duplicates and discarded, even if their code differs. This prevents the database from filling up with semantically equivalent functions.

## Metrics

### Progress
- `iterations`: Completed evolutionary cycles (prompt generated, sampled, evaluated, result processed)
- `total_stored_programs`: Functions that executed successfully and passed deduplication

### Rejections
- `execution_failed`: Execution failed or timed out
- `duplicates_discarded`: Identical output hash
- `version_mismatch_discarded`: Function was generated from a prompt whose island was reset before the result returned

### Parallel vs Sequential

Tracks if database changed between consecutive prompts:
- `parallel_prompts`: Same database state as previous prompt
- `sequential_prompts`: New functions were stored since previous prompt

### Resources
- `cumulative_evaluator_cpu_time`: Total CPU seconds evaluating
- `cumulative_sampler_gpu_time`: Total GPU seconds generating
- `cumulative_input_tokens` / `cumulative_output_tokens`: LLM token counts
- `cumulative_cost`: Total dollars spent (requires `cost_model` in SamplerConfig, uses LiteLLM pricing)

## Island Reset

Weak islands are periodically reset by copying the best performing function from a randomly sampled surviving island.

## Configuration

```python
ProgramsDatabaseConfig(
    num_islands=10,                              # Number of independent populations
    reset_programs=1200,                         # Programs per island before weak islands reset
    cluster_sampling_temperature_init=0.1,       # Starting temperature for cluster sampling
    cluster_sampling_temperature_period=30_000,  # Stored programs before temperature resets
    no_deduplication=False,                      # Skip hash-based deduplication
    save_lineage=False,                          # Track parent/child relationships
    initial_program_copies=100,                  # Copies of seed function per island at startup
    batch_size=1,                                # Messages to process per batch
    batch_timeout=0.01,                          # Max seconds to wait for batch to fill
)
```
