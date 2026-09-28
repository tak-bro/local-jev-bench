# System One local benchmark

Questions: `bench/questions_banking77.jsonl`
300 questions x 1 reps, sequential, 3 warm-up calls excluded.
`first-call p50` covers only each item's first call. Repeats of the same state can be served from an engine's
embedding cache (CLM caches state vectors), so all-call percentiles understate uncached latency.
Accuracy carries a 95% Wilson interval; overlapping intervals are not a difference.
`order-flip` asks each item once more with every choice's options reversed and counts changed choices.

| engine | cold ms | first-call p50 ms | all-call p50 ms | all-call p95 ms | samples | accuracy | order-flip | errors | worst memory pressure |
|---|---|---|---|---|---|---|---|---|---|
| ollaya | 580.6 | 27.8 | 27.8 | 30.3 | 300 | 180/300 (60%, 54-65) | 93/300 (31%) | 0 | normal |
| kev | 2017.5 | 240.0 | 240.0 | 285.4 | 300 | 266/300 (89%, 85-92) | 23/300 (8%) | 0 | normal |
| clm | 1127.9 | 359.7 | 359.7 | 523.7 | 300 | 61/300 (20%, 16-25) | 0/300 (0%) | 0 | warn |
| anyjev-raw | 3742.2 | 624.0 | 624.0 | 859.8 | 300 | 224/300 (75%, 69-79) | 68/300 (23%) | 0 | warn |
| anyjev-l0 | 8648.0 | 8489.3 | 8489.3 | 9218.3 | 300 | 241/300 (80%, 75-84) | 22/300 (7%) | 0 | normal |
