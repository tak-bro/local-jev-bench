# System One local benchmark

30 questions x 3 reps, sequential, 3 warm-up calls excluded.
`first-call p50` covers only each item's first call. Repeats of the same state can be served from an engine's
embedding cache (CLM caches state vectors), so all-call percentiles understate uncached latency.
Accuracy over 30 hand-written items is a sanity check, not a benchmark.

| engine | cold ms | first-call p50 ms | all-call p50 ms | all-call p95 ms | samples | accuracy | errors | worst memory pressure |
|---|---|---|---|---|---|---|---|---|
| clm | 4302.8 | 184.0 | 3.0 | 211.5 | 90 | 126/315 (40%) | 0 | warn |
| ollaya | 498.8 | 40.4 | 39.5 | 54.4 | 90 | 237/315 (75%) | 0 | warn |
