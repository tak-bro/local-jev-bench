# System One local benchmark

Questions: `bench/questions_ko.jsonl`
30 questions x 3 reps, sequential, 3 warm-up calls excluded.
`first-call p50` covers only each item's first call. Repeats of the same state can be served from an engine's
embedding cache (CLM caches state vectors), so all-call percentiles understate uncached latency.
Accuracy carries a 95% Wilson interval; overlapping intervals are not a difference.
`order-flip` asks each item once more with every choice's options reversed and counts changed choices.

| engine | cold ms | first-call p50 ms | all-call p50 ms | all-call p95 ms | samples | accuracy | order-flip | errors | worst memory pressure |
|---|---|---|---|---|---|---|---|---|---|
| kev | 264.5 | 215.0 | 195.1 | 266.2 | 90 | 273/315 (87%, 82-90) | 1/30 (3%) | 0 | normal |
| ollaya | 1153.7 | 20.7 | 21.1 | 25.8 | 90 | 207/315 (66%, 60-71) | 6/30 (20%) | 0 | normal |
| clm | 577.8 | 266.5 | 2.0 | 275.0 | 90 | 129/315 (41%, 36-46) | 0/30 (0%) | 0 | warn |
| anyjev-l0 | 799.7 | 444.1 | 228.5 | 514.2 | 90 | 264/315 (84%, 79-87) | 0/30 (0%) | 0 | normal |
| anyjev-raw | 420.9 | 182.2 | 182.8 | 206.2 | 90 | 261/315 (83%, 78-87) | 2/30 (7%) | 0 | normal |
