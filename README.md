# local-jev-bench: local Jev-style decision models on Apple Silicon (AnyJev, Kev, Laya, CLM)

English | [한국어](README.ko.md)

local-jev-bench runs Jev-style System One decision models (typed questions in, calibrated answers out) locally on an
Apple Silicon Mac, and benchmarks them on the same questions: English and Korean customer tickets, and the public
BANKING77 20-way intent set.

**Key results (M3 Max 36 GB, 2026-09-29).** Kev-4B is the most accurate engine on every set: 279/315 English,
273/315 Korean, 266/300 BANKING77-20, at 215-240 ms per call. Ollaya (Laya) is the only engine under 50 ms per call
(20.7-35.8 ms) and loses accuracy with many options. AnyJev on Qwen3-8B reproduces its own README on BANKING77-20
(order-flip 68/300 → 22/300). Details in [Results](#results-2026-09-29-m3-max-36-gb).

It sets up four engines:

- **CLM** ([Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM)): a Qwen3-8B encoder served by [vllm-metal](https://github.com/vllm-project/vllm-metal), plus CLM's 75 MB head.
- **Ollaya** ([ollaya-dev/ollaya](https://github.com/ollaya-dev/ollaya)): the `laya` model, run on MLX.
- **AnyJev** ([nokia-applied-research/AnyJev](https://github.com/nokia-applied-research/AnyJev)): training-free; reads label logprobs from Qwen3-8B on vllm-metal, with no correction (`anyjev-raw`) or with cyclic option shifts plus a batch prior (`anyjev-l0`).
- **Kev** ([jaredpalmer/kev](https://github.com/jaredpalmer/kev)): a LoRA on Qwen3.5-4B-Base, run on MLX.

All speak TypeSafe's `POST /v1/systemone` wire format.

## Install

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal   # the tap needs the URL
brew install vllm-project/vllm-metal/vllm-metal
OLLAYA_INSTALL_DIR=$HOME/.local OLLAYA_NO_SERVICE=1 sh -c "$(curl -fsSL https://ollaya.dev/install.sh)"
uv sync
```

[AnyJev](https://github.com/nokia-applied-research/AnyJev) comes from `uv sync`. Its vLLM backend reads label logprobs over HTTP and imports `transformers` only for the tokenizer, so torch is not needed. `serve/anyjev_server.py` wraps it in the `/v1/systemone` format. The request's `model` picks the correction level (`anyjev-raw` or `anyjev-l0`). The adapter asks the generate server for each label's logprob by id (`logprob_token_ids`) rather than AnyJev's `allowed_token_ids` + top-K, because vllm-metal 0.30.0 reports logprobs before that filter and a label can fall out of the top K. A label token missing from the server's logprobs is an HTTP 502, not a filled-in probability.

[Kev](https://github.com/jaredpalmer/kev) keeps its own uv environment (torch, mlx-lm) in its checkout, so it is not a dependency here. `git clone https://github.com/jaredpalmer/kev ~/workspace/tak-bro/kev && (cd ~/workspace/tak-bro/kev && uv sync --extra serve)` sets it up. `scripts/serve-kev.sh` runs its `/v1/systemone` server with the `jaredpalmer/kev-4b` adapter on Qwen3.5-4B-Base.

`contrastive-lm` declares `vllm` as a dependency, but it only calls the embeddings endpoint over HTTP. `pyproject.toml` overrides that dependency away so a second vLLM is not installed.

## Ports

Every server binds to `127.0.0.1` only.

| Port | Process | Start |
|---|---|---|
| 8091 | small embedding model (smoke only) | `scripts/serve-embed.sh mlx-community/Qwen3-Embedding-0.6B-8bit 8091 embed-small` |
| 8090 | Qwen3-8B encoder for CLM | `scripts/serve-embed.sh Qwen/Qwen3-8B 8090 qwen3-8b` |
| 8700 | CLM System One API | `scripts/serve-clm.sh` |
| 11435 | Ollaya daemon | `OLLAYA_HOST=127.0.0.1:11435 ~/.local/bin/ollaya serve` |
| 8092 | Qwen3-8B generate server for AnyJev | `scripts/serve-llm.sh Qwen/Qwen3-8B 8092 qwen3-8b` |
| 8710 | AnyJev System One API (`anyjev-raw`, `anyjev-l0`) | `scripts/serve-anyjev.sh` |
| 8009 | Kev System One API (`kev-latest`) | `KEV_DIR=~/workspace/tak-bro/kev scripts/serve-kev.sh` |

The serve scripts refuse to start when their port is already taken.

`serve-embed.sh` pins three settings that were needed on this Mac (M3 Max, 36 GB, vllm-metal 0.30.0):

- `GLOO_SOCKET_IFNAME=lo0 VLLM_HOST_IP=127.0.0.1`: without these, the engine hangs at init because torch.distributed cannot resolve the `.local` hostname.
- `--pooler-config '{"seq_pooling_type": "LAST", "use_activation": true}'`: last-token pooling with L2 normalisation, which is what CLM's head expects. Without it, vLLM falls back to the architecture default.
- `--gpu-memory-utilization 0.7`: vllm-metal applies the fraction to the Metal wired limit (28.1 GB here). The default of 0.92 pushed memory pressure to warn. 0.55 left no KV cache for the 16 GB of bf16 weights.

## Check and benchmark

```bash
scripts/smoke-embed.sh 8091                   # vector returned, L2-normalised
scripts/smoke-embed.sh 8090 4096              # Qwen3-8B: 4096 dims, normalised
uv run python bench/run.py --smoke --engine clm      # CLM README example within tolerance
uv run python bench/run.py --smoke --engine ollaya   # same example, shape only
uv run python bench/run.py                           # 30 questions x 3 reps on CLM and Ollaya, with kernel memory pressure
uv run python bench/run.py --smoke --engine anyjev-raw --engine anyjev-l0 --engine kev   # shape only
uv run python bench/run.py --engine <e> --reps 1 --questions bench/questions_banking77.jsonl
uv run python bench/score.py bench/runs/questions --check   # report.md matches the raw logs
uv run python bench/make_sets.py transfer-v4        # Kev's out-of-distribution dev set (764) into bench/data/, gitignored
uv run python bench/make_sets.py typed-decisions    # LocalLLaMA/typed-decisions test (400 cases, 2,000 decisions) with gold distributions
uv run pytest -q                              # offline tests, fake servers, no model loaded
```

AnyJev and Kev each need most of the Metal memory, so every engine is measured alone: start its servers, run
`bench/run.py --engine <e> --questions <set>`, stop them. Each run logs every call (answers with probabilities,
latency, memory pressure) to `bench/runs/<set>/<engine>.jsonl`, replacing that engine's earlier log, and
`bench/score.py` rebuilds `bench/runs/<set>/report.md` from all the logs in that directory. It refuses a log measured
on another version of the set or with another `--reps`. Restart the AnyJev adapter before each run: the L0 batch prior
accumulates across every call the adapter has served.

The benchmark rejects any item whose state plus question instructions (the text CLM actually embeds) exceeds 2048 Qwen3 tokens, because `clm-serve` would otherwise truncate it without an error. Treat the accuracy over 30 hand-written items as a sanity check, not a benchmark.

## Results (2026-09-29, M3 Max 36 GB)

30 questions x 3 reps per set, one engine running at a time. Accuracy carries a 95% Wilson interval; `order-flip`
counts items whose choice changed when the options were listed in reverse. Full tables: `bench/report.md` (English)
and `bench/report_ko.md` (Korean). Ollaya ran `laya` on English and `laya:multilingual` on Korean
(`OLLAYA_MODEL=laya:multilingual`); on Korean the English model scored 33% (2026-09-28).

| engine | first-call p50 ms (en / ko) | accuracy en | accuracy ko | order-flip (en / ko) |
|---|---|---|---|---|
| Kev-4B (MLX, bf16) | 225.9 / 215.0 | 279/315 (89%, 85-92) | 273/315 (87%, 82-90) | 1/30 / 1/30 |
| AnyJev L0 (Qwen3-8B, vllm-metal) | 434.5 / 444.1 | 254/315 (81%, 76-85) | 264/315 (84%, 79-87) | 0/30 / 0/30 |
| AnyJev raw (Qwen3-8B, vllm-metal) | 178.5 / 182.2 | 258/315 (82%, 77-86) | 261/315 (83%, 78-87) | 0/30 / 2/30 |
| Ollaya (`laya` en / `laya:multilingual` ko, MLX) | 35.8 / 20.7 | 237/315 (75%, 70-80) | 207/315 (66%, 60-71) | 2/30 / 6/30 |
| CLM-8B (vllm-metal, bf16) | 225.7 / 266.5 | 123/315 (39%, 34-45) | 129/315 (41%, 36-46) | 0/30 / 0/30 |

No run reached `critical` memory pressure (worst: `warn`, for CLM). In the run log, the first AnyJev runs after the
generate server started took about 1 s per call (warm-up); those rows were discarded, and both AnyJev rows above
come from runs on a warmed server.

CLM picks a choice by comparing embeddings of the option texts, so its 0 order-flips may be structural rather than
a sign of order robustness (not verified). Do not compare its order-flip column with the others.

Which engine to use (intervals that overlap count as no difference):

- **Under 50 ms per call:** only Ollaya fits (35.8 ms en, 20.7 ms ko first-call p50). On English only Kev beats it
  (279 vs 237, 85-92 vs 70-80); on Korean Kev and both AnyJev levels do (273, 264, 261 vs 207; 60-71 is clear of
  82-90, 79-87 and 78-87).
- **Korean:** Kev or AnyJev; they cannot be told apart (82-90 vs 79-87 / 78-87). Kev needs one 4B model (server RSS
  2.7 GB after startup in the 2026-09-29 smoke run); AnyJev needs Qwen3-8B on vllm-metal.
- **Cannot be told apart on these 30 items:** Kev vs either AnyJev level on both sets (on English, Kev vs L0 only
  just: Kev's lower bound 84.58 against L0's upper 84.62), AnyJev raw vs L0, and either AnyJev level vs Ollaya on English.

Earlier findings on CLM (2026-09-28):

- The vllm-metal embeddings match transformers (MPS, bf16, last-token, L2) with cosine ≥ 0.9998 on four probe texts, so the encoder side is faithful.
- CLM reproduces the README's `department` (billing, 0.987 vs 0.939) and `frustration` (2.00 vs 1.98), but gives `urgency` 0.852 against the README's 0.41. Laya gives 0.795 on the same ticket. The cause is unresolved.
- On this question set CLM answers `frustration` at about 2.0 every time and leans toward `billing` for `department`.

### BANKING77 20-way (2026-09-29)

`bench/questions_banking77.jsonl` restates AnyJev's `banking20` task, so the numbers can be set beside its README.
It uses the 20 intents most frequent in BANKING77 train, listed by label id with no descriptions, and the question
"What is the customer's intent?". It keeps the first 300 test items after `random.Random(0)`, read from
`mteb/banking77` at a pinned revision. `uv run bench/make_banking77.py` regenerates it. One call per item
(`--reps 1`); full table: `bench/report_banking77.md`.

| engine | first-call p50 ms | accuracy | order-flip |
|---|---|---|---|
| Kev-4B | 240.0 | 266/300 (89%, 85-92) | 23/300 (8%) |
| AnyJev L0 | 8489.3 | 241/300 (80%, 75-84) | 22/300 (7%) |
| AnyJev raw | 624.0 | 224/300 (75%, 69-79) | 68/300 (23%) |
| Ollaya `laya` | 27.8 | 180/300 (60%, 54-65) | 93/300 (31%) |
| CLM-8B | 359.7 | 61/300 (20%, 16-25) | 0/300 (0%) |

- **AnyJev reproduces its README here.** Its README reports, for Qwen3-8B on BANKING77 20-way with 300 test items,
  order-flip 0.230 raw → 0.073 L0 and accuracy 0.747 → 0.803. This run gives 68/300 (0.227) → 22/300 (0.073) and
  224/300 (0.747) → 241/300 (0.803).
- **L0 cuts order-flips; its accuracy gain is not shown on 300 items.** Order-flips fall from 68/300 to 22/300,
  while the accuracy intervals 69-79 and 75-84 overlap.
- **Kev is the most accurate engine here** (85-92, clear of every other interval) within one order-flip of L0
  (23/300 vs 22/300), and 240.0 ms against L0's 8489.3 ms. L0 prefills the prompt once per cyclic shift, up to 20 times on a
  20-way question.
- **Ollaya falls to 60% with 20 options** and flips 31% of its choices when they are reversed.
- For scale, not comparison: Laya zero-shot is quoted at 38% on all 77 intents, against 76% for Jev
  (dhruvmehra/jevbench, as reported in a Laya fine-tuning write-up; not verified here). That set is 77-way, this
  one 20-way.

## FAQ

**What is a System One decision model?**
A model that answers typed questions about a piece of text (a choice among options, a yes/no probability, or a
score) in one forward pass, with calibrated probabilities instead of generated text. TypeSafe's Jev is the hosted
original; its `POST /v1/systemone` format is what every engine here speaks.

**Which local decision model is the most accurate on a Mac?**
Kev-4B in these runs: 89% on the English and BANKING77-20 sets and 87% on Korean, clear of every other engine on
BANKING77-20 (Wilson 85-92).

**Which one is the fastest?**
Ollaya running Laya: 20.7-35.8 ms first-call p50, against 178.5-444.1 ms for the others on the 30-item sets.

**Does AnyJev work on vllm-metal?**
Yes, with one change: vllm-metal 0.30.0 reports raw logprobs whatever `--logprobs-mode` says, so the adapter asks for
each label by `logprob_token_ids`. With that, AnyJev reproduces its README on BANKING77-20.

**Can these models answer in Korean?**
Kev (273/315) and AnyJev (261-264/315) can; they cannot be told apart on 30 items. Ollaya needs
`laya:multilingual` and reaches 207/315.

**How much memory do they need?**
Kev-4B's server used 2.7 GB RSS after startup. AnyJev and CLM run Qwen3-8B on vllm-metal, which reserves about
20 GB (0.7 of the 28.1 GB Metal wired limit); measure them one at a time.

## vllm-metal vs Ollama, Qwen3-8B chat (2026-09-28, M3 Max 36 GB)

`scripts/serve-llm.sh` serves a chat model through vllm-metal. `bench/llm_compare.py` sends the same 8 prompts (max 256 tokens, temperature 0, thinking off) to any OpenAI-compatible server and appends a row to `bench/llm_report.md`.

| server | TTFT p50 ms | single decode tok/s | 8 concurrent, total tok/s | memory |
|---|---|---|---|---|
| vllm-metal, bf16 | 197 | 16.8 | 77.3 | reserves ~20 GB (0.7 × wired limit) |
| vllm-metal, MLX 4bit | 291 | 30.2 | 48.6 | same reservation |
| Ollama 0.34.2, Q4 (`qwen3:8b`) | 233 | 29.4 | 30.2 | 10 GB (32K context) |

- For a single user, 4-bit models decode at about the same speed on vllm-metal and Ollama (about 30 tok/s). Ollama uses half the memory.
- Under 8 concurrent requests only vllm-metal batches. Ollama's default settings (left untouched) served them at single-request speed.
- Ollama's OpenAI endpoint ignores `/no_think` and `think: false`. Send `reasoning_effort: "none"` to turn thinking off, or `content` stays empty until the thinking tokens finish.
- These numbers come from one run with 8 prompts each and are not repeated.

## License

MIT. See [LICENSE](LICENSE). The benchmarked engines and datasets keep their own licenses.
