# local-jev-bench: local Jev-style decision models on Apple Silicon (Kev, Winnow, Clef, Von, Jeff, AnyJev, Laya, CLM)

English | [한국어](README.ko.md)

local-jev-bench runs Jev-style System One decision models (typed questions in, calibrated answers out) locally on an
Apple Silicon Mac, and benchmarks them on the same questions. The sets are English and Korean customer tickets, the
public BANKING77 20-way intent set, Kev's out-of-distribution transfer-v4 set, the typed-decisions leaderboard set,
and two Korean sets: NSMC movie reviews and KLUE-YNAT news headlines.

**Key results (M3 Max 36 GB, 2026-09-30; Clef, Clef-flash and Von 2026-10-05).** Kev-9B, Kev-4B, Winnow-E4B, Clef
and Clef-flash are the most accurate engines overall, and paired McNemar tests separate them on only some sets:

- **transfer-v4:** Kev beats Winnow (Kev-9B 81% vs 77%, p = 0.003), but only on the set's two held-out
  policy-structure sources, built from the kind of programmatic policy data Kev trains on (Kev-4B 160/176, Winnow
  125/176). On the six public sources Kev never trained on, the three are level (454, 461 and 460 of 588, p > 0.45), and so is
  training-free AnyJev (458 raw and L0).
- **typed-decisions:** Winnow beats Kev-4B (73% vs 67%, p < 0.001) and is level with Kev-9B (72%, p = 0.529).
- **NSMC and KLUE-YNAT (Korean):** the three, Clef-flash and Clef cannot be told apart, except Clef over Kev-4B on
  NSMC (p = 0.043) and over Kev-9B on KLUE-YNAT (p = 0.047).
- **AnyJev (training-free Qwen3-8B):** level with the top group on KLUE-YNAT (75-76%) and with Winnow on transfer-v4
  (75%), but behind it on typed-decisions (61-63%), and far from the gold distributions there (KL 2.8-3.8 against
  0.2-0.3). It reproduces its own README on BANKING77-20 again, and paired, L0's accuracy gain over raw is real there
  (p = 0.005).
- **Clef-flash (Cloudflare, 9B, measured 2026-10-05):** joins the top group. It is level with Kev-9B on every set
  but BANKING77-20, where it leads (97% vs 89%, p < 0.001; Clef-flash does not publish its training data). On
  transfer-v4 it beats Winnow (82% vs 77%, p = 0.003), again only on the two held-out policy sources (152/176 vs
  125/176); on the six public sources it is level with Kev, Winnow and AnyJev (471 of 588, p > 0.08). It has the
  lowest ECE of all engines on 4 of the 7 sets (tied with Clef on BANKING77-20; typed-decisions 0.021, KL 0.220) and never changes a choice when the
  options are reversed. It is also slow here: 488-2557 ms first-call p50 on llama.cpp, 1.4-2.3x Kev-9B (whose rows
  come from a busier machine), against the 38.8 ms median Cloudflare reports from its own GPUs.
- **Clef (Cloudflare, 27B, measured 2026-10-05):** cannot be told apart from Clef-flash on any set (p ≥ 0.092) and
  costs 3.3-4.5x its time (1600-8923 ms first-call p50, Q4_K_M; Cloudflare reports a 209.3 ms median on its own
  GPUs). Against Kev-9B it is level everywhere but BANKING77-20
  (96% vs 89%, p < 0.001) and KLUE-YNAT (78% vs 73%, p = 0.047, one of 55 pairs on that set). It is the closest of all
  engines to typed-decisions' gold distributions (KL 0.196).
- **Von (395M ModernBERT, measured 2026-10-05):** English only, as its card says: 15% on KLUE-YNAT, 49% on the Korean
  30 and 51% on NSMC (a two-way set). On the English sets it is below the top group (69% transfer-v4, 43%
  typed-decisions), at 32-411 ms first-call p50.

Ollaya (Laya) is the fastest engine, at 15-55 ms first-call p50 outside typed-decisions' long states. It is also the least
accurate engine apart from CLM and Von on every set but the 30 English items, where Jeff is lower (Von, which is
English only, is lower still on the three Korean sets), and the worst calibrated
(ECE) of Kev, Winnow, Jeff and Ollaya on the same sets; AnyJev and CLM are worse calibrated still on some. Kev-0.8B takes 28-58 ms on the same sets and is as
accurate as Ollaya or more on every set. Kev-4B reproduces its model card on transfer-v4 (534/656 against 0.817). Details are in
[Results](#results-2026-09-30-m3-max-36-gb).

It sets up nine engines:

- **CLM** ([Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM)): a Qwen3-8B encoder served by [vllm-metal](https://github.com/vllm-project/vllm-metal), plus CLM's 75 MB head.
- **Ollaya** ([ollaya-dev/ollaya](https://github.com/ollaya-dev/ollaya)): the `laya` model, run on MLX.
- **Winnow** ([ollaya.dev/library/winnow](https://ollaya.dev/library/winnow)): `winnow:e4b`, built on Gemma 4 E4B and run by Ollaya on llama.cpp (Q8_0).
- **AnyJev** ([nokia-applied-research/AnyJev](https://github.com/nokia-applied-research/AnyJev)): training-free; reads label logprobs from Qwen3-8B on vllm-metal, with no correction (`anyjev-raw`) or with cyclic option shifts plus a batch prior (`anyjev-l0`).
- **Kev** ([jaredpalmer/kev](https://github.com/jaredpalmer/kev)): a LoRA plus pointer head on Qwen3.5-0.8B, 4B or 9B Base (`kev-0.8b`, `kev-4b`, `kev-9b`), run on MLX.
- **Jeff** ([firelex/jeff](https://github.com/firelex/jeff)): Jeff-Qwen3.5-2B, a fine-tuned Qwen3.5-2B with a trained answer readout, run on MLX.
- **Clef-flash** ([Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)): Cloudflare's 9B decision model on Qwen3.5-9B, as `ggml-org/Clef-Flash-GGUF` Q8_0, served by llama.cpp's `llama-server` (`clef-flash`). It decides all the questions of a request jointly, in one prompt; the other engines answer each question on its own.
- **Clef** ([Cloudflare/clef](https://huggingface.co/Cloudflare/clef)): the 27B sibling on Qwen3.8-27B, as `ggml-org/Clef-GGUF` Q4_K_M on the same `llama-server` (`clef`). Q8_0 (28.7 GB) leaves too little of the 36 GB.
- **Von** ([wfzyx/von](https://github.com/wfzyx/von)): a 395M ModernBERT encoder with an option-marker head, served by its own `von serve` on Metal (`von`).

All speak TypeSafe's `POST /v1/systemone` wire format.

## Install

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal   # the tap needs the URL
brew install vllm-project/vllm-metal/vllm-metal
OLLAYA_INSTALL_DIR=$HOME/.local OLLAYA_NO_SERVICE=1 sh -c "$(curl -fsSL https://ollaya.dev/install.sh)"
uv sync
```

[AnyJev](https://github.com/nokia-applied-research/AnyJev) comes from `uv sync`. Its vLLM backend reads label logprobs over HTTP and imports `transformers` only for the tokenizer, so torch is not needed. `serve/anyjev_server.py` wraps it in the `/v1/systemone` format. The request's `model` picks the correction level (`anyjev-raw` or `anyjev-l0`). The adapter asks the generate server for each label's logprob by id (`logprob_token_ids`) rather than AnyJev's `allowed_token_ids` + top-K, because vllm-metal 0.30.0 reports logprobs before that filter and a label can fall out of the top K. A label token missing from the server's logprobs is an HTTP 502, not a filled-in probability.

[Kev](https://github.com/jaredpalmer/kev) keeps its own uv environment (torch, mlx-lm) in its checkout, so it is not a dependency here. `git clone https://github.com/jaredpalmer/kev ~/workspace/tak-bro/kev && (cd ~/workspace/tak-bro/kev && uv sync --extra serve)` sets it up. `scripts/serve-kev.sh` runs its `/v1/systemone` server with the `jaredpalmer/kev-4b` adapter on Qwen3.5-4B-Base; `KEV_RUN=jaredpalmer/kev-0.8b` or `jaredpalmer/kev-9b` serves another size on the same port. The engines `kev-0.8b`, `kev-4b` and `kev-9b` check `/v1/models` before a run and refuse a server running another size.

[Winnow](https://ollaya.dev/library/winnow) runs in Ollaya: `ollaya pull winnow:e4b` (8.0 GB, Gemma 4, Q8_0), measured as the `winnow` engine. `ollaya stop winnow:e4b` unloads it.

[Jeff](https://github.com/firelex/jeff) keeps its own uv environment too, and its `pyproject.toml` requires uv 0.12.19 or newer, hence `uvx`: `git clone https://github.com/firelex/jeff ~/workspace/tak-bro/jeff && git -C ~/workspace/tak-bro/jeff checkout f06788292874c21a5b5c41549ac220dd9e15da7f`, then in it `uvx --from 'uv>=0.12.19' uv sync --no-default-groups --extra mac` and `uvx --from 'uv>=0.12.19' uv run --no-default-groups hf download mstrasser/Jeff-Qwen3.5-2B --local-dir checkpoints/jeff-2b`. `scripts/serve-jeff.sh` serves it on MLX, which runs Jeff's Qwen models only, so Jeff-Gemma4-E2B is not used here.

[Clef-flash](https://huggingface.co/ggml-org/Clef-Flash-GGUF) needs llama.cpp build 11403: `/v1/systemone` and the clef architecture landed after Homebrew's 0.5.0. Unpack the [b11403 release](https://github.com/ggml-org/llama.cpp/releases/tag/b11403) `llama-b11403-bin-macos-arm64.tar.gz` into `~/.local/opt/llama.cpp/b11403/` (or point `LLAMA_SERVER` at its `llama-server`). `scripts/serve-llama.sh clef-flash` (or `clef`) checks the build, downloads the GGUF at a pinned revision (9.7 GB, or 19.2 GB for Clef) and serves it with the whole prompt in one batch (`-ub 8192`), which clef requires.

[Von](https://github.com/wfzyx/von) runs from PyPI through `uvx` (`von-sdk==1.3.7`), with weights pinned to one revision of `wfzyx/von` (3.2 GB). `scripts/serve-von.sh` serves it on Metal with `--noul-decision raw`: by default Von moves every noul probability outside 0.2-0.8, which keeps the decision but not the calibration scored here.

`contrastive-lm` declares `vllm` as a dependency, but it only calls the embeddings endpoint over HTTP. `pyproject.toml` overrides that dependency away so a second vLLM is not installed.

## Ports

Every server binds to `127.0.0.1` only.

| Port | Process | Start |
|---|---|---|
| 8091 | small embedding model (smoke only) | `scripts/serve-embed.sh mlx-community/Qwen3-Embedding-0.6B-8bit 8091 embed-small` |
| 8090 | Qwen3-8B encoder for CLM | `scripts/serve-embed.sh Qwen/Qwen3-8B 8090 qwen3-8b` |
| 8700 | CLM System One API | `scripts/serve-clm.sh` |
| 11435 | Ollaya daemon (`laya`, `winnow:e4b`) | `OLLAYA_HOST=127.0.0.1:11435 ~/.local/bin/ollaya serve` |
| 8092 | Qwen3-8B generate server for AnyJev | `scripts/serve-llm.sh Qwen/Qwen3-8B 8092 qwen3-8b` |
| 8710 | AnyJev System One API (`anyjev-raw`, `anyjev-l0`) | `scripts/serve-anyjev.sh` |
| 8009 | Kev System One API (`kev-latest`) | `KEV_RUN=jaredpalmer/kev-4b scripts/serve-kev.sh` (or `kev-0.8b`, `kev-9b`) |
| 8765 | Jeff System One API (`jeff-latest`) | `scripts/serve-jeff.sh` |
| 8020 | llama-server System One API (`Clef-Flash-Q8_0@4a7a08c` or `Clef-Q4_K_M@5f70656`) | `scripts/serve-llama.sh clef-flash` (or `clef`) |
| 8030 | Von System One API (`von-latest`) | `scripts/serve-von.sh` |

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
uv run python bench/run.py --smoke --engine <e>       # shape only (anyjev-raw, anyjev-l0, kev-0.8b, kev-4b, kev-9b, winnow, jeff), its server up
uv run python bench/run.py --engine <e> --reps 1 --questions bench/questions_banking77.jsonl
uv run python bench/score.py bench/runs/questions --check   # report.md matches the raw logs
uv run python bench/make_sets.py transfer-v4        # Kev's out-of-distribution dev set (764) into bench/data/, gitignored
uv run python bench/make_sets.py typed-decisions    # LocalLLaMA/typed-decisions test (400 cases, 2,000 decisions) with gold distributions
uv run python bench/make_sets.py nsmc               # 300 Korean movie reviews, positive or not (HF card: CC BY 2.0)
uv run python bench/make_sets.py klue-ynat          # 300 Korean headlines, 7 topics (KLUE, CC BY-SA 4.0)
uv run pytest -q                              # offline tests, fake servers, no model loaded
```

AnyJev and Kev each need most of the Metal memory, so every engine is measured alone: start its servers, run
`bench/run.py --engine <e> --questions <set>`, stop them. `scripts/measure.sh [--reps N] <engine> <set>...` does
all three and stops the servers even when a step fails; server output goes to `logs/<name>.log`. Each run logs every call (answers with probabilities,
latency, memory pressure) to `bench/runs/<set>/<engine>.jsonl`, replacing that engine's earlier log once the run finishes, and
`bench/score.py` rebuilds `bench/runs/<set>/report.md` from all the logs in that directory. It refuses a log measured
on another version of the set or with another `--reps`. Restart the AnyJev adapter before each run: the L0 batch prior
accumulates across every call the adapter has served.

The benchmark rejects any item whose state plus question instructions (the text CLM actually embeds) exceeds 2048 Qwen3 tokens, because `clm-serve` would otherwise truncate it without an error. Treat the accuracy over 30 hand-written items as a sanity check, not a benchmark.

## Results (2026-09-30, M3 Max 36 GB)

`scripts/measure.sh` measured one engine at a time. The 30-item sets ran 3 reps; every other set ran 1 rep, plus a
reversed-option call where the set has choice questions (all but NSMC). Accuracy is graded per decision (item x question) from each item's first timed call. The full
tables are in `bench/runs/<set>/report.md`: counts with 95% Wilson intervals, Brier, ECE, order-flip, the pairwise
McNemar table, and per-source breakdowns. Each table is rebuilt from the raw logs next to it by
`bench/score.py <dir> --check`. Ollaya ran `laya` on the English sets and `laya:multilingual` on the Korean ones.
Clef, Clef-flash and Von were measured later, on 2026-10-05, against the same set files (same sha256), on an idle machine;
the other rows are from 2026-09-30. Their interpretation is in `reports/2026-10-05-analysis.md`.

Accuracy, first timed call (`†` = the engine trained on a train split of data in this set):

| engine | English 30 | Korean 30 | BANKING77-20 | transfer-v4 | typed-decisions | NSMC | KLUE-YNAT |
|---|---|---|---|---|---|---|---|
| decisions | 105 | 105 | 300 | 764 | 2,000 | 300 | 300 |
| Kev-9B | 90% | 90% | 89%† | 81% | 72% | 86% | 73% |
| Kev-4B | 89% | 87% | 89%† | 80% | 67% | 83% | 74% |
| Clef-flash | 86% | 85% | 97% | 82% | 71% | 86% | 77% |
| Clef | 90% | 91% | 96% | 83% | 72% | 87% | 78% |
| Winnow-E4B | 89% | 86% | 81% | 77% | 73% | 84% | 74% |
| Kev-0.8B | 75% | 75% | 88%† | 65% | 46% | 80% | 63% |
| Jeff-Qwen3.5-2B | 64% | 75% | 65% | 69%† | 52% | 79% | 74% |
| Ollaya | 75% | 66% | 60% | 63% | 36% | 56% | 43% |
| AnyJev L0 | 81% | 84% | 80% | 75% | 63% | 80% | 75% |
| AnyJev raw | 82% | 83% | 75% | 75% | 61% | 81% | 76% |
| Von | 73% | 49% | 80%† | 69%† | 43% | 51% | 15% |
| CLM-8B | 39% | 41% | 20% | - | - | - | - |

`†`: Kev (all three sizes) trained on BANKING77's train split, and Jeff on PAWS's. PAWS's test split is 80 of
transfer-v4's 764 decisions. Von's card lists Banking77 and dair-ai/emotion in its training corpus without naming
the split; transfer-v4's `emotion` source (116 decisions) is dair-ai/emotion. An engine without a mark either did not train on the set or does not publish its training
data (Winnow, Laya, CLM, Clef, Clef-flash). See the training table below.

First-call p50, ms:

| engine | English 30 | Korean 30 | BANKING77-20 | transfer-v4 | typed-decisions | NSMC | KLUE-YNAT |
|---|---|---|---|---|---|---|---|
| Kev-9B | 534.7 | 538.6 | 545.6 | 367.0 | 1552.2 | 296.6 | 413.9 |
| Kev-4B | 289.4 | 306.9 | 290.1 | 185.5 | 858.7 | 158.5 | 243.0 |
| Clef-flash | 769.5 | 1113.0 | 1274.5 | 628.5 | 2557.4 | 488.2 | 808.5 |
| Clef | 3480.6 | 3774.5 | 4525.3 | 2186.5 | 8923.0 | 1599.5 | 2791.8 |
| Winnow-E4B | 720.0 | 632.3 | 574.8 | 339.4 | 1545.8 | 291.4 | 426.4 |
| Kev-0.8B | 54.1 | 52.0 | 57.6 | 40.3 | 152.4 | 27.9 | 42.3 |
| Jeff-Qwen3.5-2B | 251.1 | 258.7 | 129.8 | 99.3 | 986.5 | 69.7 | 103.9 |
| Ollaya | 36.6 | 55.3 | 28.3 | 22.1 | 292.9 | 14.7 | 21.7 |
| AnyJev L0 | 937.3 | 1056.2 | 7860.1 | 571.7 | 2319.5 | 391.3 | 2554.0 |
| AnyJev raw | 436.5 | 576.5 | 582.3 | 292.2 | 1507.6 | 204.1 | 465.4 |
| Von | 126.4 | 158.1 | 38.1 | 31.8 | 411.4 | 50.9 | 60.4 |
| CLM-8B | 326.6 | 354.0 | 142.3 | - | - | - | - |

What each engine was trained on, as far as its authors publish it:

| engine | this repo's sets in its training data | source |
|---|---|---|
| Kev (all three sizes) | BANKING77 train split. Of transfer-v4's sources, Kev never trained on the six public ones. The two held-out policy structures come from the kind of policy data it trains on | model cards (`datasets`, transfer-v4 description), `kev/data.py` `TRAINABLE` |
| Jeff | PAWS `labeled_final` train split. transfer-v4's `paws` source (80 decisions) is that dataset's test split. None of the other sets is in its source list | `docs/data-sources.md` and `src/jeff/data.py` at `f0678829`, `kev/data.py` (`paws`: train, test) |
| AnyJev | none: training-free, Qwen3-8B as released | AnyJev README |
| Von | Banking77 and dair-ai/emotion (split not named), among about 290k decisions; "no JevBench item" | `wfzyx/von` model card, Training data |
| Clef-flash, Clef | unknown: post-trained from Qwen3.5-9B and Qwen3.8-27B, data not published | `Cloudflare/clef-flash` and `Cloudflare/clef` model cards |
| Winnow, Laya, CLM | unknown | |

Nobody trained on the 30-item sets, which were written for this repo.

Read these results with the following caveats:

- **Memory:** a vLLM Qwen3-8B server left over from a test (`--gpu-memory-utilization 0.7`, idle) ran through the whole
  matrix, so the latencies may be higher than on an idle machine. For example, Kev-4B's English first-call p50 is 289.4 ms
  here and was 225.9 ms on 2026-09-29.
- **Memory pressure:** it reached `critical` on 3 of Jeff's 604 KLUE-YNAT calls, with no errors or slow calls. All other runs stayed at `normal` or `warn`.
- **Ollaya errors:** Ollaya refused 7 typed-decisions items (35 decisions) with `STATE_TRUNCATED`, because the state
  did not fit `laya:en`'s context. They are counted as failed, not wrong.
- **AnyJev:** measured afterwards (18:59-22:02), once the leftover server that held its port 8092 was stopped, so its
  latencies come from an idle machine and do not compare directly with the rows above. `measure.sh` starts a fresh
  generate server for it. The 2026-09-29 AnyJev rows came from a server warmed by an earlier, discarded run over the
  same items, which is why its first-call p50 was lower then (raw English 178.5 ms, here 436.5 ms).
- **Clef-flash and Clef:** `ggml-org/Clef-Flash-GGUF` Q8_0 and `ggml-org/Clef-GGUF` Q4_K_M on llama.cpp b11403, one request at a time. Both read all the
  questions of a request in one prompt and decide them jointly; the bench sends each item's questions in one request,
  so their multi-question items are decided that way. Their latencies come from llama.cpp's Metal backend, the others' from
  MLX, Ollaya or vllm-metal, so they compare runtimes as much as models. Clef-flash's first measurement was cut off by a
  stopped session halfway through BANKING77; that set and the four after it were measured again from the start.
  Clef's runs stayed at `warn` memory pressure with 19.2 GB of weights loaded.
- **Von:** `von-sdk` 1.3.7 on Metal (`mps`) with `--noul-decision raw`. Its 0 order-flips come from scoring each
  option independently (`independent_options` in its calibration file), by design.
- **CLM:** measured on the 30-item and BANKING77 sets only. It picks a choice by comparing embeddings of the option
  texts, so its 0 order-flips may be structural rather than a sign of order robustness (not verified).

Which engine to use. Engines count as different when the exact McNemar p is below 0.05. With 45-55 pairs per set, some p values under 0.05 come by chance.

- **Most accurate:** Kev-9B, Kev-4B, Winnow, Clef-flash or Clef.
  - **Clef vs Clef-flash:** no set separates them (p ≥ 0.092; transfer-v4 public 474 vs 471, holdouts 158 vs 152 of
    176). Like Clef-flash, Clef beats Kev-9B, Kev-4B and Winnow on BANKING77-20, Winnow on transfer-v4 (on the
    holdouts) and Kev-4B on typed-decisions (all p < 0.001). Beyond that, it beats Kev-4B on NSMC (p = 0.043) and on
    transfer-v4's public sources (474 vs 454, p = 0.047), and Kev-9B on KLUE-YNAT (p = 0.047); each of those three is
    one of 45-55 pairs per set.
  - **Clef-flash vs the others:** level with Kev-9B on every set but BANKING77-20 (97% vs 89%, 24 vs 1 discordant,
    p < 0.001), where it is ahead of every engine but Clef (p = 0.375). It beats Kev-4B on typed-decisions (71% vs 67%, p < 0.001) and is
    level there with Kev-9B (p = 0.193) and Winnow (p = 0.051). On transfer-v4 it beats Winnow (p = 0.003) on the policy
    holdouts only (152 vs 125 of 176); on the six public sources all of them are level (Clef-flash 471 of 588, Kev-9B
    461, p = 0.268). On the Korean sets it cannot be told apart from Kev and Winnow (p ≥ 0.108).
  - **Kev-9B vs Kev-4B:** they differ only on typed-decisions (72% vs 67%, 265 vs 165 discordant, p < 0.001).
  - **Kev vs Winnow on transfer-v4:** Kev is ahead (Kev-4B p = 0.012, Kev-9B p = 0.003), but only on
    `composition_holdout` and `legacy_holdout`. Those are held-out structures of the policy data Kev trains on:
    Kev-4B 160/176, Kev-9B 157/176, Winnow 125/176 (p < 0.001). On the six public sources (`emotion`, `mmlu`, `paws`,
    `qnli`, `sciq`, `tweet_offensive`) they are level: 454, 461 and 460 of 588 (Kev-4B vs Winnow p = 0.58, Kev-9B vs
    Winnow p = 1.0). The counts are sums of the report's `## By source` table. The per-part p values are computed from the raw logs
    (snippet in `reports/2026-09-30-analysis.md`).
  - **Kev vs Winnow on BANKING77:** Kev is ahead, but that set is in Kev's training distribution.
  - **Winnow vs Kev-4B on typed-decisions:** Winnow is ahead (289 vs 176, p < 0.001).
  - **No difference between Kev and Winnow:** on the 30-item sets, NSMC and KLUE-YNAT.
- **typed-decisions:** Winnow (0.7265) and Kev-9B (0.720) land where the dataset card puts Jev 1.13.0 (0.727, ceiling
  0.735). Those are the card's numbers from its own harness, not re-run here. Clef is closest to the gold
  distributions (KL 0.196, Brier 0.107), then Kev-9B (0.209, 0.107) and Clef-flash (0.220, 0.113), against Winnow's 0.286
  and 0.128. The report's uniform row reproduces the
  card's KL 0.444 and Brier 0.238.
- **Under 60 ms:** Kev-0.8B (28-58 ms) or Ollaya (15-55 ms), outside typed-decisions' long states. Von (32-51 ms
  on transfer-v4, BANKING77-20 and NSMC) is English only.
  - **Accuracy:** Kev-0.8B is more accurate on typed-decisions, NSMC and KLUE-YNAT (p < 0.001) and on BANKING77
    (in-distribution). It does not differ from Ollaya on the 30-item sets or transfer-v4.
  - **Order-flip:** Ollaya changes 30-45% of its choices on BANKING77, typed-decisions and KLUE-YNAT when the options
    are reversed. Kev-0.8B changes 8-19%.
- **Korean:** not Von, which is English only.
  - **NSMC:** Kev-9B, Winnow and Kev-4B (83-86%) cannot be told apart, and Kev-9B beats Jeff (p = 0.008) and Kev-0.8B (p = 0.012).
  - **KLUE-YNAT:** Kev-4B, Jeff, Winnow and Kev-9B (73-74%) cannot be told apart (p ≥ 0.86). Kev-0.8B (63%) and
    Ollaya (43%) are below them (p < 0.001).
- **AnyJev:**
  - **KLUE-YNAT:** raw 76% and L0 75% are level with Kev-4B, Kev-9B, Winnow and Jeff (p ≥ 0.33).
  - **transfer-v4:** 75% for both, level with Winnow (p > 0.24) and below Kev-4B and Kev-9B (p < 0.001). The whole gap
    to Kev is on the two policy holdouts (Kev-4B 49 vs 2 discordant). On the six public sources AnyJev scores 458 of 588
    against Kev-4B's 454 (p = 0.76).
  - **typed-decisions:** raw 61% and L0 63% are below Kev-4B, Kev-9B and Winnow (p < 0.001). Their forecasts are
    far from gold: KL 3.783 (raw) and 2.813 (L0), ECE 0.35 and 0.31.
  - **NSMC:** 80-81%, below Kev-9B (p = 0.040 raw, 0.033 L0) and level with the rest of the top group.
  - **L0 vs raw:** L0 cuts order-flips on every set where raw flips any (Korean 30 2 → 0, BANKING77 68 → 22, transfer-v4 76 → 27,
    typed-decisions 101 → 31, KLUE-YNAT 46 → 26). It is more accurate on BANKING77 (p = 0.005) and typed-decisions
    (p < 0.001), and it costs 1.5-13.5x the first-call time.
- **Jeff-Qwen3.5-2B:** 52% on typed-decisions. The card lists another checkpoint, Jeff-Gemma4-E2B, at 0.561. Jeff is
  level with the 4B and 9B models on KLUE-YNAT, and flips 24% of BANKING77 choices.
- **Kev-4B reproduces its model card on transfer-v4:** 534/656 (81.4%, Wilson 78-84) on the clean questions against the
  card's 0.817 over the same 656. The set's other 108 decisions are its none_absent, none_present and permuted
  variants, 36 each (`## By variant` in the report).

Earlier findings on CLM (2026-09-28):

- The vllm-metal embeddings match transformers (MPS, bf16, last-token, L2) with cosine ≥ 0.9998 on four probe texts, so the encoder side is faithful.
- CLM reproduces the README's `department` (billing, 0.987 vs 0.939) and `frustration` (2.00 vs 1.98), but gives `urgency` 0.852 against the README's 0.41. Laya gives 0.795 on the same ticket. The cause is unresolved.
- On this question set CLM answers `frustration` at about 2.0 every time and leans toward `billing` for `department`.

### BANKING77 20-way

`bench/questions_banking77.jsonl` restates AnyJev's `banking20` task, so the numbers can be set beside its README.
It uses the 20 intents most frequent in BANKING77 train, listed by label id with no descriptions, and the question
"What is the customer's intent?". It keeps the first 300 test items after `random.Random(0)`, read from
`mteb/banking77` at a pinned revision. `uv run bench/make_banking77.py` regenerates it. The accuracy is in the table above.
Order-flip on this set (2026-09-30; Clef, Clef-flash and Von 2026-10-05):

| engine | order-flip |
|---|---|
| Kev-9B | 30/300 (10%) |
| Kev-4B | 23/300 (8%) |
| Kev-0.8B | 23/300 (8%) |
| Winnow-E4B | 38/300 (13%) |
| Jeff-Qwen3.5-2B | 73/300 (24%) |
| Ollaya `laya` | 93/300 (31%) |
| AnyJev L0 | 22/300 (7%) |
| AnyJev raw | 68/300 (23%) |
| Clef-flash | 0/300 (0%) |
| Clef | 0/300 (0%) |
| Von | 0/300 (0%) |
| CLM-8B | 0/300 (0%) |

- **Kev's BANKING77 numbers are in-distribution.** All three Kev cards list `legacy-datasets/banking77` as training data. Here
  Kev-0.8B cannot be told apart from the larger sizes (p ≥ 0.80), unlike on transfer-v4, typed-decisions and KLUE-YNAT
  (p < 0.001). Do not compare them with engines that did not train on it.
- **AnyJev reproduces its README here.** Its README reports, for Qwen3-8B on BANKING77 20-way with 300 test items,
  order-flip 0.230 raw → 0.073 L0 and accuracy 0.747 → 0.803. This run gives 68/300 (0.227) → 22/300 (0.073) and
  224/300 (0.747) → 241/300 (0.803), the same counts as the 2026-09-29 run. The Wilson intervals overlap (69-79 and
  75-84), but the paired test shows the accuracy gain: 25 decisions only L0 got right against 8 only raw got right,
  p = 0.005. L0 is level with Winnow here (p = 0.868).
- **Clef-flash (290/300) and Clef (287/300) lead here.** Their card reports BANKING77 as one of its own benchmarks
  (macro-F1 on all 77 intents: Clef 94.2, Clef-flash 90.9) and does not publish the training data, so whether this set
  is in their training distribution is unknown.
- **L0 is slow on many options.** It prefills the prompt once per cyclic shift, up to 20 times on a 20-way question:
  7860.1 ms first-call p50 against 582.3 ms raw.
- For scale, not comparison: Laya zero-shot is quoted at 38% on all 77 intents, against 76% for Jev
  (dhruvmehra/jevbench, as reported in a Laya fine-tuning write-up; not verified here). That set is 77-way, this
  one 20-way.

## FAQ

**What is a System One decision model?**
A model that answers typed questions about a piece of text (a choice among options, a yes/no probability, or a
score) in one forward pass, with calibrated probabilities instead of generated text. TypeSafe's Jev is the hosted
original; its `POST /v1/systemone` format is what every engine here speaks.

**Which local decision model is the most accurate on a Mac?**
Kev-9B, Kev-4B, Winnow-E4B, Clef-flash or Clef. Kev, Clef-flash and Clef lead on Kev's transfer-v4 set (80-83% against 77%), but only on its two policy-structure holdouts. Winnow, Kev-9B,
Clef-flash and Clef lead on typed-decisions (71-73%, against Kev-4B's 67%). On the Korean sets they cannot be told apart, but for Clef over Kev-4B on NSMC (p = 0.043) and over Kev-9B on KLUE-YNAT (p = 0.047). Clef-flash has the lowest ECE of them on 4 of the 7 sets (tied with Clef on one). Clef (27B) is no more accurate than Clef-flash on any set and takes 3.3-4.5x its time.

**Which one is the fastest?**
Ollaya running Laya: 14.7-55.3 ms first-call p50, or 292.9 ms on typed-decisions' long states. Kev-0.8B is next
(27.9-57.6 ms) and is as accurate or more on every set. Von is faster than Kev-0.8B on transfer-v4 (31.8 ms) and
BANKING77-20 (38.1 ms). It is level with Kev-0.8B on the English 30, transfer-v4 and typed-decisions (p ≥ 0.068)
and below it on BANKING77-20 and the three Korean sets (p < 0.001).

**Does AnyJev work on vllm-metal?**
Yes, with one change: vllm-metal 0.30.0 reports raw logprobs whatever `--logprobs-mode` says, so the adapter asks for
each label by `logprob_token_ids`. With that, AnyJev reproduces its README on BANKING77-20.

**Can these models answer in Korean?**
Kev-4B, Kev-9B, Winnow and Jeff can: 83-86% on NSMC for the first three, and 73-74% on KLUE-YNAT for all four.
AnyJev can too, with no training: 80-81% on NSMC and 75-76% on KLUE-YNAT.
Clef-flash and Clef can: 86-87% on NSMC and 77-78% on KLUE-YNAT, level with Kev and Winnow (Clef is ahead of Kev-4B on NSMC, p = 0.043, and of Kev-9B on KLUE-YNAT, p = 0.047).
Ollaya needs `laya:multilingual` and reaches 56% and 43%. Von cannot: its card says English only, and it scores 51%
on NSMC and 15% on KLUE-YNAT.

**How much memory do they need?**
Kev-4B's server used 2.7 GB RSS after startup. Winnow is an 8.0 GB download, Clef-flash Q8_0 a 9.7 GB GGUF, Clef Q4_K_M a 19.2 GB GGUF and Von
3.2 GB of weights. AnyJev and CLM run Qwen3-8B on vllm-metal, which reserves about
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
