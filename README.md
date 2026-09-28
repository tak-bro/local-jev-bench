# local-sys1

This repo runs Jev-style System One decision models (typed questions in, calibrated answers out) locally on an Apple Silicon Mac. It sets up two engines and benchmarks them on the same questions:

- **CLM** ([Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM)): a Qwen3-8B encoder served by [vllm-metal](https://github.com/vllm-project/vllm-metal), plus CLM's 75 MB head.
- **Ollaya** ([ollaya-dev/ollaya](https://github.com/ollaya-dev/ollaya)): the `laya` model, run on MLX.

Both speak TypeSafe's `POST /v1/systemone` wire format.

## Install

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal   # the tap needs the URL
brew install vllm-project/vllm-metal/vllm-metal
OLLAYA_INSTALL_DIR=$HOME/.local OLLAYA_NO_SERVICE=1 sh -c "$(curl -fsSL https://ollaya.dev/install.sh)"
uv sync
```

`contrastive-lm` declares `vllm` as a dependency, but it only calls the embeddings endpoint over HTTP. `pyproject.toml` overrides that dependency away so a second vLLM is not installed.

## Ports

Every server binds to `127.0.0.1` only.

| Port | Process | Start |
|---|---|---|
| 8091 | small embedding model (smoke only) | `scripts/serve-embed.sh mlx-community/Qwen3-Embedding-0.6B-8bit 8091 embed-small` |
| 8090 | Qwen3-8B encoder for CLM | `scripts/serve-embed.sh Qwen/Qwen3-8B 8090 qwen3-8b` |
| 8700 | CLM System One API | `scripts/serve-clm.sh` |
| 11435 | Ollaya daemon | `OLLAYA_HOST=127.0.0.1:11435 ~/.local/bin/ollaya serve` |

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
uv run python bench/run.py --out bench/report.md     # 30 questions x 3 reps on both engines, with kernel memory pressure
uv run pytest -q                              # offline tests, fake servers, no model loaded
```

The benchmark rejects any item whose state plus question instructions (the text CLM actually embeds) exceeds 2048 Qwen3 tokens, because `clm-serve` would otherwise truncate it without an error. Treat the accuracy over 30 hand-written items as a sanity check, not a benchmark.

## Results (2026-09-28, M3 Max 36 GB)

| engine | cold ms | first-call p50 ms | accuracy | worst memory pressure |
|---|---|---|---|---|
| CLM-8B (vllm-metal, bf16) | 4303 | 184 | 126/315 (40%) | warn |
| Ollaya `laya:en` (MLX, F32) | 499 | 40 | 237/315 (75%) | warn |

- The vllm-metal embeddings match transformers (MPS, bf16, last-token, L2) with cosine ≥ 0.9998 on four probe texts, so the encoder side is faithful.
- CLM reproduces the README's `department` (billing, 0.987 vs 0.939) and `frustration` (2.00 vs 1.98), but gives `urgency` 0.852 against the README's 0.41. Laya gives 0.795 on the same ticket. The cause is unresolved.
- On this question set CLM answers `frustration` at about 2.0 every time and leans toward `billing` for `department`. See `bench/report.md` for the full table.

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
