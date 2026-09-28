# vllm-metal vs Ollama, Qwen3-8B chat

8 prompts, max_tokens=256, temperature 0, `/no_think`. Single = sequential streaming; concurrent = parallel non-streaming.

| server | single TTFT p50 ms | single decode tok/s p50 | concurrent tok/s | concurrent wall s | concurrent tokens | free mem after single | pressure after concurrent | token count source |
|---|---|---|---|---|---|---|---|---|
| vllm-bf16 | 197 | 16.8 | 77.3 | 14.8 | 1143 | 24% | normal | usage |
| vllm-mlx-4bit | 291 | 30.2 | 48.6 | 24.8 | 1206 | 21% | normal | usage |
| ollama-q4 | 233 | 29.4 | 30.2 | 36.1 | 1091 | 59% | normal | usage |
| ollama-q4-par1-ctx4k | 134 | 45.3 | 40.7 | 26.8 | 1091 | 72% | normal | usage |
| ollama-q4-par4-ctx4k | 163 | 37.3 | 17.5 | 63.6 | 1111 | 67% | normal | usage |
| ollama-q4-par8-ctx4k | 596 | 7.8 | 8.9 | 124.4 | 1111 | 46% | normal | usage |
| ollama-q4-par1-ctx4k-fa | 342 | 9.1 | 9.5 | 115.3 | 1091 | 54% | normal | usage | (discarded: ko-qmd's paraphrase job was using the Ollama app during this run)
| ollama-q4-par4-ctx4k-fa | 267 | 23.6 | 20.4 | 54.3 | 1111 | 65% | normal | usage | (discarded: ko-qmd's paraphrase job was using the Ollama app during this run)
| ollama-q4-par8-ctx4k-fa | 349 | 15.9 | 19.4 | 57.4 | 1111 | 28% | normal | usage | (discarded: ko-qmd's paraphrase job was using the Ollama app during this run)
