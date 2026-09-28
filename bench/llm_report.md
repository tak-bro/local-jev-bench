# vllm-metal vs Ollama, Qwen3-8B chat

8 prompts, max_tokens=256, temperature 0, `/no_think`. Single = sequential streaming; concurrent = parallel non-streaming.

| server | single TTFT p50 ms | single decode tok/s p50 | concurrent tok/s | concurrent wall s | concurrent tokens | free mem after single | pressure after concurrent | token count source |
|---|---|---|---|---|---|---|---|---|
| vllm-bf16 | 197 | 16.8 | 77.3 | 14.8 | 1143 | 24% | normal | usage |
| vllm-mlx-4bit | 291 | 30.2 | 48.6 | 24.8 | 1206 | 21% | normal | usage |
| ollama-q4 | 233 | 29.4 | 30.2 | 36.1 | 1091 | 59% | normal | usage |
