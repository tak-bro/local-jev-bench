# local-jev-bench: Apple Silicon 맥에서 돌리는 Jev 스타일 결정 모델 (Kev, Winnow, Clef, Von, Jeff, AnyJev, Laya, CLM)

[English](README.md) | 한국어

local-jev-bench는 Jev 스타일의 System One 결정 모델을 Apple Silicon 맥에서 로컬로 돌린다. 결정 모델은 타입이 정해진 질문을 받아 보정된 확률로 답하는 모델이다. 여기서는 여러 엔진을 같은 질문으로 비교한다. 비교에 쓰는 셋은 다음과 같다.

- 영어·한국어 고객 문의
- 공개 셋 BANKING77 20-way 인텐트 분류
- Kev의 분포 밖 셋 transfer-v4
- typed-decisions 리더보드 셋
- 한국어 셋 두 개: NSMC 영화 리뷰, KLUE-YNAT 뉴스 제목

**핵심 결과 (M3 Max 36GB, 2026-09-30; Clef·Clef-flash·Von은 2026-10-05).** 전체적으로 가장 정확한 엔진은 Kev-9B, Kev-4B, Winnow-E4B, Clef, Clef-flash다. 이들 사이의 차이는 대응 McNemar 검정으로 봤을 때 일부 셋에서만 난다.

- **transfer-v4**: Kev가 Winnow보다 낫다(Kev-9B 81% vs 77%, p = 0.003). 하지만 차이는 정책 구조 홀드아웃 출처 두 개에서만 난다. 이 두 출처는 Kev가 학습하는 프로그램 생성 정책 데이터에서 구조만 뺀 것이다(Kev-4B 160/176, Winnow 125/176). Kev가 학습하지 않은 공개 출처 여섯 개에서는 셋이 같고(588개 중 454, 461, 460, p > 0.45), 학습 없는 AnyJev도 같다(raw·L0 모두 458).
- **typed-decisions**: Winnow가 Kev-4B보다 낫고(73% vs 67%, p < 0.001), Kev-9B와는 같다(72%, p = 0.529).
- **NSMC와 KLUE-YNAT(한국어)**: 셋과 Clef-flash·Clef를 구분할 수 없다. 예외는 NSMC에서 Clef가 Kev-4B보다(p = 0.043), KLUE-YNAT에서 Kev-9B보다(p = 0.047) 나은 두 경우다.
- **AnyJev(학습 없는 Qwen3-8B)**: KLUE-YNAT에서는 상위 그룹과 같고(75-76%), transfer-v4에서는 Winnow와 같다(75%). typed-decisions에서는 뒤진다(61-63%). 거기서는 gold 분포와의 거리도 멀다(KL 2.8-3.8, 상위 그룹은 0.2-0.3). BANKING77-20에서는 자기 README를 다시 재현했고, 대응 검정으로 보면 L0의 정확도 향상이 실제로 있다(p = 0.005).
- **Clef-flash(Cloudflare, 9B, 2026-10-05 측정)**: 상위 그룹에 든다. BANKING77-20을 빼면 모든 셋에서 Kev-9B와 같다. BANKING77-20에서는 앞선다(97% vs 89%, p < 0.001; Clef-flash는 학습 데이터를 공개하지 않는다). transfer-v4에서는 Winnow보다 낫지만(82% vs 77%, p = 0.003), 이것도 정책 홀드아웃 두 출처에서만이다(176개 중 152 vs 125). 공개 출처 여섯 개에서는 Kev, Winnow, AnyJev와 같다(588개 중 471, p > 0.08). 7개 셋 중 4개에서 전체 엔진 가운데 ECE가 가장 낮고(BANKING77-20은 Clef와 같다)(typed-decisions 0.021, KL 0.220), 선택지 순서를 뒤집어도 답을 하나도 바꾸지 않는다. 대신 여기서는 느리다. llama.cpp에서 첫 호출 p50이 488-2557ms로 Kev-9B의 1.4-2.3배다(Kev-9B 행은 더 바쁜 머신에서 쟀다). Cloudflare가 자사 GPU에서 보고한 중앙값은 38.8ms다.
- **Clef(Cloudflare, 27B, 2026-10-05 측정)**: 어느 셋에서도 Clef-flash와 구분되지 않는다(p ≥ 0.092). 시간은 3.3-4.5배 든다(첫 호출 p50 1600-8923ms, Q4_K_M. Cloudflare가 자사 GPU에서 보고한 중앙값은 209.3ms). Kev-9B와는 BANKING77-20(96% vs 89%, p < 0.001)과 KLUE-YNAT(78% vs 73%, p = 0.047, 이 셋의 55쌍 중 하나)을 빼면 같다. typed-decisions gold 분포에는 전체 엔진 중 가장 가깝다(KL 0.196).
- **Von(395M ModernBERT, 2026-10-05 측정)**: 모델 카드대로 영어 전용이다. KLUE-YNAT 15%, 한국어 30문항 49%, NSMC 51%(2지선다)다. 영어 셋에서도 상위 그룹보다 낮다(transfer-v4 69%, typed-decisions 43%). 첫 호출 p50은 32-411ms다.

가장 빠른 엔진은 Ollaya(Laya)다. typed-decisions의 긴 state를 빼면 첫 호출 p50이 15-55ms다. 대신 CLM과 Von을 빼면 정확도가 가장 낮다. 예외는 영어 30문항으로, 여기서는 Jeff가 더 낮다(영어 전용인 Von은 한국어 셋 세 개에서 그보다도 낮다). 같은 셋들에서 보정(ECE)은 Kev·Winnow·Jeff·Ollaya 중 가장 나쁘다. AnyJev와 CLM은 일부 셋에서 그보다도 나쁘다. Kev-0.8B는 같은 셋에서 28-58ms이고, 모든 셋에서 정확도가 Ollaya와 같거나 더 높다. Kev-4B는 transfer-v4에서 모델 카드 수치를 재현했다(534/656, 카드 0.817). 자세한 내용은 [결과](#결과-2026-09-30-m3-max-36gb) 절에 있다.

엔진은 아홉 가지다.

- **CLM** ([Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM)): [vllm-metal](https://github.com/vllm-project/vllm-metal)로 서빙하는 Qwen3-8B 인코더와 75MB짜리 CLM 헤드.
- **Ollaya** ([ollaya-dev/ollaya](https://github.com/ollaya-dev/ollaya)): `laya` 모델을 MLX로 돌린다.
- **Winnow** ([ollaya.dev/library/winnow](https://ollaya.dev/library/winnow)): Gemma 4 E4B 기반의 `winnow:e4b`. Ollaya가 llama.cpp(Q8_0)로 돌린다.
- **AnyJev** ([nokia-applied-research/AnyJev](https://github.com/nokia-applied-research/AnyJev)): 학습이 필요 없다. vllm-metal 위 Qwen3-8B에서 라벨 logprob을 읽는다. `anyjev-raw`는 보정 없이, `anyjev-l0`는 선택지를 순환 이동하고 배치 prior로 보정한다.
- **Kev** ([jaredpalmer/kev](https://github.com/jaredpalmer/kev)): Qwen3.5-0.8B·4B·9B Base 위의 LoRA와 포인터 헤드(`kev-0.8b`, `kev-4b`, `kev-9b`). MLX로 돌린다.
- **Jeff** ([firelex/jeff](https://github.com/firelex/jeff)): Jeff-Qwen3.5-2B. Qwen3.5-2B를 파인튜닝하고 답 readout을 학습한 모델이다. MLX로 돌린다.
- **Clef-flash** ([Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)): Qwen3.5-9B 기반 Cloudflare 9B 결정 모델. `ggml-org/Clef-Flash-GGUF` Q8_0을 llama.cpp의 `llama-server`로 돌린다(`clef-flash`). 한 요청의 질문을 한 프롬프트로 함께 판단한다. 다른 엔진은 질문마다 따로 답한다.
- **Clef** ([Cloudflare/clef](https://huggingface.co/Cloudflare/clef)): Qwen3.8-27B 기반의 27B 형제 모델. `ggml-org/Clef-GGUF` Q4_K_M을 같은 `llama-server`로 돌린다(`clef`). Q8_0(28.7GB)은 36GB에서 여유가 너무 적다.
- **Von** ([wfzyx/von](https://github.com/wfzyx/von)): 395M ModernBERT 인코더와 옵션 마커 헤드. 자체 `von serve`로 Metal에서 돌린다(`von`).

모든 엔진은 TypeSafe의 `POST /v1/systemone` 형식으로 응답한다.

## 설치

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal   # the tap needs the URL
brew install vllm-project/vllm-metal/vllm-metal
OLLAYA_INSTALL_DIR=$HOME/.local OLLAYA_NO_SERVICE=1 sh -c "$(curl -fsSL https://ollaya.dev/install.sh)"
uv sync
```

[AnyJev](https://github.com/nokia-applied-research/AnyJev)는 `uv sync`로 설치된다.

- AnyJev의 vLLM 백엔드는 라벨 logprob을 HTTP로 읽는다. `transformers`는 토크나이저에만 쓰므로 torch가 필요 없다.
- `serve/anyjev_server.py`가 AnyJev를 `/v1/systemone` 형식으로 감싼다. 요청의 `model`로 보정 레벨(`anyjev-raw` 또는 `anyjev-l0`)을 고른다.
- 어댑터는 AnyJev 기본 방식(`allowed_token_ids` + top-K) 대신, 라벨마다 id로 logprob을 요청한다(`logprob_token_ids`). vllm-metal 0.30.0은 그 필터를 적용하기 전의 logprob을 주기 때문에, 라벨이 top-K에서 빠질 수 있다.
- 서버 응답에 라벨 토큰이 빠져 있으면 확률을 임의로 채우지 않고 HTTP 502를 돌려준다.

[Kev](https://github.com/jaredpalmer/kev)는 자기 체크아웃 안에 별도 uv 환경(torch, mlx-lm)을 둔다. 그래서 이 레포의 의존성이 아니다. `git clone https://github.com/jaredpalmer/kev ~/workspace/tak-bro/kev && (cd ~/workspace/tak-bro/kev && uv sync --extra serve)`로 준비한다. `scripts/serve-kev.sh`는 Qwen3.5-4B-Base 위에 `jaredpalmer/kev-4b` 어댑터를 얹은 `/v1/systemone` 서버를 띄운다. `KEV_RUN=jaredpalmer/kev-0.8b`나 `jaredpalmer/kev-9b`를 주면 같은 포트에서 다른 크기를 띄운다. 엔진 `kev-0.8b`·`kev-4b`·`kev-9b`는 측정 전에 `/v1/models`를 확인해 다른 크기가 떠 있으면 거부한다.

[Winnow](https://ollaya.dev/library/winnow)는 Ollaya에서 돈다. `ollaya pull winnow:e4b`(8.0GB, Gemma 4, Q8_0)로 받고 `winnow` 엔진으로 잰다. `ollaya stop winnow:e4b`로 내린다.

[Jeff](https://github.com/firelex/jeff)도 자기 uv 환경을 둔다. `pyproject.toml`이 uv 0.12.19 이상을 요구해서 `uvx`를 쓴다. `git clone https://github.com/firelex/jeff ~/workspace/tak-bro/jeff && git -C ~/workspace/tak-bro/jeff checkout f06788292874c21a5b5c41549ac220dd9e15da7f` 뒤 그 안에서 `uvx --from 'uv>=0.12.19' uv sync --no-default-groups --extra mac`, `uvx --from 'uv>=0.12.19' uv run --no-default-groups hf download mstrasser/Jeff-Qwen3.5-2B --local-dir checkpoints/jeff-2b`를 실행한다. `scripts/serve-jeff.sh`는 MLX로 띄운다. MLX는 Jeff의 Qwen 모델만 돌리므로 Jeff-Gemma4-E2B는 쓰지 않는다.

[Clef-flash](https://huggingface.co/ggml-org/Clef-Flash-GGUF)는 llama.cpp 빌드 11403이 필요하다. `/v1/systemone`과 clef 아키텍처가 Homebrew 0.5.0 뒤에 들어왔다. [b11403 릴리스](https://github.com/ggml-org/llama.cpp/releases/tag/b11403)의 `llama-b11403-bin-macos-arm64.tar.gz`를 `~/.local/opt/llama.cpp/b11403/`에 푼다(또는 `LLAMA_SERVER`로 그 `llama-server`를 가리킨다). `scripts/serve-llama.sh clef-flash`(또는 `clef`)가 빌드를 확인하고, GGUF를 고정된 리비전으로 받아(9.7GB, Clef는 19.2GB) 프롬프트 전체를 한 배치(`-ub 8192`)로 서빙한다. clef는 이렇게 해야 돈다.

[Von](https://github.com/wfzyx/von)은 PyPI의 `von-sdk==1.3.7`을 `uvx`로 돌리고, 가중치는 `wfzyx/von`의 한 리비전으로 고정한다(3.2GB). `scripts/serve-von.sh`는 Metal에서 `--noul-decision raw`로 띄운다. 기본값은 noul 확률을 모두 0.2-0.8 밖으로 옮기는데, 판단은 그대로지만 여기서 채점하는 보정이 사라진다.

`contrastive-lm`은 `vllm`을 의존성으로 선언하지만, 실제로는 임베딩 엔드포인트를 HTTP로 호출할 뿐이다. 그래서 `pyproject.toml`에서 이 의존성을 빼서 vLLM이 두 번 설치되지 않게 했다.

## 포트

모든 서버는 `127.0.0.1`에만 바인드한다.

| 포트 | 프로세스 | 시작 |
|---|---|---|
| 8091 | 작은 임베딩 모델(smoke 전용) | `scripts/serve-embed.sh mlx-community/Qwen3-Embedding-0.6B-8bit 8091 embed-small` |
| 8090 | CLM용 Qwen3-8B 인코더 | `scripts/serve-embed.sh Qwen/Qwen3-8B 8090 qwen3-8b` |
| 8700 | CLM System One API | `scripts/serve-clm.sh` |
| 11435 | Ollaya 데몬 (`laya`, `winnow:e4b`) | `OLLAYA_HOST=127.0.0.1:11435 ~/.local/bin/ollaya serve` |
| 8092 | AnyJev용 Qwen3-8B 생성 서버 | `scripts/serve-llm.sh Qwen/Qwen3-8B 8092 qwen3-8b` |
| 8710 | AnyJev System One API (`anyjev-raw`, `anyjev-l0`) | `scripts/serve-anyjev.sh` |
| 8009 | Kev System One API (`kev-latest`) | `KEV_RUN=jaredpalmer/kev-4b scripts/serve-kev.sh` (또는 `kev-0.8b`, `kev-9b`) |
| 8765 | Jeff System One API (`jeff-latest`) | `scripts/serve-jeff.sh` |
| 8020 | llama-server System One API (`Clef-Flash-Q8_0@4a7a08c` 또는 `Clef-Q4_K_M@5f70656`) | `scripts/serve-llama.sh clef-flash` (또는 `clef`) |
| 8030 | Von System One API (`von-latest`) | `scripts/serve-von.sh` |

포트가 이미 쓰이고 있으면 serve 스크립트는 시작하지 않는다.

`serve-embed.sh`는 이 맥(M3 Max 36GB, vllm-metal 0.30.0)에서 필요했던 설정 세 가지를 고정한다.

- `GLOO_SOCKET_IFNAME=lo0 VLLM_HOST_IP=127.0.0.1`: 이 설정이 없으면 torch.distributed가 `.local` 호스트명을 풀지 못해 엔진 초기화에서 멈춘다.
- `--pooler-config '{"seq_pooling_type": "LAST", "use_activation": true}'`: 마지막 토큰 풀링과 L2 정규화. CLM 헤드가 이 형식을 기대한다. 설정이 없으면 vLLM은 아키텍처 기본값을 쓴다.
- `--gpu-memory-utilization 0.7`: vllm-metal은 이 비율을 Metal wired 한도(여기서는 28.1GB)에 적용한다.
  - 기본값 0.92는 메모리 압력을 warn까지 올렸다.
  - 0.55는 bf16 가중치 16GB를 올리고 나면 KV 캐시 자리가 남지 않았다.

## 확인과 벤치마크

```bash
scripts/smoke-embed.sh 8091                   # vector returned, L2-normalised
scripts/smoke-embed.sh 8090 4096              # Qwen3-8B: 4096 dims, normalised
uv run python bench/run.py --smoke --engine clm      # CLM README example within tolerance
uv run python bench/run.py --smoke --engine ollaya   # same example, shape only
uv run python bench/run.py                           # 30 questions x 3 reps on CLM and Ollaya, with kernel memory pressure
uv run python bench/run.py --smoke --engine <e>       # shape only (anyjev-raw, anyjev-l0, kev-0.8b, kev-4b, kev-9b, winnow, jeff), its server up
uv run python bench/run.py --engine <e> --reps 1 --questions bench/questions_banking77.jsonl
uv run python bench/score.py bench/runs/questions --check   # report.md matches the raw logs
uv run python bench/make_sets.py transfer-v4        # Kev의 분포 밖 development 세트(764)를 bench/data/에, gitignore
uv run python bench/make_sets.py typed-decisions    # LocalLLaMA/typed-decisions test(400케이스, 2,000판단), gold 분포 포함
uv run python bench/make_sets.py nsmc               # 한국어 영화 리뷰 300개, 긍정 여부 (HF 카드: CC BY 2.0)
uv run python bench/make_sets.py klue-ynat          # 한국어 뉴스 제목 300개, 7개 분야 (KLUE, CC BY-SA 4.0)
uv run pytest -q                              # offline tests, fake servers, no model loaded
```

AnyJev와 Kev는 각각 Metal 메모리 대부분을 쓴다. 그래서 엔진은 하나씩 따로 측정한다.

1. 그 엔진의 서버를 띄운다.
2. `bench/run.py --engine <e> --questions <셋>`을 실행한다.
3. 서버를 내린다.

`scripts/measure.sh [--reps N] <엔진> <셋>...`이 세 단계를 한 번에 하고, 중간에 실패해도 서버를 내린다. 서버 출력은 `logs/<이름>.log`에 남는다.

실행은 모든 호출(확률을 포함한 답·지연·메모리 압력)을 `bench/runs/<셋>/<엔진>.jsonl`에 남기고 실행이 끝나면 그 엔진의 이전 기록을 덮어쓴다(중단된 실행은 이전 기록을 남긴다). `bench/score.py`는 그 디렉터리의 기록 전부로 `bench/runs/<셋>/report.md`를 다시 만든다. 다른 버전의 셋이나 다른 `--reps`로 잰 기록은 거부한다. AnyJev 어댑터는 측정할 때마다 재시작한다. L0의 배치 prior가 어댑터가 처리한 모든 호출에 걸쳐 누적되기 때문이다.

벤치마크는 상태와 질문 지시문을 합친 텍스트가 Qwen3 토큰 2048개를 넘는 문항을 거부한다. 이 텍스트가 CLM이 실제로 임베딩하는 입력인데, `clm-serve`는 긴 입력을 오류 없이 잘라 버리기 때문이다. 손으로 만든 30문항 정확도는 동작 확인용이지 벤치마크가 아니다.

## 결과 (2026-09-30, M3 Max 36GB)

`scripts/measure.sh`로 한 번에 엔진 하나만 쟀다.

- **반복 횟수**: 30문항 셋은 3번 반복했다. 나머지 셋은 1번 돌렸고, choice 질문이 있는 셋(NSMC 빼고 전부)은 선택지를 뒤집은 호출을 한 번 더 했다.
- **채점**: 정확도는 판단(문항 x 질문)마다 각 문항의 첫 timed 호출로 매긴다.
- **전체 표**: `bench/runs/<셋>/report.md`에 있다. 문항 수와 Wilson 95% 구간, Brier, ECE, order-flip, McNemar 대응 표, 출처별 분해가 들어 있다.
  `bench/score.py <dir> --check`가 옆의 원자료로 이 표를 다시 만들어 같은지 확인한다.
- **Ollaya 모델**: 영어 셋에는 `laya`, 한국어 셋에는 `laya:multilingual`을 썼다.
- **측정 시점**: Clef·Clef-flash·Von은 2026-10-05에 같은 셋 파일(같은 sha256)로, 유휴 머신에서 쟀다. 나머지 행은 2026-09-30 측정이다. 해석은 `reports/2026-10-05-analysis.md`에 있다.

정확도, 첫 timed 호출 기준(`†` = 그 엔진이 이 셋에 든 데이터의 train split으로 학습했다):

| 엔진 | 영어 30 | 한국어 30 | BANKING77-20 | transfer-v4 | typed-decisions | NSMC | KLUE-YNAT |
|---|---|---|---|---|---|---|---|
| 판단 수 | 105 | 105 | 300 | 764 | 2,000 | 300 | 300 |
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

`†`: Kev(세 크기 모두)는 BANKING77 train split으로, Jeff는 PAWS train split으로 학습했다. PAWS의 test split은 transfer-v4 764개 판단 중 80개다. Von의 카드는 학습 코퍼스에 Banking77과 dair-ai/emotion이 들었다고 적지만 split은 밝히지 않는다. transfer-v4의 `emotion` 출처(116개 판단)가 dair-ai/emotion이다. 표시가 없는 엔진은 그 셋으로 학습하지 않았거나, 학습 데이터를 공개하지 않은 것이다(Winnow, Laya, CLM, Clef, Clef-flash). 아래 학습 데이터 표를 참고한다.

첫 호출 p50, ms:

| 엔진 | 영어 30 | 한국어 30 | BANKING77-20 | transfer-v4 | typed-decisions | NSMC | KLUE-YNAT |
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

각 엔진이 무엇으로 학습했는지(만든 쪽이 공개한 범위):

| 엔진 | 학습 데이터에 든 이 레포의 셋 | 출처 |
|---|---|---|
| Kev (세 크기 모두) | BANKING77 train split. transfer-v4 출처 가운데 공개 출처 여섯 개는 Kev가 학습하지 않았다. 정책 구조 홀드아웃 두 개는 Kev가 학습하는 정책 데이터에서 나왔다 | 모델 카드(`datasets`, transfer-v4 설명), `kev/data.py` `TRAINABLE` |
| Jeff | PAWS `labeled_final` train split. transfer-v4의 `paws` 출처(80판단)는 같은 데이터의 test split이다. 다른 셋은 출처 목록에 없다 | `f0678829` 시점의 `docs/data-sources.md`, `src/jeff/data.py`, `kev/data.py`(`paws`: train, test) |
| AnyJev | 없다. 학습이 없고, 공개된 Qwen3-8B를 그대로 쓴다 | AnyJev README |
| Von | 약 29만 개 판단 가운데 Banking77과 dair-ai/emotion(split 미기재). "JevBench 항목은 없음" | `wfzyx/von` 모델 카드 Training data |
| Clef-flash, Clef | 알 수 없다. Qwen3.5-9B와 Qwen3.8-27B에서 사후 학습했고, 데이터는 공개하지 않았다 | `Cloudflare/clef-flash`, `Cloudflare/clef` 모델 카드 |
| Winnow, Laya, CLM | 알 수 없음 | |

30문항 셋은 이 레포를 위해 새로 쓴 것이라 어떤 엔진도 학습하지 않았다.

주의할 점:

- **메모리**: 테스트에서 남은 vLLM Qwen3-8B 서버(`--gpu-memory-utilization 0.7`, 유휴)가 측정 내내 떠 있었다. 그래서 지연은 한가한 기계보다 높게 나왔을 수 있다. 예를 들어 Kev-4B의 영어 첫 호출 p50은 이번 289.4ms, 2026-09-29에는 225.9ms였다.
- **메모리 압력**: Jeff의 KLUE-YNAT 호출 604번 중 3번이 `critical`이었다. 에러도, 느려진 호출도 없었다. 나머지 측정은 모두 `normal`이나 `warn`이었다.
- **Ollaya 에러**: typed-decisions 7문항(판단 35개)을 `STATE_TRUNCATED`로 거부했다. state가 `laya:en` 컨텍스트에 들어가지 않았기 때문이다. 이 판단들은 오답이 아니라 실패로 센다.
- **AnyJev**: 포트 8092를 잡고 있던 남은 서버를 내린 뒤 따로 쟀다(18:59-22:02). 그래서 AnyJev의 지연은 한가한 기계에서 잰 값이라 위의 다른 행과 바로 비교할 수 없다. `measure.sh`는 AnyJev용 생성 서버를 새로 띄운다. 2026-09-29 AnyJev 수치는 같은 문항을 한 번 돌려(그 결과는 버림) 데워진 서버에서 쟀기 때문에 첫 호출 p50이 더 낮았다(raw 영어 178.5ms, 이번 436.5ms).
- **Clef-flash와 Clef**: `ggml-org/Clef-Flash-GGUF` Q8_0과 `ggml-org/Clef-GGUF` Q4_K_M을 llama.cpp b11403에서 한 번에 한 요청씩 돌렸다. 둘 다 한 요청의 질문을 한 프롬프트로 읽어 함께 판단한다. 벤치는 항목의 질문을 한 요청에 담아 보내므로, 질문이 여러 개인 항목은 그렇게 판단됐다. 지연은 llama.cpp Metal 백엔드의 값이고, 다른 엔진은 MLX·Ollaya·vllm-metal의 값이다. 그래서 모델만큼이나 런타임을 비교한다. Clef-flash의 첫 측정은 BANKING77 도중 세션이 끝나며 끊겼고, 그 셋부터 뒤의 네 셋까지 처음부터 다시 쟀다. Clef는 가중치 19.2GB를 올린 채 메모리 압력 `warn`에 머물렀다.
- **Von**: `von-sdk` 1.3.7을 Metal(`mps`)에서 `--noul-decision raw`로 돌렸다. order-flip 0은 선택지를 하나씩 따로 채점하는 설계(보정 파일의 `independent_options`) 때문이다.
- **CLM**: 30문항 셋과 BANKING77만 쟀다. CLM은 선택지 텍스트 임베딩끼리의 유사도로 답을 고른다. 그래서 order-flip 0은 순서에 강해서라기보다 구조 때문일 수 있다(미검증).

어떤 엔진을 쓸까. 정확 McNemar p가 0.05 미만이면 차이가 있는 것으로 본다. 셋마다 비교 쌍이 45-55개라서, 0.05 미만 가운데 일부는 우연히 나온다.

- **가장 정확한 쪽**: Kev-9B, Kev-4B, Winnow, Clef-flash, Clef.
  - Clef와 Clef-flash는 어느 셋에서도 갈리지 않는다(p ≥ 0.092; transfer-v4 공개 474 vs 471, 홀드아웃 176개 중 158 vs 152). Clef도 Clef-flash처럼 BANKING77-20에서 Kev-9B·Kev-4B·Winnow를, transfer-v4에서 Winnow를(홀드아웃에서), typed-decisions에서 Kev-4B를 앞선다(모두 p < 0.001). 그 밖에 NSMC(p = 0.043)와 transfer-v4 공개 출처(474 vs 454, p = 0.047)에서 Kev-4B보다, KLUE-YNAT에서 Kev-9B보다(p = 0.047) 낫다. 이 셋은 각각 셋마다 45-55쌍 가운데 하나다.
  - Clef-flash는 BANKING77-20을 빼면 모든 셋에서 Kev-9B와 같다. BANKING77-20에서는 Clef를 뺀 모든 엔진보다 앞선다(97% vs 89%, 불일치 24 vs 1, p < 0.001; Clef와는 p = 0.375). typed-decisions에서는 Kev-4B보다 낫고(71% vs 67%, p < 0.001), Kev-9B(p = 0.193)·Winnow(p = 0.051)와 같다. transfer-v4에서는 Winnow보다 낫지만(p = 0.003) 정책 홀드아웃에서만이다(176개 중 152 vs 125). 공개 출처 여섯 개에서는 모두 같다(Clef-flash 588개 중 471, Kev-9B 461, p = 0.268). 한국어 셋에서는 Kev, Winnow와 구분할 수 없다(p ≥ 0.108).
  - Kev-9B와 Kev-4B는 typed-decisions에서만 다르다(72% vs 67%, 불일치 265 vs 165, p < 0.001).
  - transfer-v4에서는 Kev가 Winnow보다 낫다(Kev-4B p = 0.012, Kev-9B p = 0.003). 하지만 차이는 `composition_holdout`과 `legacy_holdout`에서만 난다.
    - 이 둘은 Kev가 학습하는 정책 데이터에서 구조를 빼 둔 출처다. Kev-4B 160/176, Kev-9B 157/176, Winnow 125/176이다(p < 0.001).
    - 공개 출처 여섯 개(`emotion`, `mmlu`, `paws`, `qnli`, `sciq`, `tweet_offensive`)에서는 같다. 588개 중 454, 461, 460이다(Kev-4B vs Winnow p = 0.58, Kev-9B vs Winnow p = 1.0).
    - 문항 수는 report의 `## By source` 표를 더한 값이다. 부분별 p 값은 원자료로 계산했다(코드는 `reports/2026-09-30-analysis.md`).
  - BANKING77에서도 Kev가 앞서지만, 이 셋은 Kev의 학습 분포 안이다.
  - typed-decisions에서는 Winnow가 Kev-4B보다 낫다(289 vs 176, p < 0.001).
  - Kev와 Winnow는 30문항 셋, NSMC, KLUE-YNAT에서 차이가 없다.
- **typed-decisions**: Winnow(0.7265)와 Kev-9B(0.720)는 데이터셋 카드가 Jev 1.13.0에 매긴 값(0.727, 천장 0.735) 근처다. 카드 수치는 카드 쪽 하네스 값이고, 여기서 다시 돌리지 않았다.
  - gold 분포에 가장 가까운 것은 Clef다(KL 0.196, Brier 0.107). 다음은 Kev-9B(0.209, 0.107)와 Clef-flash(0.220, 0.113)이고, Winnow는 0.286, 0.128이다.
  - report의 uniform 행은 카드의 KL 0.444, Brier 0.238을 재현한다.
- **60ms 미만**: Kev-0.8B(28-58ms)나 Ollaya(15-55ms). typed-decisions의 긴 state는 빼고 본 값이다. Von(transfer-v4·BANKING77-20·NSMC에서 32-51ms)은 영어 전용이다.
  - Kev-0.8B가 typed-decisions, NSMC, KLUE-YNAT(p < 0.001)와 BANKING77(학습 분포 안)에서 더 정확하다. 30문항 셋과 transfer-v4에서는 차이가 없다.
  - 선택지를 뒤집으면 Ollaya는 BANKING77, typed-decisions, KLUE-YNAT에서 답의 30-45%가 바뀐다. Kev-0.8B는 8-19%다.
- **한국어**: Von은 빼야 한다. 영어 전용이다.
  - NSMC에서는 Kev-9B, Winnow, Kev-4B(83-86%)를 구분할 수 없다. Kev-9B는 Jeff(p = 0.008)와 Kev-0.8B(p = 0.012)보다 낫다.
  - KLUE-YNAT에서는 Kev-4B, Jeff, Winnow, Kev-9B(73-74%)를 구분할 수 없다(p ≥ 0.86). Kev-0.8B(63%)와 Ollaya(43%)는 그보다 낮다(p < 0.001).
- **AnyJev**:
  - KLUE-YNAT에서 raw 76%, L0 75%로 Kev-4B, Kev-9B, Winnow, Jeff와 같다(p ≥ 0.33).
  - transfer-v4에서는 둘 다 75%로 Winnow와 같고(p > 0.24), Kev-4B와 Kev-9B보다 낮다(p < 0.001). Kev와의 차이는 전부 정책 홀드아웃 두 출처에서 난다(Kev-4B와의 불일치 49 대 2). 공개 출처 여섯 개에서는 AnyJev가 588개 중 458개로 Kev-4B의 454개와 같다(p = 0.76).
  - typed-decisions에서 raw 61%, L0 63%로 Kev-4B, Kev-9B, Winnow보다 낮다(p < 0.001). 예측이 gold에서 멀다. KL은 raw 3.783, L0 2.813이고, ECE는 0.35와 0.31이다.
  - NSMC에서는 80-81%다. Kev-9B보다 낮고(raw p = 0.040, L0 p = 0.033), 상위 그룹의 나머지와는 같다.
  - L0 vs raw:
    - L0는 raw가 한 번이라도 뒤집힌 모든 셋에서 order-flip을 줄인다(한국어 30 2 → 0, BANKING77 68 → 22, transfer-v4 76 → 27, typed-decisions 101 → 31, KLUE-YNAT 46 → 26).
    - BANKING77(p = 0.005)과 typed-decisions(p < 0.001)에서는 더 정확하다.
    - 대신 첫 호출 시간이 1.5-13.5배 걸린다.
- **Jeff-Qwen3.5-2B**:
  - typed-decisions에서 52%다. 카드는 다른 체크포인트인 Jeff-Gemma4-E2B를 0.561로 올려 두었다.
  - KLUE-YNAT에서는 4B, 9B 모델과 같다.
  - BANKING77에서는 선택지를 뒤집으면 답의 24%가 바뀐다.
- **Kev-4B는 transfer-v4에서 모델 카드를 재현한다.**
  - clean 질문에서 534/656(81.4%, Wilson 78-84)이다. 카드는 같은 656개에서 0.817이다.
  - 나머지 108개 판단은 none_absent, none_present, permuted 변형이 36개씩이다(report의 `## By variant`).

CLM에 대한 이전 발견 (2026-09-28):

- vllm-metal 임베딩은 transformers(MPS, bf16, 마지막 토큰, L2)와 프로브 텍스트 4개에서 cosine 0.9998 이상으로 일치했다. 인코더 쪽은 원본과 같다.
- CLM은 README 예제에서 `department`(billing, 0.987 vs 0.939)와 `frustration`(2.00 vs 1.98)을 재현했다. 하지만 `urgency`는 README의 0.41과 달리 0.852가 나왔다. 같은 티켓에 Laya는 0.795를 냈다. 원인은 찾지 못했다.
- 이 질문셋에서 CLM은 `frustration`을 거의 항상 2.0으로 답하고, `department`는 `billing` 쪽으로 쏠린다.

### BANKING77 20-way

`bench/questions_banking77.jsonl`은 AnyJev의 `banking20` 태스크를 그대로 재구성한 셋이다. 그래서 AnyJev README 수치와 직접 비교할 수 있다.

- BANKING77 train에서 가장 흔한 인텐트 20개를 라벨 id 순서로, 설명 없이 나열한다.
- 질문은 "What is the customer's intent?"다.
- 고정 리비전의 `mteb/banking77`에서 test 문항을 읽어 `random.Random(0)`으로 섞고 앞 300개를 쓴다.
- `uv run bench/make_banking77.py`로 다시 만들 수 있다.
- 정확도는 위 표에 있다. 이 셋의 order-flip(2026-09-30, Clef·Clef-flash·Von은 2026-10-05)은 아래와 같다.

| 엔진 | order-flip |
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

- **Kev의 BANKING77 수치는 학습 분포 안에서 잰 값이다.**
  - Kev 카드 세 개 모두 학습 데이터에 `legacy-datasets/banking77`을 적어 두었다.
  - 여기서는 Kev-0.8B를 더 큰 크기와 구분할 수 없다(p ≥ 0.80). transfer-v4, typed-decisions, KLUE-YNAT에서는 구분된다(p < 0.001).
  - 이 셋으로 학습하지 않은 엔진과 비교하지 않는다.
- **AnyJev는 여기서 자기 README를 재현한다.**
  - README 수치(Qwen3-8B, BANKING77 20-way, test 300문항): order-flip 0.230 raw → 0.073 L0, 정확도 0.747 → 0.803
  - 이번 측정: order-flip 68/300(0.227) → 22/300(0.073), 정확도 224/300(0.747) → 241/300(0.803). 2026-09-29 측정과 같은 수다.
  - Wilson 구간(69-79와 75-84)은 겹친다. 하지만 대응 검정으로는 정확도 향상이 보인다. L0만 맞힌 판단이 25개, raw만 맞힌 판단이 8개로 p = 0.005다.
  - L0는 여기서 Winnow와 같다(p = 0.868).
- **여기서는 Clef-flash(300개 중 290)와 Clef(287)가 가장 높다.** 카드가 BANKING77을 자사 벤치마크 중 하나로 보고하고(인텐트 77개 전체 macro-F1: Clef 94.2, Clef-flash 90.9) 학습 데이터는 공개하지 않는다. 그래서 이 셋이 학습 분포 안에 있는지는 알 수 없다.
- **L0는 선택지가 많으면 느리다.** 순환 이동마다 프롬프트를 한 번씩 prefill하는데, 20-way 질문이면 최대 20번이다. 첫 호출 p50이 raw 582.3ms 대비 7860.1ms다.
- 참고용이며 직접 비교할 수는 없다: 인텐트 77개 전체에서 Laya zero-shot 38%, Jev 76%라는 수치가 있다(dhruvmehra/jevbench, Laya 파인튜닝 글에서 인용, 여기서 검증하지 않음). 그 셋은 77-way이고 이 셋은 20-way다.

## FAQ

**System One 결정 모델이 뭔가?**
텍스트 하나에 대해 타입이 정해진 질문에 답하는 모델이다. 질문은 선택지 고르기, 예/아니오 확률, 점수 중 하나다. 한 번의 forward pass로 답하고, 텍스트를 생성하는 대신 보정된 확률을 준다. TypeSafe의 Jev가 호스팅되는 원조이고, 이 레포의 모든 엔진이 Jev의 `POST /v1/systemone` 형식을 쓴다.

**맥에서 가장 정확한 로컬 결정 모델은?**
Kev-9B, Kev-4B, Winnow-E4B, Clef-flash, Clef다. Kev의 transfer-v4 셋에서는 Kev, Clef-flash, Clef가 앞선다(80-83% 대 77%). 다만 정책 구조 홀드아웃 두 출처에서만 앞선다. typed-decisions에서는 Winnow, Kev-9B, Clef-flash, Clef가 앞선다(71-73% 대 Kev-4B 67%). 한국어 셋에서는 이들을 구분할 수 없다(NSMC에서 Clef가 Kev-4B보다, p = 0.043, KLUE-YNAT에서 Kev-9B보다, p = 0.047 나은 것만 예외). Clef-flash는 7개 셋 중 4개에서 이들 가운데 ECE가 가장 낮다(하나는 Clef와 같다). Clef(27B)는 어느 셋에서도 Clef-flash보다 정확하지 않고, 시간은 3.3-4.5배 든다.

**가장 빠른 것은?**
Laya를 돌리는 Ollaya다. 첫 호출 p50이 14.7-55.3ms이고, typed-decisions의 긴 state에서는 292.9ms다. 다음은 Kev-0.8B(27.9-57.6ms)이고, 모든 셋에서 정확도가 Ollaya와 같거나 더 높다. Von은 transfer-v4(31.8ms)와 BANKING77-20(38.1ms)에서 Kev-0.8B보다 빠르다. 정확도는 영어 30문항·transfer-v4·typed-decisions에서 Kev-0.8B와 같고(p ≥ 0.068), BANKING77-20과 한국어 셋 세 개에서는 낮다(p < 0.001).

**AnyJev가 vllm-metal에서 동작하나?**
한 가지만 바꾸면 동작한다. vllm-metal 0.30.0은 `--logprobs-mode` 설정과 상관없이 raw logprob을 주므로, 어댑터가 라벨마다 `logprob_token_ids`로 요청한다. 그렇게 하면 AnyJev는 BANKING77-20에서 자기 README를 재현한다.

**한국어로도 답하나?**
Kev-4B, Kev-9B, Winnow, Jeff는 된다. NSMC에서 앞의 셋이 83-86%, KLUE-YNAT에서 넷 모두 73-74%다. AnyJev도 학습 없이 된다(NSMC 80-81%, KLUE-YNAT 75-76%). Clef-flash와 Clef도 된다(NSMC 86-87%, KLUE-YNAT 77-78%, Kev·Winnow와 같다. NSMC에서 Clef는 Kev-4B보다, p = 0.043, KLUE-YNAT에서 Kev-9B보다, p = 0.047 낫다). Ollaya는 `laya:multilingual`이 필요하고, 56%와 43%가 나온다. Von은 안 된다. 카드에 영어 전용이라 적혀 있고, NSMC 51%, KLUE-YNAT 15%다.

**메모리는 얼마나 필요한가?**
Kev-4B 서버는 기동 직후 RSS 2.7GB였다. Winnow는 내려받는 크기가 8.0GB, Clef-flash Q8_0 GGUF는 9.7GB, Clef Q4_K_M GGUF는 19.2GB, Von 가중치는 3.2GB다. AnyJev와 CLM은 vllm-metal에서 Qwen3-8B를 돌리는데, vllm-metal이 약 20GB(Metal wired 한도 28.1GB의 0.7)를 예약한다. 그래서 하나씩 측정한다.

## vllm-metal vs Ollama, Qwen3-8B 채팅 (2026-09-28, M3 Max 36GB)

`scripts/serve-llm.sh`는 vllm-metal로 채팅 모델을 서빙한다. `bench/llm_compare.py`는 같은 프롬프트 8개(최대 256토큰, temperature 0, thinking 끔)를 OpenAI 호환 서버에 보내고 `bench/llm_report.md`에 행을 추가한다.

| 서버 | TTFT p50 ms | 단건 디코드 tok/s | 동시 8개, 전체 tok/s | 메모리 |
|---|---|---|---|---|
| vllm-metal, bf16 | 197 | 16.8 | 77.3 | 약 20GB 예약 (0.7 × wired 한도) |
| vllm-metal, MLX 4bit | 291 | 30.2 | 48.6 | 같은 예약 |
| Ollama 0.34.2, Q4 (`qwen3:8b`) | 233 | 29.4 | 30.2 | 10GB (32K 컨텍스트) |

- 사용자 한 명 기준으로는 4bit 모델의 디코드 속도가 vllm-metal과 Ollama에서 거의 같다(약 30 tok/s). Ollama가 메모리를 절반만 쓴다.
- 동시 요청 8개에서는 vllm-metal만 배치 처리를 한다. Ollama는 기본 설정 그대로 두었더니 단건 속도로 처리했다.
- Ollama의 OpenAI 엔드포인트는 `/no_think`와 `think: false`를 무시한다. thinking을 끄려면 `reasoning_effort: "none"`을 보내야 한다. 그러지 않으면 thinking 토큰이 끝날 때까지 `content`가 비어 있다.
- 이 수치는 프롬프트 8개로 한 번 잰 값이고 반복 측정하지 않았다.

## 라이선스

MIT. [LICENSE](LICENSE) 참고. 벤치마크한 엔진과 데이터셋은 각자의 라이선스를 따른다.
