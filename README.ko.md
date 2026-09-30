# local-jev-bench: Apple Silicon 맥에서 돌리는 Jev 스타일 결정 모델 (AnyJev, Kev, Laya, CLM)

[English](README.md) | 한국어

local-jev-bench는 Jev 스타일의 System One 결정 모델을 Apple Silicon 맥에서 로컬로 돌린다. 결정 모델은 타입이 정해진 질문을 받아 보정된 확률로 답하는 모델이다. 여기서는 영어·한국어 고객 문의와 공개 셋인 BANKING77 20-way 인텐트 분류에서 여러 엔진을 같은 질문으로 비교한다.

**핵심 결과 (M3 Max 36GB, 2026-09-29).** 모든 셋에서 Kev-4B가 가장 정확하다. 영어 279/315, 한국어 273/315, BANKING77-20 266/300이고, 호출당 215~240ms다. 호출당 50ms 미만인 엔진은 Ollaya(Laya)뿐이다(20.7~35.8ms). 대신 선택지가 많아지면 정확도가 떨어진다. Qwen3-8B 위의 AnyJev는 BANKING77-20에서 자기 README 수치를 그대로 재현했다(order-flip 68/300 → 22/300). 자세한 내용은 [결과](#결과-2026-09-29-m3-max-36gb) 절에 있다.

엔진은 네 가지다.

- **CLM** ([Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM)): [vllm-metal](https://github.com/vllm-project/vllm-metal)로 서빙하는 Qwen3-8B 인코더와 75MB짜리 CLM 헤드.
- **Ollaya** ([ollaya-dev/ollaya](https://github.com/ollaya-dev/ollaya)): `laya` 모델을 MLX로 돌린다.
- **AnyJev** ([nokia-applied-research/AnyJev](https://github.com/nokia-applied-research/AnyJev)): 학습이 필요 없다. vllm-metal 위 Qwen3-8B에서 라벨 logprob을 읽는다. `anyjev-raw`는 보정 없이, `anyjev-l0`는 선택지를 순환 이동하고 배치 prior로 보정한다.
- **Kev** ([jaredpalmer/kev](https://github.com/jaredpalmer/kev)): Qwen3.5-4B-Base 위의 LoRA를 MLX로 돌린다.

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

[Kev](https://github.com/jaredpalmer/kev)는 자기 체크아웃 안에 별도 uv 환경(torch, mlx-lm)을 둔다. 그래서 이 레포의 의존성이 아니다. `git clone https://github.com/jaredpalmer/kev ~/workspace/tak-bro/kev && (cd ~/workspace/tak-bro/kev && uv sync --extra serve)`로 준비한다. `scripts/serve-kev.sh`는 Qwen3.5-4B-Base 위에 `jaredpalmer/kev-4b` 어댑터를 얹은 `/v1/systemone` 서버를 띄운다.

`contrastive-lm`은 `vllm`을 의존성으로 선언하지만, 실제로는 임베딩 엔드포인트를 HTTP로 호출할 뿐이다. 그래서 `pyproject.toml`에서 이 의존성을 빼서 vLLM이 두 번 설치되지 않게 했다.

## 포트

모든 서버는 `127.0.0.1`에만 바인드한다.

| 포트 | 프로세스 | 시작 |
|---|---|---|
| 8091 | 작은 임베딩 모델(smoke 전용) | `scripts/serve-embed.sh mlx-community/Qwen3-Embedding-0.6B-8bit 8091 embed-small` |
| 8090 | CLM용 Qwen3-8B 인코더 | `scripts/serve-embed.sh Qwen/Qwen3-8B 8090 qwen3-8b` |
| 8700 | CLM System One API | `scripts/serve-clm.sh` |
| 11435 | Ollaya 데몬 | `OLLAYA_HOST=127.0.0.1:11435 ~/.local/bin/ollaya serve` |
| 8092 | AnyJev용 Qwen3-8B 생성 서버 | `scripts/serve-llm.sh Qwen/Qwen3-8B 8092 qwen3-8b` |
| 8710 | AnyJev System One API (`anyjev-raw`, `anyjev-l0`) | `scripts/serve-anyjev.sh` |
| 8009 | Kev System One API (`kev-latest`) | `KEV_DIR=~/workspace/tak-bro/kev scripts/serve-kev.sh` |

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
uv run python bench/run.py --smoke --engine anyjev-raw --engine anyjev-l0 --engine kev   # shape only
uv run python bench/run.py --engine <e> --reps 1 --questions bench/questions_banking77.jsonl
uv run python bench/score.py bench/runs/questions --check   # report.md matches the raw logs
uv run python bench/make_sets.py transfer-v4        # Kev의 분포 밖 development 세트(764)를 bench/data/에, gitignore
uv run python bench/make_sets.py typed-decisions    # LocalLLaMA/typed-decisions test(400케이스, 2,000판단), gold 분포 포함
uv run pytest -q                              # offline tests, fake servers, no model loaded
```

AnyJev와 Kev는 각각 Metal 메모리 대부분을 쓴다. 그래서 엔진은 하나씩 따로 측정한다.

1. 그 엔진의 서버를 띄운다.
2. `bench/run.py --engine <e> --questions <셋>`을 실행한다.
3. 서버를 내린다.

실행은 모든 호출(확률을 포함한 답·지연·메모리 압력)을 `bench/runs/<셋>/<엔진>.jsonl`에 남기고 그 엔진의 이전 기록을 덮어쓴다. `bench/score.py`는 그 디렉터리의 기록 전부로 `bench/runs/<셋>/report.md`를 다시 만든다. 다른 버전의 셋이나 다른 `--reps`로 잰 기록은 거부한다. AnyJev 어댑터는 측정할 때마다 재시작한다. L0의 배치 prior가 어댑터가 처리한 모든 호출에 걸쳐 누적되기 때문이다.

벤치마크는 상태와 질문 지시문을 합친 텍스트가 Qwen3 토큰 2048개를 넘는 문항을 거부한다. 이 텍스트가 CLM이 실제로 임베딩하는 입력인데, `clm-serve`는 긴 입력을 오류 없이 잘라 버리기 때문이다. 손으로 만든 30문항 정확도는 동작 확인용이지 벤치마크가 아니다.

## 결과 (2026-09-29, M3 Max 36GB)

질문셋마다 30문항을 3번씩 돌렸고, 한 번에 엔진 하나만 띄웠다.

- 정확도에는 Wilson 95% 구간을 붙였다.
- `order-flip`은 선택지를 역순으로 나열했을 때 답이 바뀐 문항 수다.
- 전체 표는 `bench/report.md`(영어)와 `bench/report_ko.md`(한국어)에 있다.
- Ollaya는 영어에 `laya`, 한국어에 `laya:multilingual`(`OLLAYA_MODEL=laya:multilingual`)을 썼다. 영어 모델로는 한국어에서 33%였다(2026-09-28).

| 엔진 | 첫 호출 p50 ms (영/한) | 정확도 영어 | 정확도 한국어 | order-flip (영/한) |
|---|---|---|---|---|
| Kev-4B (MLX, bf16) | 225.9 / 215.0 | 279/315 (89%, 85-92) | 273/315 (87%, 82-90) | 1/30 / 1/30 |
| AnyJev L0 (Qwen3-8B, vllm-metal) | 434.5 / 444.1 | 254/315 (81%, 76-85) | 264/315 (84%, 79-87) | 0/30 / 0/30 |
| AnyJev raw (Qwen3-8B, vllm-metal) | 178.5 / 182.2 | 258/315 (82%, 77-86) | 261/315 (83%, 78-87) | 0/30 / 2/30 |
| Ollaya (영 `laya` / 한 `laya:multilingual`, MLX) | 35.8 / 20.7 | 237/315 (75%, 70-80) | 207/315 (66%, 60-71) | 2/30 / 6/30 |
| CLM-8B (vllm-metal, bf16) | 225.7 / 266.5 | 123/315 (39%, 34-45) | 129/315 (41%, 36-46) | 0/30 / 0/30 |

- 어떤 측정에서도 메모리 압력은 `critical`까지 가지 않았다. 가장 높았던 것은 CLM의 `warn`이다.
- 실행 로그를 보면 생성 서버를 막 띄운 뒤 첫 AnyJev 실행은 호출당 약 1초가 걸렸다(워밍업). 그 행은 버렸고, 위 AnyJev 두 행은 워밍된 서버에서 잰 값이다.

CLM은 선택지 텍스트 임베딩끼리의 유사도로 답을 고른다. 그래서 order-flip 0은 순서에 강해서라기보다 구조 때문일 수 있다(미검증). CLM의 order-flip 열은 다른 엔진과 비교하지 않는다.

어떤 엔진을 쓸까 (구간이 겹치면 차이가 없는 것으로 본다):

- **호출당 50ms 미만**: Ollaya만 가능하다(첫 호출 p50 영어 35.8ms, 한국어 20.7ms).
  - 영어에서 Ollaya보다 확실히 나은 엔진은 Kev뿐이다(279 vs 237, 85-92 vs 70-80).
  - 한국어에서는 Kev와 AnyJev 두 레벨 모두 Ollaya보다 낫다(273, 264, 261 vs 207). 60-71은 82-90, 79-87, 78-87 어느 것과도 겹치지 않는다.
- **한국어**: Kev나 AnyJev를 쓴다. 둘은 구분되지 않는다(82-90 vs 79-87 / 78-87).
  - Kev는 4B 모델 하나면 된다. 2026-09-29 smoke 실행에서 서버 RSS는 기동 직후 2.7GB였다.
  - AnyJev는 vllm-metal 위에 Qwen3-8B가 필요하다.
- **이 30문항으로는 구분되지 않는 조합**:
  - Kev와 AnyJev 두 레벨, 두 셋 모두. 영어의 Kev vs L0는 겨우 겹친다(Kev 하한 84.58, L0 상한 84.62).
  - AnyJev raw와 L0.
  - 영어에서 AnyJev 두 레벨과 Ollaya.

CLM에 대한 이전 발견 (2026-09-28):

- vllm-metal 임베딩은 transformers(MPS, bf16, 마지막 토큰, L2)와 프로브 텍스트 4개에서 cosine 0.9998 이상으로 일치했다. 인코더 쪽은 원본과 같다.
- CLM은 README 예제에서 `department`(billing, 0.987 vs 0.939)와 `frustration`(2.00 vs 1.98)을 재현했다. 하지만 `urgency`는 README의 0.41과 달리 0.852가 나왔다. 같은 티켓에 Laya는 0.795를 냈다. 원인은 찾지 못했다.
- 이 질문셋에서 CLM은 `frustration`을 거의 항상 2.0으로 답하고, `department`는 `billing` 쪽으로 쏠린다.

### BANKING77 20-way (2026-09-29)

`bench/questions_banking77.jsonl`은 AnyJev의 `banking20` 태스크를 그대로 재구성한 셋이다. 그래서 AnyJev README 수치와 직접 비교할 수 있다.

- BANKING77 train에서 가장 흔한 인텐트 20개를 라벨 id 순서로, 설명 없이 나열한다.
- 질문은 "What is the customer's intent?"다.
- 고정 리비전의 `mteb/banking77`에서 test 문항을 읽어 `random.Random(0)`으로 섞고 앞 300개를 쓴다.
- `uv run bench/make_banking77.py`로 다시 만들 수 있다.
- 문항당 1번 호출했다(`--reps 1`). 전체 표는 `bench/report_banking77.md`에 있다.

| 엔진 | 첫 호출 p50 ms | 정확도 | order-flip |
|---|---|---|---|
| Kev-4B | 240.0 | 266/300 (89%, 85-92) | 23/300 (8%) |
| AnyJev L0 | 8489.3 | 241/300 (80%, 75-84) | 22/300 (7%) |
| AnyJev raw | 624.0 | 224/300 (75%, 69-79) | 68/300 (23%) |
| Ollaya `laya` | 27.8 | 180/300 (60%, 54-65) | 93/300 (31%) |
| CLM-8B | 359.7 | 61/300 (20%, 16-25) | 0/300 (0%) |

- **AnyJev는 여기서 자기 README를 재현한다.**
  - README 수치(Qwen3-8B, BANKING77 20-way, test 300문항): order-flip 0.230 raw → 0.073 L0, 정확도 0.747 → 0.803
  - 이번 측정: order-flip 68/300(0.227) → 22/300(0.073), 정확도 224/300(0.747) → 241/300(0.803)
- **L0는 order-flip을 줄인다. 정확도 향상은 300문항에서는 보이지 않는다.** order-flip은 68/300에서 22/300으로 줄었지만, 정확도 구간 69-79와 75-84는 겹친다.
- **여기서 가장 정확한 엔진은 Kev다.** 구간 85-92가 다른 모든 엔진의 구간과 겹치지 않는다. order-flip은 L0와 1문항 차이(23/300 vs 22/300)이고, 속도는 240.0ms 대 8489.3ms다.
  - L0가 느린 이유: 순환 이동마다 프롬프트를 한 번씩 prefill하는데, 20-way 질문이면 최대 20번이다.
- **Ollaya는 선택지가 20개면 60%로 떨어지고**, 선택지를 뒤집으면 답의 31%가 바뀐다.
- 참고용이며 직접 비교할 수는 없다: 인텐트 77개 전체에서 Laya zero-shot 38%, Jev 76%라는 수치가 있다(dhruvmehra/jevbench, Laya 파인튜닝 글에서 인용, 여기서 검증하지 않음). 그 셋은 77-way이고 이 셋은 20-way다.

## FAQ

**System One 결정 모델이 뭔가?**
텍스트 하나에 대해 타입이 정해진 질문에 답하는 모델이다. 질문은 선택지 고르기, 예/아니오 확률, 점수 중 하나다. 한 번의 forward pass로 답하고, 텍스트를 생성하는 대신 보정된 확률을 준다. TypeSafe의 Jev가 호스팅되는 원조이고, 이 레포의 모든 엔진이 Jev의 `POST /v1/systemone` 형식을 쓴다.

**맥에서 가장 정확한 로컬 결정 모델은?**
이번 측정에서는 Kev-4B다. 영어와 BANKING77-20에서 89%, 한국어에서 87%이고, BANKING77-20에서는 다른 모든 엔진보다 확실히 앞선다(Wilson 85-92).

**가장 빠른 것은?**
Laya를 돌리는 Ollaya다. 30문항 셋에서 첫 호출 p50이 20.7~35.8ms이고, 나머지 엔진은 178.5~444.1ms다.

**AnyJev가 vllm-metal에서 동작하나?**
한 가지만 바꾸면 동작한다. vllm-metal 0.30.0은 `--logprobs-mode` 설정과 상관없이 raw logprob을 주므로, 어댑터가 라벨마다 `logprob_token_ids`로 요청한다. 그렇게 하면 AnyJev는 BANKING77-20에서 자기 README를 재현한다.

**한국어로도 답하나?**
Kev(273/315)와 AnyJev(261~264/315)는 된다. 30문항으로는 둘을 구분할 수 없다. Ollaya는 `laya:multilingual`이 필요하고, 207/315까지 나온다.

**메모리는 얼마나 필요한가?**
Kev-4B 서버는 기동 직후 RSS 2.7GB였다. AnyJev와 CLM은 vllm-metal에서 Qwen3-8B를 돌리는데, vllm-metal이 약 20GB(Metal wired 한도 28.1GB의 0.7)를 예약한다. 그래서 하나씩 측정한다.

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
