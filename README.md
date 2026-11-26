# Glossary RAG Pipeline

한영 병렬 데이터에서 용어집을 구축하고, 별도의 평가 원문과 한국어 정답본을 사용해 Plain No-RAG, CoT No-RAG, 용어 RAG, 용어+예문 RAG를 비교한다.

실행 결과는 기본적으로 `result/<실행시각>/`에 모인다. 번호를 붙인 주요 파일은 상세 결과를 가리키는 상대 링크다. 단계별 원본 파일과 실행 로그는 `details/<단계>/attempt-*/`, 실행 설정과 완료 상태는 `details/pipeline.json`에 있다. 실행 폴더를 통째로 옮기면 상대 링크가 유지된다.

## How to use

### 0. 새 환경 준비

**실행 환경과 모델이 이미 준비돼 있으면 사용하는 Python 환경을 활성화하고 [1번](#1-10문장-예제-실행)으로 넘어간다.** 새 서버에서는 아래 순서로 준비한다.

#### 환경 및 의존성

Linux, Python 3.11, CUDA를 지원하는 NVIDIA GPU와 드라이버를 기준으로 한다. 아래 설치 명령의 CUDA 12.6은 한 가지 예시다. 시스템에 맞는 PyTorch와 torchvision 조합을 사용한다. [PyTorch 설치표](https://pytorch.org/get-started/previous-versions/) · [bitsandbytes 설치 지원표](https://huggingface.co/docs/bitsandbytes/main/installation)

| 용도 | 주요 패키지 |
|---|---|
| 한영 토큰화·기본형 | `kiwipiepy`, `nltk`, `numpy`, NLTK 영어 POS 태거와 WordNet |
| BGE 정렬·용어 후보 매핑 | CUDA `torch`, `transformers`, `sentence-transformers`, `huggingface-hub`, `sentencepiece`, `protobuf`, `safetensors` |
| Gemma 판정·번역 | Transformers 5, `accelerate`, `torchvision`, `pillow`, `json-repair`, NF4 사용 시 `bitsandbytes` |
| 용어 검색·문장 선별 | `pyahocorasick` |
| 번역 평가 | `unbabel-comet`, CUDA `torch`, `sacrebleu`, `pyahocorasick`. Lightning, torchmetrics 등은 COMET과 함께 설치된다. |

Gemma 4와 COMET은 서로 다른 Transformers 버전을 사용하므로 두 환경으로 설치한다. `unbabel-comet==2.2.7`은 Transformers 4와 Hugging Face Hub 1 미만을 요구한다. 파이프라인은 `eval_python` 또는 `GLOSSARY_EVAL_PYTHON`에 지정한 Python을 08단계에서 사용한다. 이미 동작하는 통합 환경에서는 같은 Python을 사용할 수 있다. [Gemma 모델 카드](https://huggingface.co/google/gemma-4-12B-it) · [COMET 패키지](https://pypi.org/project/unbabel-comet/2.2.7/)

#### 설치 순서

프로젝트 루트에서 실행한다.

```bash
# 01~07단계
conda create -y -n glossary-rag python=3.11 pip
conda activate glossary-rag
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install \
  "numpy<2" "kiwipiepy==0.24.1" "nltk==3.10.4" \
  "transformers==5.12.1" "sentence-transformers==5.6.1" \
  "accelerate>=1,<2" "huggingface-hub>=1,<2" \
  "sentencepiece>=0.2,<0.3" "protobuf>=4.24.4,<5" \
  safetensors pillow json-repair pyahocorasick
python -m pip install "bitsandbytes==0.50.0"
python 01-word-alignment/setup_resources.py
python -m pip check
```

예제는 NF4를 사용한다. 기본 토크나이저는 `C_nltk_treebank`이며, 별도로 spaCy 방식을 선택할 때는 해당 언어 모델도 준비한다.

```bash
# 08단계
conda create -y -n glossary-eval python=3.11 pip
conda activate glossary-eval
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install "setuptools<81" "transformers==4.57.6" \
  "unbabel-comet==2.2.7" "sacrebleu>=2.4,<3" pyahocorasick
python -m pip check
export GLOSSARY_EVAL_PYTHON="$(python -c 'import sys; print(sys.executable)')"
python -c 'import comet, ahocorasick, torch; print(torch.cuda.is_available())'
conda activate glossary-rag
python -c 'import kiwipiepy, nltk, sentence_transformers, torch; from transformers import AutoModelForMultimodalLM; print(torch.cuda.is_available())'
```

두 확인 명령에서 `True`가 나오면 CUDA가 인식된 상태다. 환경 변수는 현재 터미널에서 사용된다. 설정 파일의 `eval_python`에 경로를 적으면 그 경로가 우선한다.

#### 모델 가중치 준비

온라인 첫 실행에서 모델을 받을 수 있다. 미리 준비할 때는 아래처럼 내려받고, 같은 터미널에서 예제를 실행한다. BGE-M3 로컬 폴더에는 모델 설정, fast tokenizer, 모델 가중치와 정렬용 `colbert_linear.pt`가 필요하다. 오래된 PyTorch 환경에서는 BGE-M3를 safetensors로 준비한다. [BGE-M3 모델 파일](https://huggingface.co/BAAI/bge-m3/tree/main)

```bash
# glossary-rag 환경
export GLOSSARY_BGE_MODEL="$(hf download BAAI/bge-m3 \
  --include "*.json" "*.model" "colbert_linear.pt" "model.safetensors" "pytorch_model.bin")"
export GLOSSARY_GEMMA_MODEL="$(hf download google/gemma-4-12B-it)"

# glossary-eval 환경
conda activate glossary-eval
hf download Unbabel/wmt22-comet-da --include "hparams.yaml" "checkpoints/model.ckpt"
hf download xlm-roberta-large --include "*.json" "*.model"
conda activate glossary-rag
```

환경 변수로 내려받은 위치를 전달하면 정렬과 용어 후보 매핑이 같은 BGE 모델을 사용한다. 설정 파일에서 `bge_model`, `gemma_model`을 직접 지정할 수도 있다. 필요한 가중치를 모두 준비한 후 `local_files_only`를 `true`로 설정하면 오프라인으로 실행한다. COMET의 로컬 체크포인트를 지정할 때는 `comet_checkpoint`로 `checkpoints/model.ckpt`를 가리키고, 그 상위 모델 폴더에 `hparams.yaml`을 함께 둔다. [COMET 모델 카드](https://huggingface.co/Unbabel/wmt22-comet-da)

### 1. 10문장 예제 실행

[`examples/demo-10/`](examples/demo-10/)에 구축용 [train.json](examples/demo-10/train.json) 10쌍, 별도 평가용 [eval.json](examples/demo-10/eval.json) 10쌍, 실행용 [config.json](examples/demo-10/config.json)을 저장해 두었다. 구축용은 간호 관련 표현이 반복되는 문장을 골랐고, 평가용은 PP 검증 데이터에서 `nurse`가 나오는 문장을 골랐다. 프로젝트를 옮겨도 같은 예제 파일을 사용할 수 있다.

준비된 Python 환경을 활성화한 상태에서 실행한다.

```bash
cd /home/hjs/PP/glossary-rag-pipeline
CUDA_VISIBLE_DEVICES=1 python pipeline.py run \
  --config examples/demo-10/config.json \
  --run-dir result/demo-10
```

GPU 번호는 사용 가능한 장치에 맞게 바꾼다. 실행을 이어갈 때는 `run --run-dir result/demo-10`만 지정한다. 처음부터 다시 실행할 때는 새 결과 경로를 지정한다.

기본 BGE-M3 모델 ID로 실행하면, 확인된 safetensors 가중치와 호환되는 설정 파일이 로컬 캐시에 있을 때 이를 우선 사용한다. 정렬과 용어 후보 매핑은 실행 폴더의 `details/model-assets/bge-m3/`에 연결된 같은 파일을 읽는다. 캐시에 필요한 파일이 없고 Transformers가 `pytorch_model.bin`을 불러오지 못하면 0번의 PyTorch 설치 방법을 따르거나, 설정의 `bge_model`에 safetensors가 포함된 로컬 모델 폴더를 지정한다.

### 2. 단계별 결과 확인

완료된 단계의 주요 파일은 `result/demo-10/`에 순서대로 보인다. JSONL은 줄마다 JSON 객체 하나, CSV는 첫 행에 열 이름, TXT는 줄마다 문장 하나를 담는다.

| 단계 | 공개 결과 파일 | 형태와 확인할 항목 |
|---|---|---|
| [01. 한영 정렬](01-word-alignment/UPSTREAM.md) | `01-aligned-terms.jsonl` | 문장쌍당 1행. `line_id`, `ko`, `en`, `pairs[]`가 있다. 개별 쌍에 실제 표기, 기본형, `ko_span`·`en_span`의 원문 문자 위치, 유형, 유사도와 근거를 담는다. |
| [02. 대표어 정리](02-word-similar/UPSTREAM.md) | `02-representative-candidates.jsonl`, `02-representative-groups.csv`, `02-group-pairs.csv` | JSONL은 원문과 `ko_candidates[]`·`en_candidates[]`를 담는다. 그룹 CSV는 대표어와 소속 표현을, 대응 CSV는 실제 한영 용어쌍·양쪽 대표어·배정 상태·빈도를 담는다. |
| [03. 한영 용어 후보 매핑 및 합성성 검사](03-word-candidates/UPSTREAM.md) | `03-domain-review-input.jsonl` | 후보쌍당 1행. `surface_a`는 한국어, `surface_b`는 영어다. `frequency`, `category`, `sample_sentences[]`에 빈도·유형·출처가 있는 한영 예문을 담는다. |
| [04. 도메인 검수](04-word-domain-check/UPSTREAM.md) | `04-domain-decisions.jsonl`, `04-target-domain.jsonl`, `04-major-domain.jsonl`, `04-other-domain.jsonl` | 용어쌍당 1행. `terms.ko/en`, `status`, `meaning`, `domains[]`, `reason`, `evidence[]`가 있다. 전체 판정에는 제외·보류·오류도 포함한다. 분야별 채택이 없으면 해당 파일은 비어 있다. |
| [05. RAG 용어집](05-glossary/UPSTREAM.md) | `05-glossary.jsonl`, `05-glossary-with-examples.jsonl` | 용어당 1행. 공통 필드는 `en`과 `ko`이며, 채택 사유가 있으면 `reason`을 담는다. 예문판에는 `examples[]`를 추가한다. |
| [06. 평가 문장 준비](06-sentence-selection/UPSTREAM.md) | `06-eval-source.en`, `06-eval-reference.ko` | 같은 행끼리 대응하는 영어 원문과 한국어 정답 TXT다. 예제는 평가 입력 10쌍을 그대로 사용한다. |
| [07. 네 조건 번역](07-translation/UPSTREAM.md) | `07-no-rag.ko.txt`, `07-rag-terms.ko.txt`, `07-rag-examples.ko.txt`, `07-no-rag-plain.ko.txt` | 원문 행 순서를 유지한 한국어 번역 TXT 4개다. Plain은 간단한 번역 지시만 사용한다. 상세 실행 CSV에서 `term_hits`와 `parse_status`를 확인한다. |
| [08. 번역 평가](08-evaluation/UPSTREAM.md) | `08-four-conditions.csv`, `08-terms-vs-no-rag.csv`, `08-examples-vs-no-rag.csv` | 4조건 CSV의 첫 네 `summary` 행은 용어 인식률, 평균 COMET, 코퍼스 BLEU, 평균 ROUGE와 CoT No-RAG 대비 변화량을 담는다. `sentence` 행에는 번역 네 개와 용어 일치 내역이 있다. 두 쌍별 CSV도 함께 생성된다. |

03단계 후보, 04단계 판정, 05단계 예문 용어집의 주요 필드를 각각 발췌한 예시다.

```jsonl
{"surface_a":"간호사","surface_b":"nurse","frequency":2,"category":"polysemy_candidate","sample_sentences":[{"source_pair":"KO_0001:EN_0001","ko_text":"간호사가 환자를 사정했다.","en_text":"A nurse assessed the patient."}]}
{"id":"L000001","terms":{"ko":"간호사","en":"nurse"},"status":"accept","domains":["의학/간호학"]}
{"en":"nurse","ko":"간호사","examples":[{"en":"A nurse assessed the patient.","ko":"간호사가 환자를 사정했다."}]}
```

중간 규칙을 확인할 때는 `details/`의 토큰과 Gemma 선별 내역(01), 직역·과다 확장·합성성 검사 통과 기록(03), 분류 요약과 체크포인트(04)를 본다. 평가에서는 `term_rate_delta_pp`와 `comet_delta`를 먼저 본다. 원문에 평가 용어가 0건이면 인식률 칸은 비고 COMET은 계산된다.

### 3. 단계별 실행과 재실행

```bash
python pipeline.py init --config examples/demo-10/config.json --run-dir result/demo-10-staged
CUDA_VISIBLE_DEVICES=1 python pipeline.py stage 01-word-alignment --run-dir result/demo-10-staged
CUDA_VISIBLE_DEVICES=1 python pipeline.py stage 02-word-similar --run-dir result/demo-10-staged
# 01·02 완료 결과를 재사용해 03부터 이어서 실행
CUDA_VISIBLE_DEVICES=1 python pipeline.py run --run-dir result/demo-10-staged
```

완료된 특정 단계를 새로 돌릴 때는 `stage`에 `--rerun`을 붙인다. 새 결과가 생성되면 이후 단계의 공개 결과와 완료 상태가 갱신된다. 이전 시도의 파일은 `details/`에 남는다. 같은 용어집을 써서 다른 평가 문장을 시험할 때는 `stage 06-sentence-selection --rerun`에 새 `--eval-en`과 `--eval-ko`를 지정한 뒤 `run --run-dir`로 이어간다.

## 완료한 정렬 결과에서 이어서 실행

완료된 C_nltk_treebank 정렬을 `--stage 2 --input`으로 지정하면, 01단계 결과를 등록하고 02 대표어 그룹화부터 진행한다. 용어 구축용 원문 경로를 다시 입력할 필요가 없다. 정렬 폴더 대신 최종 `term-pairs.jsonl`을 지정해도 된다. 기존의 `--alignment-result` 옵션도 사용할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=1 python pipeline.py run \
  --stage 2 --input /home/hjs/PP/word-alignment_v1.4/result/261120_011214 \
  --eval-json /home/hjs/PP/data/validation/의학/간호학.json \
  --domain 의학/간호학 --load-in-4bit \
  --run-dir result/from-261120_011214
```

짧게 확인할 때는 새 실행에 `--max-sentences 300 --eval-limit 10`을 추가한다. 이미 정렬한 파일을 가져오면 `--max-sentences`는 02단계에 넘길 문장 수를 제한한다. 통합 실행을 중단했다면 `run --run-dir`만 지정해 완료한 단계 다음부터 이어간다. 외부 정렬 실행이 아직 Gemma 선별 중이면 해당 정렬 실행기에서 체크포인트로 선별을 완료한 뒤 최종 파일을 넘긴다.

## 단계 번호와 입력 경로로 시작

새 실행의 공통 형식은 **`run --stage N --input <파일 또는 결과 폴더> --run-dir <새 실행 폴더>`**다. 한 파일이면 충분한 01~05단계는 파일이나 해당 파일이 있는 폴더를 받는다. 06~08단계는 입력이 여러 개이므로 폴더를 받는다. 결과 폴더를 넘길 때는 이전 통합 실행의 번호 붙은 파일 또는 아래 표의 개별 실행기 파일을 사용한다.

| 시작 번호 | `--input`에 넣는 파일 또는 폴더의 내용 |
|---|---|
| 1 | `train.json` 또는 `train.en`·`train.ko`가 있는 폴더. 병렬 JSON은 파일 경로를 직접 줄 수 있다. |
| 2 | `01-aligned-terms.jsonl` 또는 완료된 정렬의 `term-pairs.jsonl` |
| 3 | `02-representative-candidates.jsonl` 또는 `3_complete.jsonl` |
| 4 | `03-domain-review-input.jsonl` 또는 `combined_for_domain_check.jsonl`. 예문이 들어 있는 기존 `combined.jsonl`·`combined_with_polysemy.jsonl`도 사용할 수 있다. |
| 5 | `04-target-domain.jsonl` 또는 `target_domain.jsonl` |
| 6 | `05-glossary.jsonl`과 `05-glossary-with-examples.jsonl`. 번호 없는 `glossary.jsonl`·`glossary-with-examples.jsonl`도 인식한다. |
| 7 | 6번의 용어집 두 파일과 `06-eval-source.en`·`06-eval-reference.ko`. 평가 문장 파일은 번호 없이도 인식한다. |
| 8 | `06-eval-source.en`·`06-eval-reference.ko`, `07-no-rag.ko.txt`·`07-rag-terms.ko.txt`·`07-rag-examples.ko.txt`, `05-glossary.jsonl`. `07-no-rag-plain.ko.txt`가 있으면 가져오고, 없으면 Plain 번역만 생성한다. 번호 없는 이름도 인식한다. `--eval-glossary`를 주면 폴더 내 용어집은 생략할 수 있다. |

1~4에서 시작하면 `--domain`이 필요하다. 1~6에서 시작하면 이후 평가에 사용할 `--eval-json` 또는 `--eval-en`·`--eval-ko`도 지정한다. 설정 파일에 이 값을 이미 적어 뒀다면 `--config`와 `--stage`·`--input`을 함께 사용할 수 있다.

예를 들어 도메인 검수 입력이 준비되어 있다면 04단계부터 실행한다.

```bash
CUDA_VISIBLE_DEVICES=1 python pipeline.py run \
  --stage 4 --input /home/hjs/PP/data/combined.jsonl \
  --eval-json /home/hjs/PP/data/validation/의학/간호학.json \
  --domain 의학/간호학 --load-in-4bit \
  --run-dir result/from-domain-review
```

이미 번역을 끝낸 통합 실행 폴더에서 평가만 새로 할 수도 있다.

```bash
CUDA_VISIBLE_DEVICES=1 python pipeline.py run \
  --stage 8 --input /경로/이전실행폴더 \
  --run-dir result/eval-again
```

예전 세 조건 결과가 들어 있는 기존 실행은 `run --run-dir /경로/이전실행폴더`로 이어가면 세 번역을 재사용하고 저장된 모델 설정으로 Plain만 더 생성한다. 예전 결과를 새 실행의 `--stage 8 --input`으로 가져올 때 Plain이 빠져 있다면, 기존 번역과 설정을 맞추기 위해 원래 사용한 `--gemma-model`, `--gemma-revision`, `--load-in-4bit` 등의 옵션을 함께 지정한다.

`run`은 시작 번호부터 평가까지 이어서 실행한다. 한 단계만 확인하려면 같은 옵션으로 `init`한 뒤 `stage 04-word-domain-check --run-dir ...`처럼 단계명을 지정한다. 중단 후에는 `run --run-dir ...`으로 이어가면 저장된 시작 번호와 가져온 파일을 재사용한다. 외부 입력 경로를 바꿀 때는 새 `--run-dir`을 지정한다.

## 새 데이터로 전체 실행

[config.example.json](config.example.json)을 복사해 용어집 구축 데이터, 별도 평가 데이터, 목표 분야를 바꾼다. 병렬 JSON은 객체 배열 또는 `data` 배열의 항목마다 `en`과 `ko`가 있어야 한다. TXT는 구축용에 `train_en`·`train_ko`, 평가용에 `eval_en`·`eval_ko`를 쓰고, 같은 행의 문장이 대응하도록 준비한다. 상대 경로는 설정 파일의 위치를 기준으로 해석한다.

```bash
CUDA_VISIBLE_DEVICES=1 python pipeline.py run --config /경로/config.json
```

등록된 분류 경로를 `domain`에 넣고, 자체 분류표를 쓰는 경우 `taxonomy_file`을 지정한다. 작은 표본은 `max_sentences`와 `eval_limit`으로 제한한다. 목표 분야의 채택 결과가 비면 번역 전에 멈춘다. `filter_glossary`로 평가 원문에서 특정 용어가 나온 문장만 선별할 수 있다. 여러 실험에서 같은 분모를 쓰려면 `eval_glossary`를 공통으로 지정한다. 이 값을 생략하면 이번 실행의 예문 없는 용어집을 두 RAG 비교에 공통 사용한다.

설정 파일을 만드는 대신 TXT 경로를 명령에 직접 지정할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=1 python pipeline.py run \
  --train-en /경로/용어추출.en --train-ko /경로/용어추출.ko \
  --eval-en /경로/평가.en --eval-ko /경로/평가.ko \
  --domain 의학/간호학 --load-in-4bit
```

## 버전 업데이트

현재 04단계는 **word-domain-check v1.9.4.1**을 사용한다. 개별 판정이 재시도 끝에도 검증되지 않으면 오류 행을 상세 실행 폴더의 `errors.jsonl`과 전체 판정 파일에 기록한다. 검증된 목표 분야 용어가 한 건 이상이면 이 용어로 용어집 구축을 계속한다. 상세 `summary.json`에는 `complete_with_errors`와 오류 건수가 남는다. 새 실행에서 오류가 한 건이라도 있으면 중단하려면 `--no-allow-record-errors`를 사용한다. JSON 설정 파일을 쓸 때는 `"allow_record_errors": false`로 지정한다.

이전 버전으로 04단계를 완료한 실행에 v1.9.4.1을 적용하려면 아래 명령으로 04단계를 다시 수행한다. 이후 `run --run-dir`로 용어집 구축부터 평가까지 이어서 실행한다.

```bash
CUDA_VISIBLE_DEVICES=1 python pipeline.py stage 04-word-domain-check \
  --run-dir /경로/실행폴더 --rerun
```

단계별 `UPSTREAM.md`에 원본 버전과 통합 변경을 적었다. 새 원본을 가져오면 앞 단계의 실제 출력으로 해당 단계가 실행되는지, 다음 단계가 결과를 읽는지 차례로 확인한다. 입력·출력 명세가 바뀌면 `pipeline.py`의 호출과 공개 파일명을 함께 갱신한다.
