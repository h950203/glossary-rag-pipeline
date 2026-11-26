# 한영 용어 후보 매핑 및 합성성 검사

파이프라인의 03단계다. `word-similar`의 `3_complete.jsonl`에 있는 한영 원문과 언어별 대표어 후보를 입력받는다. 같은 문장 안에서 BGE-M3로 한영 후보를 비교한다. 양쪽 모두 2어절 이상인 쌍을 먼저 선택하고 원문 구간을 확보한 뒤, 남은 구간에서 단일어와 비대칭 쌍을 매핑한다. 단일어 앵커별로 관측된 번역과 빈도를 집계한다.

복합어 검토는 기존 기준대로 진행한다. 관측 빈도 5회 이상인 단일어 번역으로 영어 어근이 80% 이상 설명되면 직역으로 분리한다. 남은 복합어는 하위 복합어와 단일어를 이용한 계층적 합성성을 75% 기준으로 검토하고, 과다 확장과 관용 수식어 결합도 확인한다. 단계별 산출물인 `compound_terms.jsonl`, `literal_compounds.jsonl`, `over_extensions.jsonl`, `rich_compounds.jsonl`을 보존한다.

합성성 검사 후 출력 루트에 **`combined_for_domain_check.jsonl`**을 자동 생성한다. 합성성 검사를 통과한 복합어와 자체 다의어 클러스터의 단일어·비대칭 대응쌍을 담는다. 같은 한영 쌍이 양방향 앵커에 있으면 빈도는 최댓값을 사용하고 출처 ID를 합친다. 서로 다른 번역쌍은 각각 보존한다. `aligned_bundles.jsonl`에서 한국어·영어 문장 ID가 모두 일치하는 예문을 찾아 `sample_sentences`에 넣는다. 원문 연결을 검증한 결과는 `word-domain-check`의 `--input-file`로 바로 전달한다. 합성성 검사에서 쓰는 빈도 5 기준은 직역 판단용이며, 도메인 검수 후보의 최소 빈도에는 적용하지 않는다.

프로젝트 루트에서 03단계를 실행한다.

```bash
CUDA_VISIBLE_DEVICES=1 python pipeline.py stage 03-word-candidates \
  --run-dir /경로/실행폴더
```

로컬 BGE-M3 모델을 쓰려면 실행 설정의 `bge_model`에 모델 디렉터리를 지정한다. 아래 단독 실행 명령에서는 `--bge_model /경로/BGE-M3`를 사용한다. `torch` 버전이 낮은 환경에서는 `model.safetensors`와 설정·토크나이저 파일을 함께 준비한 디렉터리를 사용한다.

기존 후보 매핑 결과를 재사용할 때는 단독 실행 명령의 `--output_dir`에 해당 폴더를 지정하고 `--compositionality_only`를 추가한다. 해당 폴더의 `compound_terms.jsonl`, `polysemy_clusters.jsonl`, `aligned_bundles.jsonl`을 사용해 합성성을 다시 검사하고 분류 입력을 만든다. 합성성 검사에 외부 사전을 사용해도, 도메인 검수에는 현재 입력에서 관측된 한영 쌍을 전달한다.

통합 실행에서 03단계가 끝나면 도메인 검수를 이어서 실행할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=1 python pipeline.py stage 04-word-domain-check \
  --run-dir /경로/실행폴더
```

## 단독 실행 방식과 옵션

본 패키지는 **3단계에서 추출된 외부 후보군 파일(`candidates.jsonl` 또는 `candidates.json`)**을 입력받아, 파이프라인의 후보군 재생성(Kiwi/NLTK)을 건너뛰고 **Step 4 (BGE-M3 밀집 임베딩 매핑 ~ 계층적 합성성 검사)**까지 원스톱으로 수행하기 위한 실행 스크립트 및 아키텍처 문서 모음입니다.

---

## 1. 스크립트 및 문서 구성

```text
03-word-candidates/
├── README.md                                   # Step 4 실행 가이드 및 옵션 설명서 (본 문서)
├── ARCHITECTURE.md                             # 경량화 파이프라인 전체 아키텍처 설계서
├── models.py                                   # 공통 데이터 클래스 (CandidateBundle, RawTermPair 등)
├── common/
│   ├── __init__.py
│   └── text_utils.py                           # 한글 검사, 구두점 정제, 약어 토큰 정규식 유틸리티
├── pipeline_lightweight/
│   ├── __init__.py
│   ├── term_mapper.py                          # [4단계] BGE-M3 임베딩 인코딩, MNN 필터링, 2단계 스팬 마스킹
│   ├── cluster_builder.py                      # [4-1 / 4-2] 1단어 다의어 클러스터, 대칭 복합어, 약어 사전 및 리포트 빌더
│   └── compositionality_filter.py              # [4-2 합성성 검사] 1~3단계 직역/계층/수식어 필터링 엔진
└── co-work/
    ├── run_mapping.py                          # [메인 엔트리포인트] Step 4 통합 실행 및 다중 사전 합성성 검사 러너
    └── candidates_template.jsonl               # 입력 후보군 양식 샘플 템플릿
```

---

## 2. Step 4 세부 처리 흐름

1. **Step 4 — BGE-M3 임베딩 기반 후보군 매핑 (`term_mapper.py`)**
   - 입력 파일의 `ko_candidates`, `en_candidates` 고유 어휘 풀을 `BAAI/bge-m3`로 일괄 인코딩합니다.
   - **Stage 1 (대칭 복합어 우선 선점)**: 양쪽 모두 2어절 이상(`words >= 2`)인 후보군끼리 상호 최근접 이웃(MNN, 유사도 $\ge 0.60$, 복합어 부스트 $+0.04$) 매칭 후 문장 내 글자 스팬을 마스킹합니다.
   - **Stage 2 (잔여 1단어/비대칭 매핑)**: 마스킹되지 않은 잔여 구간에서 1단어 및 비대칭 후보군 간 MNN($\ge 0.60$) 매핑을 수행합니다.
2. **Step 4-1 — 1단어 용어사전 및 1:N 다의어 클러스터 (`cluster_builder.py` $\rightarrow$ `polysemy_clusters.jsonl`)**
   - 단일어 중심 1:N 번역 관계를 KO/EN 양방향 앵커로 통합 집계합니다.
   - **4-1-1 규칙**: A언어는 1단어이고 B언어는 2단어 이상인 경우, 1단어인 A언어 앵커의 번역어로 등록합니다.
   - **약어 정제 (`acronym_terms.jsonl`)**: 한·영 양쪽에 공통 등장하는 대문자 약어를 추출하되, 불리언 검색 연산자(`AND`, `OR`, `NOT`)와 익명화 마스킹(`AAA`, `BBB`, `FFF`, `OOO`, `OO` 등)은 자동 제거합니다.
3. **Step 4-2 — 복합어 용어사전 및 계층적 합성성 검사 (`compositionality_filter.py`)**
   - 양쪽 2단어 이상인 1차 대칭 복합어(`compound_terms.jsonl`)를 기준 1단어 다의어 사전(`polysemy_clusters.jsonl`)과 대조하여 순차 필터링합니다:
     - **1단계 (`base_1word_literal`)**: 구성 단어를 쪼개어 1단어 다의어 사전만으로 영어 어근의 80% 이상이 직역되면 일반 결합어로 분리
     - **2단계 (`hierarchical_literal` / `over_extension`)**: 3단어 이상 복합어에서 하위 2~4단어 복합어 앵커 + 1단어 조합(75% 이상) 여부 및 과다 확장 격리
     - **3단계 (`modifier_*`)**: 일반/도메인 관용 수식어(58종) 결합 직역어 분리
   - 위 1~3단계 직역 생성이 불가능하여 끝까지 살아남은 용어만 최종 도메인 복합어 사전(**`rich_compounds.jsonl`**)으로 저장합니다.

---

## 3. 실행 방법 (CLI)

`03-word-candidates/`에서 실행한다.

```bash
# 1) 기본 실행 (Step 4 전체 매핑 + 자체 다의어 사전 기반 합성성 검사)
python co-work/run_mapping.py \
    --candidates_file /경로/3_complete.jsonl \
    --output_dir /경로/후보-매핑-결과 \
    --batch_size 32 \
    --comp_out_dir compositionality_self

# 2) 외부(일반 도메인 등) 다의어 사전을 기준 사전으로 사용하여 합성성 검사만 실행
python co-work/run_mapping.py \
    --compositionality_only \
    --output_dir /경로/후보-매핑-결과 \
    --external_polysemy_path path/to/general_polysemy_clusters.jsonl \
    --external_acronym_path path/to/general_acronym_terms.jsonl \
    --comp_out_dir compositionality_external

# 3) 자체 다의어 사전 + 외부 다의어 사전을 병합(빈도 합산)하여 합성성 검사 실행
python co-work/run_mapping.py \
    --compositionality_only \
    --output_dir /경로/후보-매핑-결과 \
    --external_polysemy_path path/to/general_polysemy_clusters.jsonl \
    --external_acronym_path path/to/general_acronym_terms.jsonl \
    --merge_external_polysemy \
    --comp_out_dir compositionality_merged
```

### 주요 CLI 옵션
- `--candidates_file`: 입력 후보군 파일 경로 (`.jsonl` 및 `.json` 배열 포맷 모두 자동 인식)
- `--output_dir`: 결과물 저장 루트 폴더 (기본값: `co-work/output`)
- `--comp_out_dir`: 합성성 검사 결과 저장 하위 폴더명 (기본값: `compositionality`)
- `--compositionality_only`: 이미 생성된 `compound_terms.jsonl`을 대상으로 4-2 합성성 검사만 단독 실행 (약 3~5초 소요)
- `--external_polysemy_path` / `--external_acronym_path`: 합성성 검사에 사용할 외부 기준 다의어/약어 사전 경로
- `--merge_external_polysemy`: 자체 사전과 외부 사전을 병합(빈도 합산)하여 기준 사전으로 사용
- `--min_trans_freq`: 합성성 검사용 1단어 사전 최소 번역 빈도 임계치 (기본값: `5`)
