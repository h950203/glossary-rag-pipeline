# 번역 비교 평가

- **가져온 파일** — `nmt/lang_eval_rag_compare.py`와 로컬 의존 파일 `lang_eval_v3.2-args.py`, `lang_eval_v3.1-args.py`, `lang_eval_v3.1.py`.
- **통합 변경** — `lang_eval_rag_with_plain.py`가 공통 원문·정답·용어집으로 네 조건의 COMET을 한 번에 평가한다. 두 쌍별 CSV와 4조건 CSV가 같은 CoT No-RAG 점수를 공유한다. 설정에 `eval_python`을 지정하면 평가에 별도 Python 환경을 사용한다.
- **입출력** — 영어 원문, 한국어 정답, 네 한국어 번역, 평가 용어집을 읽는다. CSV의 `summary`에 용어 인식률, `wmt22-comet-da` 평균, 코퍼스 BLEU, ROUGE-1·2·L 문장 평균과 변화량을 저장한다.
- **업데이트** — 같은 용어집으로 분모가 동일한지, 네 조건과 두 쌍별 CSV의 기준 COMET이 같은지, 이전 평가 CSV로 Plain만 평가하는 경로가 동작하는지 확인한다.

## 평가 기준

- 문장별 인식률은 `terms_json`에서 조건별 `no_rag_matched_ko` 또는 `rag_matched_ko`의 항목 수를 더한 뒤 전체 `occurrences`로 나눈다. 전체 인식률은 모든 문장의 일치 횟수와 출현 횟수를 각각 합산한다. 원문에 평가 용어가 없으면 인식률은 빈칸으로 둔다.
- BLEU와 ROUGE는 한국어 정답을 참조한다. NFC로 정규화한 뒤 유니코드 단어와 문장부호로 토큰을 나눈다. BLEU는 SacreBLEU로 0~100 점수를 계산하며 요약 행에는 코퍼스 BLEU를 기록한다. ROUGE-1·2·L은 0~100 범위의 F1을 문장별로 계산하고 요약 행에는 평균을 기록한다.
- 기존 평가 CSV에서 인식률과 BLEU·ROUGE를 추가할 때는 `--input-csv`와 새 `--output-file`을 지정한다. 이 경우 CSV에 저장된 COMET 값을 재사용한다.

```bash
python 08-evaluation/lang_eval_rag_compare.py \
  --input-csv /경로/기존-평가.csv \
  --output-file /경로/평가-BLEU-ROUGE.csv
```

## 저장된 두 평가 CSV에 Plain 추가

`lang_eval_rag_with_plain.py`에 기존 두 평가 CSV, 동일한 평가 용어집, 일반 지시로 번역한 전체 TXT를 전달한다. 두 CSV의 원문, 정답, 기존 No-RAG 결과와 용어 일치 건수를 검증한다. Plain TXT와 이름이 같은 번역 로그 CSV가 있으면 원문과 행 순서도 검증한다.

```bash
python 08-evaluation/lang_eval_rag_with_plain.py \
  --terms-eval-csv /경로/용어-RAG-평가.csv \
  --examples-eval-csv /경로/예문-RAG-평가.csv \
  --plain-file /경로/no-rag-plain.txt \
  --glossary /경로/공통-평가-용어집.jsonl \
  --output-file /경로/네-조건-평가.csv \
  --device cuda
```

기존 세 조건의 문장별 COMET을 재사용하고 Plain 조건의 COMET을 계산한다. 로컬 체크포인트를 지정할 때는 기존 평가와 같은 모델을 사용한다. 네 조건 모두 같은 기준으로 용어 인식률, 코퍼스 BLEU와 ROUGE-1·2·L을 계산한다. 결과 CSV의 처음 네 `summary` 행에서 조건별 점수 및 기존 CoT No-RAG 대비 변화량을 확인하고, `sentence` 행에서 같은 원문에 대한 번역 네 개와 용어 매칭 근거를 확인한다.
