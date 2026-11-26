# 한영 용어 정렬

- **가져온 버전** — `PP/word-alignment_v1.4`.
- **통합 변경** — `align.py`의 결과 루트를 `PP_STAGE_RESULT_ROOT`로 전달한다. `word_alignment/gemma_selection.py`에 `--gemma-model`을 추가했다. NLTK 로컬 자원과 정렬 설정을 함께 복사했다. 통합 실행기에 `train_json`을 지정하면 먼저 한영 파일로 저장한다.
- **입출력** — 행 순서가 대응하는 `.en`과 `.ko`를 받아 BGE 기본 정렬과 Gemma 선별을 수행한다. 최종 `term-pairs.jsonl`을 02단계에 넘긴다.
- **공용 BGE 모델** — 기본 BGE-M3 사용 시 중앙 실행기가 검증된 safetensors 캐시를 발견하면 로컬 모델 경로를 구성한다. 03단계도 이 경로를 사용한다. 명시한 모델 경로는 그대로 전달한다.
- **업데이트** — 결과 루트 연결과 Gemma 모델 인자를 다시 적용한다. `C_nltk_treebank` 결과의 원문·좌표·유형·기본형을 02단계에서 읽을 수 있는지 확인한다.
