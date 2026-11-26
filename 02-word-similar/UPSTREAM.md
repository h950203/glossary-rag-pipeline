# 대표어·확장 표현 그룹화

- **가져온 버전** — `PP/word-similar_v1.3`, `covered_expansions_small_sample`.
- **통합 변경** — `experiment.py`에서 `PP_STAGE_RESULT_ROOT`를 받고, 독립 실행 시 정렬 검색 경로를 번호가 붙은 형제 폴더로 맞췄다. 중앙 실행기는 `--input`을 직접 지정한다.
- **입출력** — `term-pairs.jsonl`을 읽고 `groups-simple.csv`, `group-pairs.csv`, `3_complete.jsonl`을 쓴다. 03단계에는 언어별 후보 배열과 양쪽 원문이 있는 `3_complete.jsonl`을 전달한다.
- **업데이트** — 그룹화 규칙과 외부 출력 형식을 함께 확인한다. 03단계의 매핑 실행기가 `line_id`, `ko_sentence`, `en_sentence`, `ko_candidates`, `en_candidates`를 읽는지 확인한다.
