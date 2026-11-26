# 한영 용어 후보 매핑 및 합성성 검사

- **가져온 버전** — `PP/word-candidates_v1.1`.
- **통합 변경** — 중앙 실행기가 `--candidates_file`, `--output_dir`, `--bge_model`을 직접 지정한다.
- **입출력** — `3_complete.jsonl`의 언어별 후보를 BGE-M3로 다시 매핑한다. 합성성 검사를 통과한 복합어와 단일어·비대칭 대응을 예문과 함께 `combined_for_domain_check.jsonl`로 내보낸다.
- **업데이트** — 직역·과다 확장 분리, 양방향 중복 제거 및 실제 예문 연결이 유지되는지 확인한다. 최종 파일의 `surface_a`, `surface_b`, `sample_sentences[].ko_text/en_text`를 04단계 입력으로 확인한다.
