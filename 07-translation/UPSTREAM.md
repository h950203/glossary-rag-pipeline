# No-RAG와 RAG 번역

- **가져온 버전** — `nmt/translate_rag_v1.0.py`, `nmt/translate_rag_v1.1.py`.
- **통합 변경** — CoT No-RAG와 두 RAG 조건은 v1.1과 `--fixed-prompt`를 사용한다. Plain은 짧은 번역 지시를 사용하고 v1.1의 모델 로딩과 생성 설정을 공유한다. 네 조건에 같은 리비전, 양자화, 배치를 적용하며 Plain을 직접 실행할 때도 `--load-in-4bit`와 `--revision`을 지정할 수 있다.
- **입출력** — 같은 영어 파일을 CoT No-RAG, 용어 RAG, 용어+예문 RAG, Plain No-RAG로 번역한다. `.txt`는 평가에, 같은 위치의 실행 `.csv`는 검색 용어와 응답 검증에 사용한다. Plain의 `term_hits`에는 빈 배열이 들어간다.
- **업데이트** — 원문 행 수와 순서, 동일한 모델 로딩 및 생성 설정, CoT와 Plain 출력 형식, 예문 개수와 실패 표기 처리를 확인한다. 이전 실행 폴더에서는 기존 번역 세 조건의 재사용도 확인한다.
