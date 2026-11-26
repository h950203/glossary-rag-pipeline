# word-similar v1.3

한영 용어 정렬 결과에 `covered_expansions_small_sample`을 적용해 대표어 후보와 확장 표현을 만든다. 그룹화 코드와 규칙을 이 폴더에 포함했다. 독립 실행에는 Python 3.11 이상과 정렬 결과 파일이 필요하다.

```bash
python /home/hjs/PP/glossary-rag-pipeline/pipeline.py stage 02-word-similar \
  --run-dir /경로/실행폴더
```

통합 실행에서는 같은 실행의 최종 정렬 파일을 읽고 공용 결과 폴더에 저장한다. 독립 실행의 기본값은 선택한 정렬 방식의 전체 문장이며, `experiment.py --input`에 정렬 파일을 지정할 수 있다.

결과 폴더의 `groups-simple.csv`, `group-pairs.csv`, `3_complete.jsonl`과 `details/`는 v1.2와 같은 형식이다. 문장별 미배정까지 Codex에서 검토하려면 `--batch-size 25`로 `details/codex/`에 검토 자료를 저장할 수 있다.

[통합 실행 방법](../README.md) · [가져온 버전과 업데이트 방법](UPSTREAM.md)
