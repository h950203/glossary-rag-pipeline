#!/usr/bin/env python3
"""Convert JSONL to the translate_rag_v1.1 glossary with usage examples."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unicodedata


def field(record, path: str):
    if path == ".":
        return record
    for key in path.split("."):
        if not isinstance(record, dict):
            return None
        record = record.get(key)
    return record


def clean(value) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split()) if isinstance(value, str) else ""


def read_examples(record: dict, args: argparse.Namespace):
    values = field(record, args.examples_field)
    if values is None:
        return
    if not isinstance(values, list):
        values = [values]
    for sample in values:
        en = clean(field(sample, args.example_en_field))
        ko = clean(field(sample, args.example_ko_field))
        if en:
            yield {"en": en, "ko": ko} if ko else {"en": en}


def convert(args: argparse.Namespace) -> None:
    entries: dict[str, dict[str, dict]] = {}
    accepted = 0
    with args.input_file.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSON 객체가 필요합니다")
                if not args.no_status_filter:
                    status = field(row, args.status_field)
                    if not isinstance(status, str):
                        raise ValueError(f"판정 필드 {args.status_field}가 필요합니다. 판정 없는 입력은 --no-status-filter를 사용하세요")
                    if status != args.accepted_value:
                        continue
                en = clean(field(row, args.en_field))
                ko = clean(field(row, args.ko_field))
                if not en or not ko:
                    raise ValueError(f"{args.en_field}와 {args.ko_field}에 용어 문자열이 필요합니다")
            except ValueError as error:
                raise ValueError(f"{args.input_file}:{number}: {error}") from error

            entry = entries.setdefault(en.casefold(), {}).setdefault(ko, {"en": en, "ko": ko})
            reason = field(row, args.reason_field)
            if isinstance(reason, str) and reason.strip():
                entry.setdefault("reason", reason)
            for example in read_examples(row, args):
                examples = entry.setdefault("examples", [])
                if example not in examples:
                    examples.append(example)
            accepted += 1

    unique = []
    conflicts = {}
    for key, translations in entries.items():
        if len(translations) != 1:
            conflicts[key] = sorted(translations)
            continue
        entry = next(iter(translations.values()))
        if examples := entry.get("examples"):
            # Prefer examples that visibly contain the recorded surfaces on both sides.
            examples.sort(key=lambda ex: -sum(entry[lang].casefold() in ex.get(lang, "").casefold()
                                              for lang in ("en", "ko")))
            if args.max_examples_per_term is not None:
                del examples[args.max_examples_per_term:]
                if not examples:
                    del entry["examples"]
        unique.append(entry)

    output = args.output_file
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "glossary.jsonl"
        with staged.open("w", encoding="utf-8") as handle:
            for row in unique:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.hardlink_to(staged)

    for key, translations in conflicts.items():
        print(f"충돌 제외 · {key}: {' / '.join(translations)}", file=sys.stderr)
    with_examples = sum(bool(row.get("examples")) for row in unique)
    print(f"저장 {output} · 채택 입력 {accepted}행 · 충돌 표제어 {len(conflicts)}개 · "
          f"최종 {len(unique)}개 · 예문 포함 {with_examples}개")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, epilog="필드 경로는 점으로 연결합니다. '.'은 현재 객체나 문자열입니다.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--en-field", default="terms.en", help="영어 용어 필드 (기본: terms.en)")
    parser.add_argument("--ko-field", default="terms.ko", help="한국어 용어 필드 (기본: terms.ko)")
    parser.add_argument("--examples-field", default="original.sample_sentences", help="예문 배열 또는 단일 예문 필드")
    parser.add_argument("--example-en-field", default="en_text", help="개별 예문의 영어 문장 필드 (기본: en_text)")
    parser.add_argument("--example-ko-field", default="ko_text", help="개별 예문의 한국어 문장 필드 (기본: ko_text)")
    parser.add_argument("--max-examples-per-term", type=int,
                        help="용어당 저장할 최대 예문 수 (기본: 전체, 0: 예문 생략)")
    parser.add_argument("--reason-field", default="reason", help="근거 필드 (기본: reason)")
    parser.add_argument("--status-field", default="status", help="판정 필드 (기본: status)")
    parser.add_argument("--accepted-value", default="accept", help="포함할 판정 값 (기본: accept)")
    parser.add_argument("--no-status-filter", action="store_true", help="판정 필드가 없는 입력을 처리")
    args = parser.parse_args()
    if args.max_examples_per_term is not None and args.max_examples_per_term < 0:
        parser.error("--max-examples-per-term은 0 이상이어야 합니다")
    try:
        convert(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"오류: {error}\n")


if __name__ == "__main__":
    main()
