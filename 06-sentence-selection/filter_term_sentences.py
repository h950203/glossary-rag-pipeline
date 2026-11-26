#!/usr/bin/env python3
"""Save English lines containing a glossary term and their optional Korean pairs."""
from __future__ import annotations

import argparse
import json
import unicodedata
from itertools import islice, zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def load_index(path: Path, limit: int | None):
    import ahocorasick

    index = ahocorasick.Automaton()
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(islice(source, limit), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: 올바르지 않은 JSON입니다.") from exc
            term = record.get("en") if isinstance(record, dict) else None
            if not isinstance(term, str) or not term.strip():
                raise ValueError(f"{path}:{number}: 비어 있지 않은 en 문자열이 필요합니다.")
            key = normalize(term)
            index.add_word(key, len(key))
    if not len(index):
        raise ValueError(f"{path}: 사용할 영어 용어가 없습니다.")
    index.make_automaton()
    return index


def is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def contains_term(line: str, index) -> bool:
    text = normalize(line)
    for end, length in index.iter(text):
        start = end + 1 - length
        stop = end + 1
        if start > 0 and is_word_char(text[start - 1]):
            continue
        if stop < len(text) and is_word_char(text[stop]):
            continue
        return True
    return False


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수를 입력하세요.")
    return number


def filter_parallel(input_en: Path, input_ko: Path, output_en: Path, output_ko: Path,
                    index, limit: int | None) -> tuple[int, int]:
    for path in (output_en, output_ko):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"출력 파일이 이미 있습니다: {path}")

    scanned = matched = 0
    with TemporaryDirectory(dir=output_en.parent) as en_dir, TemporaryDirectory(dir=output_ko.parent) as ko_dir:
        staged_en, staged_ko = Path(en_dir) / "matched.en", Path(ko_dir) / "matched.ko"
        with (input_en.open(encoding="utf-8-sig", newline="") as source_en,
              input_ko.open(encoding="utf-8-sig", newline="") as source_ko,
              staged_en.open("w", encoding="utf-8", newline="") as target_en,
              staged_ko.open("w", encoding="utf-8", newline="") as target_ko):
            pairs = zip_longest(islice(source_en, limit), islice(source_ko, limit))
            for scanned, (en, ko) in enumerate(pairs, start=1):
                if en is None or ko is None:
                    shorter = input_en if en is None else input_ko
                    raise ValueError(f"한영 입력의 줄 수가 다릅니다. {shorter}에 {scanned:,}번째 줄이 없습니다.")
                if contains_term(en, index):
                    target_en.write(en)
                    target_ko.write(ko)
                    matched += 1
        output_en.hardlink_to(staged_en)
        try:
            output_ko.hardlink_to(staged_ko)
        except BaseException:
            output_en.unlink()
            raise
    return scanned, matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glossary", type=Path, required=True, help="en 필드가 있는 용어집 JSONL 경로")
    parser.add_argument("--input", "--data", "--input-en", type=Path, required=True, help="한 줄에 한 문장이 있는 영어 데이터 경로")
    parser.add_argument("--input-ko", type=Path, help="영어와 행 순서가 대응하는 한국어 데이터 경로")
    parser.add_argument("--output", "--output-en", type=Path, help="기본값은 입력 파일 옆의 <이름>-tech-matched.en")
    parser.add_argument("--output-ko", type=Path, help="기본값은 영어 출력 경로의 확장자를 .ko로 바꾼 경로")
    parser.add_argument("--glossary-lines", type=positive_int, help="용어집을 앞 N줄만 읽습니다.")
    parser.add_argument("--input-lines", type=positive_int, help="입력을 앞 N줄만 읽습니다.")
    args = parser.parse_args()
    output = args.output or args.input.with_name(f"{args.input.stem}-tech-matched.en")
    if args.output_ko is not None and args.input_ko is None:
        parser.error("--output-ko를 지정하려면 --input-ko가 필요합니다.")
    output_ko = (args.output_ko or output.with_suffix(".ko")) if args.input_ko is not None else None

    try:
        if output_ko is not None and output.resolve() == output_ko.resolve():
            raise ValueError("영어와 한국어 출력 경로는 서로 달라야 합니다.")
        index = load_index(args.glossary, args.glossary_lines)
        if output_ko is not None:
            scanned, matched = filter_parallel(args.input, args.input_ko, output, output_ko, index, args.input_lines)
        else:
            scanned = matched = 0
            with args.input.open(encoding="utf-8-sig", newline="") as source, \
                    output.open("x", encoding="utf-8", newline="") as target:
                for scanned, line in enumerate(islice(source, args.input_lines), start=1):
                    if contains_term(line, index):
                        target.write(line)
                        matched += 1
    except ModuleNotFoundError as exc:
        if exc.name != "ahocorasick":
            raise
        parser.exit(2, "pyahocorasick가 필요합니다. python -m pip install pyahocorasick\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"오류: {exc}\n")

    destinations = f"{output} · {output_ko}" if output_ko is not None else str(output)
    print(f"영어 용어 {len(index):,}개 · 입력 {scanned:,}줄 · 저장 {matched:,}줄 → {destinations}")


if __name__ == "__main__":
    main()
