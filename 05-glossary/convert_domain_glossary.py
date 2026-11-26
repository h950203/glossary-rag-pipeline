#!/usr/bin/env python3
"""Convert word-domain-check JSONL to translate_rag_v1.0.py glossary JSONL."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unicodedata


def convert(input_path: Path, output_path: Path) -> None:
    entries: dict[str, dict[str, dict[str, str]]] = {}
    accepted = 0
    with input_path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("status") not in (
                    "accept", "reject", "pending", "error", "unprocessed",
                ):
                    raise ValueError("word-domain-check 판정 행이 필요합니다")
                if row["status"] != "accept":
                    continue
                terms = row.get("terms")
                if not isinstance(terms, dict) or not all(
                    isinstance(terms.get(lang), str) and terms[lang].strip() for lang in ("en", "ko")
                ):
                    raise ValueError("채택된 행의 terms.en과 terms.ko가 필요합니다")
            except ValueError as error:
                raise ValueError(f"{input_path}:{number}: {error}") from error

            en, ko = (" ".join(unicodedata.normalize("NFC", terms[lang]).split()) for lang in ("en", "ko"))
            entry = {"en": en, "ko": ko}
            reason = row.get("reason")
            if isinstance(reason, str) and reason.strip():
                entry["reason"] = reason
            # Match the English key normalization in translate_rag_v1.0.py.
            entries.setdefault(en.casefold(), {}).setdefault(ko, entry)
            accepted += 1

    unique = []
    conflicts = {}
    for key, translations in entries.items():
        if len(translations) == 1:
            unique.append(next(iter(translations.values())))
        else:
            conflicts[key] = sorted(translations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as directory:
        staged = Path(directory) / "glossary.jsonl"
        staged.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unique), encoding="utf-8")
        output_path.hardlink_to(staged)

    for key, translations in conflicts.items():
        print(f"충돌 제외 · {key}: {' / '.join(translations)}", file=sys.stderr)
    duplicates = accepted - sum(len(translations) for translations in entries.values())
    print(f"저장 {output_path} · 채택 입력 {accepted}행 · 중복 {duplicates}행 · "
          f"충돌 표제어 {len(conflicts)}개 · 최종 {len(unique)}개")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True, help="예: target_domain.jsonl")
    parser.add_argument("--output-file", type=Path, required=True, help="새 용어사전 JSONL 경로")
    args = parser.parse_args()
    try:
        convert(args.input_file, args.output_file)
    except (OSError, ValueError) as error:
        parser.exit(1, f"오류: {error}\n")


if __name__ == "__main__":
    main()
