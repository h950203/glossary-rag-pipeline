#!/usr/bin/env python3
"""Export sentence candidates actually observed as aligned representative terms.

Standard library only. No re-tokenization, substring search, or member-to-core
substitution: identity is (language, unit_type, normalized token sequence).
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
LANGUAGES = ("ko", "en")
KINDS = {"nominal", "verbal"}


def load_representatives(path):
    representatives = {}
    with path.open(encoding="utf-8-sig", newline="") as reader:
        table = csv.DictReader(reader)
        required = {"language", "unit_type", "representative", "normalized"}
        if not required.issubset(table.fieldnames or []):
            raise ValueError(f"대표어 CSV에 필요한 열: {sorted(required)}")
        for number, row in enumerate(table, 2):
            lang, kind = row["language"], row["unit_type"]
            term, normalized = row["representative"], row["normalized"]
            if lang not in LANGUAGES or kind not in KINDS or not term.strip() or not normalized.strip():
                raise ValueError(f"대표어 CSV {number}행: 언어·유형·대표어·정규화 값을 확인하세요")
            key = (lang, kind, normalized)
            if key in representatives and representatives[key] != term:
                raise ValueError(f"대표어 CSV {number}행: 같은 정규화 값에 다른 대표어가 있습니다")
            representatives[key] = term
    return representatives


def convert_row(row, representatives, candidate_form, include_occurrences):
    if not isinstance(row, dict) or row.get("schema") != "term-pairs-v1.1":
        raise ValueError("word-alignment_v1.1의 term-pairs.jsonl을 입력하세요")
    if type(row.get("line_id")) is not int or row["line_id"] < 1:
        raise ValueError("양의 정수 line_id가 필요합니다")
    if not all(isinstance(row.get(lang), str) for lang in LANGUAGES):
        raise ValueError("한국어·영어 원문 전체가 필요합니다")
    if not isinstance(row.get("pairs"), list):
        raise ValueError("pairs 배열이 필요합니다")
    candidates = {lang: {} for lang in LANGUAGES}
    occurrences = {lang: [] for lang in LANGUAGES}
    seen_occurrences = {lang: set() for lang in LANGUAGES}
    matched = set()
    for pair in row["pairs"]:
        if not isinstance(pair, dict) or pair.get("unit_type") not in KINDS:
            raise ValueError("유형이 명시된 용어쌍이 필요합니다")
        kind = pair["unit_type"]
        for lang in LANGUAGES:
            unit = pair.get(lang + "_unit")
            if not isinstance(unit, dict) or unit.get("type") != kind or unit.get("shape_valid") is not True:
                raise ValueError(f"{lang}: 유효한 상위 용어 분석 정보가 필요합니다")
            tokens = unit.get("tokens")
            if (not isinstance(tokens, list) or not tokens
                    or any(not isinstance(t, str) or not t.strip() for t in tokens)):
                raise ValueError(f"{lang}: 정규화 토큰 배열이 필요합니다")
            normalized = " ".join(tokens)
            if unit.get("normalized") != normalized:
                raise ValueError(f"{lang}: 정규화 토큰과 normalized가 다릅니다")
            span = unit.get("span")
            if (not isinstance(span, list) or len(span) != 2
                    or any(type(x) is not int for x in span)
                    or not 0 <= span[0] < span[1] <= len(row[lang])):
                raise ValueError(f"{lang}: 원문 좌표가 잘못되었습니다")
            surface = row[lang][span[0]:span[1]]
            if (surface != pair.get(lang) or surface != unit.get("surface")
                    or pair.get(lang + "_span") != span):
                raise ValueError(f"{lang}: 좌표와 원문 표현이 다릅니다")
            if (not isinstance(unit.get("canonical"), str) or not unit["canonical"].strip()
                    or pair.get(lang + "_canonical") != unit["canonical"]):
                raise ValueError(f"{lang}: 기본형 정보가 다릅니다")
            key = (lang, kind, normalized)
            if key not in representatives:
                continue
            matched.add(key)
            representative = representatives[key]
            text = representative if candidate_form == "representative" else surface
            candidates[lang].setdefault(text, None)
            identity = (kind, tuple(span), representative)
            if include_occurrences and identity not in seen_occurrences[lang]:
                seen_occurrences[lang].add(identity)
                occurrences[lang].append({"representative": representative, "surface": surface,
                                          "span": span, "unit_type": kind})
    result = {"line_id": row["line_id"], "ko_sentence": row["ko"], "en_sentence": row["en"],
              "ko_candidates": list(candidates["ko"]), "en_candidates": list(candidates["en"])}
    if include_occurrences:
        result["candidate_occurrences"] = occurrences
    return result, matched


def parser():
    p = argparse.ArgumentParser(description="대표어와 실제 정렬된 출현 문장을 연결합니다(GPU·LLM 불필요).")
    p.add_argument("--groups-csv", type=Path, required=True)
    p.add_argument("--alignments", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--format", choices=("jsonl", "json"), help="기본: 출력 확장자가 .json이면 JSON 배열, 그 외 JSONL")
    p.add_argument("--candidate-form", choices=("representative", "surface"), default="representative")
    p.add_argument("--include-occurrences", action="store_true", help="실제 원문 표기와 문자 좌표 추가")
    p.add_argument("--require-both", action="store_true", help="양쪽 언어에 대표어가 있는 문장만 저장")
    p.add_argument("--overwrite", action="store_true")
    return p


def run(args):
    started = time.monotonic()
    source, groups, output = (p.expanduser().resolve() for p in (args.alignments, args.groups_csv, args.output))
    configured_summary = getattr(args, "summary_path", None)
    summary_path = (Path(configured_summary).expanduser().resolve() if configured_summary
                    else output.with_name(output.stem + "-summary.json"))
    if {source, groups} & {output, summary_path}:
        raise ValueError("출력 파일은 입력 파일과 달라야 합니다")
    if not args.overwrite and (output.exists() or summary_path.exists()):
        raise ValueError("기존 결과가 있습니다. --overwrite 또는 다른 --output을 사용하세요")
    representatives = load_representatives(groups)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = args.format or ("json" if output.suffix.lower() == ".json" else "jsonl")
    counts = Counter()
    used = set()
    identities = set()
    checksum = hashlib.sha256()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix="." + output.name + ".", delete=False) as writer:
            temp_path = Path(writer.name)
            if fmt == "json":
                writer.write("[\n")
            with source.open("rb") as reader:
                for number, raw in enumerate(reader, 1):
                    checksum.update(raw)
                    if not raw.strip():
                        continue
                    try:
                        row = json.loads(raw.decode("utf-8-sig"))
                        if getattr(args, "source_format", "term-pairs") == "grouping_sources":
                            row = {"schema": row["schema"], "line_id": row["line_id"], "method": row["method"],
                                   "ko": row["ko_sentence"], "en": row["en_sentence"], "pairs": row["alignment_pairs"]}
                        record, matched = convert_row(row, representatives, args.candidate_form, args.include_occurrences)
                        identity = (row.get("method", "unknown"), row["line_id"])
                        if identity in identities:
                            raise ValueError("같은 method/line_id가 중복되었습니다")
                        identities.add(identity)
                    except (ValueError, TypeError, KeyError) as exc:
                        raise ValueError(f"정렬 입력 {number}행: {exc}") from exc
                    counts["input_sentences"] += 1
                    used.update(matched)
                    ko, en = record["ko_candidates"], record["en_candidates"]
                    if not ko and not en:
                        counts["skipped_no_representatives"] += 1
                        continue
                    if args.require_both and (not ko or not en):
                        counts["skipped_one_sided"] += 1
                        continue
                    if fmt == "json" and counts["output_sentences"]:
                        writer.write(",\n")
                    writer.write(json.dumps(record, ensure_ascii=False))
                    if fmt == "jsonl":
                        writer.write("\n")
                    counts["output_sentences"] += 1
                    counts["ko_candidate_entries"] += len(ko)
                    counts["en_candidate_entries"] += len(en)
                    counts["both_languages" if ko and en else "one_language"] += 1
            if fmt == "json":
                writer.write("\n]\n")
        if not counts["input_sentences"]:
            raise ValueError("정렬 입력 파일에 문장 행이 없습니다")
        os.replace(temp_path, output)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    summary = {"groups_csv": str(groups), "alignments": str(source), "output": str(output),
               "groups_sha256": hashlib.sha256(groups.read_bytes()).hexdigest(),
               "alignments_sha256": checksum.hexdigest(), "format": fmt,
               "candidate_form": args.candidate_form, "require_both": args.require_both,
               "source_format": getattr(args, "source_format", "term-pairs"),
               "include_occurrences": args.include_occurrences,
               "matching": "exact_language_unit_type_normalized_in_aligned_pairs",
               "representatives": len(representatives), "matched_representatives": len(used),
               "unmatched_representatives": len(representatives) - len(used),
               "counts": dict(counts), "elapsed_seconds": round(time.monotonic() - started, 3)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"저장: {output}; {counts['output_sentences']:,}문장; {summary['elapsed_seconds']:.2f}초")
    print(f"집계: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run(parser().parse_args()))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("중단됨. 미완성 임시 결과는 제거했습니다.", file=sys.stderr)
        sys.exit(130)
