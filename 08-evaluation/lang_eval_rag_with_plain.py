#!/usr/bin/env python3
"""Compare four translation conditions from text files or existing paired evaluations."""

import argparse
import csv
from itertools import zip_longest
import json
import math
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

import lang_eval_rag_compare as evaluation


CONDITIONS = ("no_rag_cot", "rag_terms", "rag_examples", "no_rag_plain")
METRICS = ("comet", *evaluation.LEXICAL_METRICS)
CSV_FIELDS = (
    "row_type", "condition", "line", "en", "reference_ko", "sentences", "sentences_with_terms",
    "glossary_entries", "en_term_occurrences", "matched_occurrences", "term_rate_pct", "term_rate_delta_pp",
    "target_match", "comet_model", "comet_source", "bleu_signature", "lexical_tokenizer",
    *METRICS, *(metric + "_delta" for metric in METRICS), "terms_json",
    *(f"{condition}_{field}" for condition in CONDITIONS
      for field in ("ko", "matched_occurrences", "term_rate_pct", *METRICS)),
)


def glossary_hits(details):
    return [(item["en"], tuple(item["expected_ko"]), item["occurrences"]) for item in details]


def saved_score(row, field, line):
    try:
        score = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{line}행에 저장된 {field} 값이 없습니다.") from error
    if not math.isfinite(score):
        raise ValueError(f"{line}행의 {field}가 유효한 COMET 점수가 아닙니다.")
    return score


def validate_plain_log(path, rows):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        log = csv.DictReader(handle)
        if not {"sent_idx", "source_en", "translation_ko"} <= set(log.fieldnames or ()):
            raise ValueError(f"Plain 번역 로그에 문장 번호·원문·번역이 필요합니다: {path}")
        for index, (entry, row) in enumerate(zip_longest(log, rows)):
            if (entry is None or row is None or entry["sent_idx"] != str(index)
                    or entry["source_en"] != row["en"] or entry["translation_ko"] != row["no_rag_plain_ko"]):
                raise ValueError(f"Plain 번역 로그 {index + 1}행의 원문 또는 번역이 평가 입력과 다릅니다: {path}")


def load_rows(args):
    term_summary, term_rows = evaluation.read_eval_csv(args.terms_eval_csv)
    example_summary, example_rows = evaluation.read_eval_csv(args.examples_eval_csv)
    plain = evaluation.base.legacy.load_aligned_text(args.plain_file)
    terms, conflicts = evaluation.load_glossary(args.glossary, len(plain))
    return combine_rows(args, (term_summary, term_rows), (example_summary, example_rows), plain, terms, conflicts)


def combine_rows(args, term_result, example_result, plain, terms, conflicts):
    term_summary, term_rows = term_result
    example_summary, example_rows = example_result
    for field in ("target_match", "comet_mode", "comet_model"):
        if not term_summary.get(field) or term_summary[field] != example_summary.get(field):
            raise ValueError(f"두 기존 평가 CSV의 {field}가 같아야 합니다.")
    if term_summary["comet_mode"] != "reference_based":
        raise ValueError("한국어 정답으로 COMET을 계산한 기존 평가 CSV가 필요합니다.")
    model = term_summary["comet_model"]
    if model != evaluation.REFERENCE_MODEL:
        if args.comet_checkpoint is None or str(args.comet_checkpoint.absolute()) != model:
            raise ValueError(f"기존 평가에 기록된 COMET 체크포인트를 사용하세요: {model}")
    mode = term_summary["target_match"]
    if mode not in {"ko-particle", "boundary", "substring"}:
        raise ValueError(f"지원하지 않는 용어 매칭 방식입니다: {mode}")

    plain = evaluation.base.legacy.load_aligned_text(args.plain_file)
    if len(term_rows) != len(example_rows) or len(plain) != len(term_rows):
        raise ValueError(f"평가 문장 수가 다릅니다. 용어 {len(term_rows):,}행, 예문 {len(example_rows):,}행, Plain {len(plain):,}행")
    for summary in (term_summary, example_summary):
        if (str(len(terms)), str(conflicts)) != (str(summary.get("glossary_entries")),
                                                 str(summary.get("conflicting_glossary_keys"))):
            raise ValueError("기존 평가에 사용한 용어집과 등록 용어 또는 충돌 표제어 수가 다릅니다.")
    index, global_ids, line_ids = evaluation.build_source_index(terms)

    rows = []
    for line, (term_row, example_row, plain_ko) in enumerate(zip(term_rows, example_rows, plain, strict=True), 1):
        if (str(term_row.get("line")) != str(line) or not term_row["reference_ko"].strip()
                or any(term_row.get(field) != example_row.get(field)
                       for field in ("line", "en", "reference_ko", "no_rag_ko"))):
            raise ValueError(f"{line}행의 원문·정답·기존 No-RAG 번역이 두 CSV에서 동일한지 확인하세요.")
        source = term_row["en"]
        counts = evaluation.find_source_terms(source, index, global_ids, line_ids.get(line, {}))
        active = [(term_id, terms[term_id]) for term_id in counts]
        expected = [(term.source, term.targets, counts[term_id]) for term_id, term in active]
        for previous in (term_row, example_row):
            if glossary_hits(json.loads(previous["terms_json"])) != expected:
                raise ValueError(f"{line}행의 평가 용어가 기존 기록과 다릅니다. 같은 용어집을 지정하세요.")

        baseline_score = saved_score(term_row, "no_rag_comet", line)
        if baseline_score != saved_score(example_row, "no_rag_comet", line):
            raise ValueError(f"{line}행의 기존 No-RAG COMET 점수가 두 CSV에서 다릅니다.")
        row = {"row_type": "sentence", "line": line, "en": source, "reference_ko": term_row["reference_ko"],
               "en_term_occurrences": sum(counts.values()), "no_rag_cot_comet": baseline_score,
               "rag_terms_comet": saved_score(term_row, "rag_comet", line),
               "rag_examples_comet": saved_score(example_row, "rag_comet", line)}
        details = [dict(en=en, expected_ko=targets, occurrences=count) for en, targets, count in expected]
        translations = (term_row["no_rag_ko"], term_row["rag_ko"], example_row["rag_ko"], plain_ko)
        for side, translation in zip(CONDITIONS, translations, strict=True):
            row[side + "_ko"] = translation
            forms = evaluation.target_forms(translation, active, counts, mode)
            for (term_id, _), detail in zip(active, details, strict=True):
                matches = forms.get(term_id, ())
                detail[side + "_matched_ko"] = [phrase for phrase, _ in matches]
                detail[side + "_matched_surfaces"] = [surface for _, surface in matches]
            matches = sum(len(detail[side + "_matched_ko"]) for detail in details)
            row[side + "_matched_occurrences"] = matches
            row[side + "_term_rate_pct"] = 100 * matches / row["en_term_occurrences"] if counts else None
        previous_counts = (term_row["no_rag_matched_occurrences"], term_row["rag_matched_occurrences"],
                           example_row["rag_matched_occurrences"])
        if tuple(row[side + "_matched_occurrences"] for side in CONDITIONS[:3]) != previous_counts:
            raise ValueError(f"{line}행의 기존 용어 일치 건수가 현재 집계와 다릅니다.")
        row["terms_json"] = json.dumps(details, ensure_ascii=False)
        rows.append(row)

    plain_log = args.plain_file.with_suffix(".csv")
    if plain_log.is_file() and plain_log != args.plain_file:
        validate_plain_log(plain_log, rows)
        print(f"[Plain] 원문·행 순서 확인: {plain_log}", flush=True)
    else:
        print("[Plain] 번역 TXT의 행 수 확인. 원문 순서는 입력 파일과 같아야 합니다.", flush=True)
    return {"target_match": mode, "comet_model": model, "glossary_entries": len(terms)}, rows


def evaluate_text_files(args):
    aligned = evaluation.load_aligned_inputs({"en": args.source_file, "reference": args.reference_file,
                                             "no_rag_cot": args.no_rag_file, "rag_terms": args.rag_terms_file,
                                             "rag_examples": args.rag_examples_file, "no_rag_plain": args.plain_file})
    sources, references = aligned["en"], aligned["reference"]
    translations = [aligned[side] for side in CONDITIONS]
    terms, conflicts = evaluation.load_glossary(args.glossary, len(sources))
    scores, model = evaluation.score_comet(sources, references, translations, args)
    pairs = [evaluation.evaluate_pair(sources, translations[0], translations[index], references,
                                      terms, conflicts, "ko-particle", (scores[0], scores[index], model))
             for index in (1, 2)]
    metadata, rows = combine_rows(args, *pairs, translations[-1], terms, conflicts)
    return metadata, rows, scores[-1], pairs


def summarize(metadata, rows, *, fresh_comet=False):
    expected = sum(row["en_term_occurrences"] for row in rows)
    matches = [sum(row[side + "_matched_occurrences"] for row in rows) for side in CONDITIONS]
    scores = [fmean(row[side + "_comet"] for row in rows) for side in CONDITIONS]
    common = {key: metadata[key] for key in ("target_match", "comet_model", "glossary_entries",
                                            "bleu_signature", "lexical_tokenizer")}
    common.update(row_type="summary", sentences=len(rows),
                  sentences_with_terms=sum(row["en_term_occurrences"] > 0 for row in rows),
                  en_term_occurrences=expected)
    summary = []
    for side, matched, score in zip(CONDITIONS, matches, scores, strict=True):
        record = dict(common, condition=side, matched_occurrences=matched, comet=score, comet_delta=score - scores[0],
                      comet_source="new" if fresh_comet or side == "no_rag_plain" else "saved",
                      term_rate_pct=100 * matched / expected if expected else None,
                      term_rate_delta_pp=100 * (matched - matches[0]) / expected if expected else None)
        for metric in evaluation.LEXICAL_METRICS:
            record[metric] = metadata[side + "_" + metric]
            record[metric + "_delta"] = record[metric] - metadata["no_rag_cot_" + metric]
        summary.append(record)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms-eval-csv", type=Path, help="기존 용어 RAG 대 No-RAG 평가 CSV")
    parser.add_argument("--examples-eval-csv", type=Path, help="기존 용어+예문 RAG 대 No-RAG 평가 CSV")
    parser.add_argument("--source-file", type=Path, help="영어 원문 TXT")
    parser.add_argument("--reference-file", type=Path, help="한국어 정답 TXT")
    parser.add_argument("--no-rag-file", type=Path, help="CoT No-RAG 한국어 번역 TXT")
    parser.add_argument("--rag-terms-file", type=Path, help="용어 RAG 한국어 번역 TXT")
    parser.add_argument("--rag-examples-file", type=Path, help="용어+예문 RAG 한국어 번역 TXT")
    parser.add_argument("--terms-output-file", type=Path, help="새 용어 RAG 대 No-RAG 평가 CSV")
    parser.add_argument("--examples-output-file", type=Path, help="새 용어+예문 RAG 대 No-RAG 평가 CSV")
    parser.add_argument("--plain-file", required=True, type=Path, help="Plain No-RAG 한국어 번역 TXT")
    parser.add_argument("--glossary", required=True, type=Path, help="공통 평가 용어집")
    parser.add_argument("--output-file", required=True, type=Path, help="4조건 요약과 문장별 결과를 저장할 새 CSV")
    parser.add_argument("--comet-checkpoint", type=Path, help="기존 평가와 같은 로컬 COMET 체크포인트")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size는 1 이상이어야 합니다.")
    saved = (args.terms_eval_csv, args.examples_eval_csv)
    text_inputs = (args.source_file, args.reference_file, args.no_rag_file, args.rag_terms_file, args.rag_examples_file)
    pair_outputs = (args.terms_output_file, args.examples_output_file)
    fresh = any((*text_inputs, *pair_outputs))
    if fresh and (any(saved) or not all((*text_inputs, *pair_outputs))):
        parser.error("새 평가는 원문·정답·네 번역과 두 쌍별 출력 경로가 필요합니다.")
    if not fresh and not all(saved):
        parser.error("기존 평가를 합치려면 --terms-eval-csv와 --examples-eval-csv를 지정하세요.")
    outputs = (args.output_file, *pair_outputs) if fresh else (args.output_file,)
    if len(set(path.resolve() for path in outputs)) != len(outputs):
        parser.error("결과 파일은 서로 다른 경로를 지정하세요.")
    for path in outputs:
        if path.exists() or path.is_symlink():
            parser.error(f"결과 파일이 이미 있습니다: {path}")
    try:
        if fresh:
            metadata, rows, plain_scores, pairs = evaluate_text_files(args)
        else:
            metadata, rows = load_rows(args)
            (plain_scores,), _ = evaluation.score_comet(
                [row["en"] for row in rows], [row["reference_ko"] for row in rows],
                ([row["no_rag_plain_ko"] for row in rows],), args,
            )
        evaluation.score_lexical(metadata, rows, CONDITIONS)
        for row, score in zip(rows, plain_scores, strict=True):
            row["no_rag_plain_comet"] = score
        summary = summarize(metadata, rows, fresh_comet=fresh)
        if fresh:
            for path, pair in zip(pair_outputs, pairs, strict=True):
                evaluation.write_eval_csv(path, *pair)
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output_file.parent) as temp:
            staged = Path(temp) / args.output_file.name
            with staged.open("x", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(summary)
                writer.writerows(rows)
            args.output_file.hardlink_to(staged)
        print(f"[CSV] {args.output_file}", flush=True)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"오류: {error}\n")


if __name__ == "__main__":
    main()
