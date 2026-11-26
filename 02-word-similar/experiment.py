"""Apply the covered expansion policy to existing aligned sentences."""
from argparse import ArgumentParser, Namespace
from itertools import islice
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
import build_glossary
import export_sentence_candidates
from groups import write_jsonl
from result_layout import logged_run, new_run, promote, run_dirs, write_json

LANGUAGES = ("ko", "en")
MODE = "covered_expansions_small_sample"
ALGORITHM = "3B_evidence"
SETTINGS = {"min_atomic_sentences": 2, "min_compound_sentences": 2,
            "min_verb_sentences": 2, "min_cvalue": 1.0}


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as source:
        for raw in source:
            if raw.strip():
                yield json.loads(raw)


def latest_alignment(method):
    """Convenience lookup when the alignment result folder is available."""
    for folder in run_dirs(ROOT.parent / "01-word-alignment/result"):
        manifest = folder / "details/run.json"
        if not manifest.is_file():
            continue
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if meta.get("status") != "complete" or meta.get("command") != "run":
            continue
        name = meta.get("public_files", {}).get(method, {}).get("term-pairs.jsonl")
        if name and (folder / name).is_file():
            return folder / name
    raise ValueError(f"{method}의 정렬 결과를 찾지 못했습니다. --input으로 term-pairs.jsonl을 지정하세요")


def freeze_input(path, target, method, start, count):
    digest = hashlib.sha256()
    selected = 0
    with path.open("rb") as source, target.open("wb") as output:
        lines = (line for line in source if line.strip() and json.loads(line).get("method") == method)
        for line in islice(lines, start - 1, start - 1 + count if count else None):
            output.write(line)
            digest.update(line)
            selected += 1
    if not selected or (count and selected != count):
        raise ValueError(f"{method}: {start}번째부터 요청한 문장 수를 충족하지 못했습니다 (선택 {selected}문장)")
    return selected, digest.hexdigest()


def term_id(lang, unit):
    # Same identity as word-similar build_terms: language, unit type, tokens.
    raw = json.dumps([unit["type"], unit["tokens"]], ensure_ascii=False)
    return lang + "_" + hashlib.sha256(raw.encode()).hexdigest()[:20]


def observed_pairs(row, unit_types):
    return [
        {"pair_index": index, "unit_type": pair["unit_type"], "score": pair["score"],
         **{lang: {"surface": pair[lang], "span": pair[f"{lang}_span"],
                   "term_id": term_id(lang, pair[f"{lang}_unit"]),
                   "normalized": pair[f"{lang}_unit"]["normalized"]}
            for lang in LANGUAGES}}
        for index, pair in enumerate(row["pairs"], 1) if pair["unit_type"] in unit_types
    ]


def label(term, membership, representative_observed):
    root_id = membership["representative_term_id"]
    if root_id is None:
        return {"role": "unassigned", "review_status": membership["status"],
                "selection_reason": membership["selection_reason"]}
    return {"role": "representative" if root_id == term["term_id"] else "expansion",
            "representative_id": root_id, "representative": membership["root_term"],
            "representative_observed_in_sentence": root_id in representative_observed,
            "review_status": membership["status"], "selection_reason": membership["selection_reason"]}


def review_rows(engine, sample, unit_types):
    fields = ("term", "representative_term_id", "status", "selection_reason")
    memberships = {row["term_id"]: {key: row[key] for key in fields}
                   for row in read_jsonl(engine / ALGORITHM / "memberships.jsonl")}
    for row in memberships.values():
        root_id = row["representative_term_id"]
        if root_id:
            row["root_term"] = memberships[root_id]["term"]
    for row in read_jsonl(sample):
        pairs = observed_pairs(row, unit_types)
        present = {pair[lang]["term_id"] for pair in pairs for lang in LANGUAGES}
        decisions = [
            {"pair_index": pair["pair_index"],
             **{lang: label(pair[lang], memberships[pair[lang]["term_id"]], present) for lang in LANGUAGES}}
            for pair in pairs
        ]
        yield {"line_id": row["line_id"], "ko_sentence": row["ko"],
               "en_sentence": row["en"], "observed_pairs": pairs, "membership": decisions}


def codex_row(row):
    pairs = [{key: p[key] for key in ("pair_index", "unit_type", "score")} |
             {lang: {k: p[lang][k] for k in ("surface", "span", "normalized")} for lang in LANGUAGES}
             for p in row["observed_pairs"]]
    decisions = [
        {"pair_index": decision["pair_index"], **{
            lang: {k: v for k, v in decision[lang].items() if k != "representative_id"}
            for lang in LANGUAGES}}
        for decision in row["membership"]
    ]
    return {"line_id": row["line_id"], "ko_sentence": row["ko_sentence"],
            "en_sentence": row["en_sentence"], "observed_pairs": pairs, "membership": decisions}


def export_codex_batches(folder, details, sample, unit_types, selected, batch_size):
    folder.mkdir()
    batches = folder / "codex_batches"
    batches.mkdir()
    rows = (codex_row(row) for row in review_rows(details, sample, unit_types))
    with (folder / "codex_review.jsonl").open("w", encoding="utf-8") as review:
        for number in range(1, (selected + batch_size - 1) // batch_size + 1):
            def batch_rows():
                for row in islice(rows, batch_size):
                    review.write(json.dumps(row, ensure_ascii=False) + "\n")
                    yield row

            write_jsonl(batches / f"{number:05}.jsonl", batch_rows())
    (folder / "codex_review_prompt.md").write_text(
        (ROOT / "codex_review_prompt.md").read_text(encoding="utf-8"), encoding="utf-8")


def execute(args):
    if args.start < 1 or min(args.sentences, args.batch_size) < 0:
        raise ValueError("--start는 양수, --sentences와 --batch-size는 0 이상이어야 합니다")
    source = (args.input or latest_alignment(args.alignment_method)).expanduser().resolve()
    path = new_run(Path(os.environ.get("PP_STAGE_RESULT_ROOT", ROOT / "result")))
    with logged_run(path, "word-similar-v1.3") as run:
        details = path / "details"
        sample = details / "sample_alignments.jsonl"
        run.update(input=str(source), alignment_method=args.alignment_method,
                   start=args.start, unit_types=args.unit_types, mode=MODE)
        selected, digest = freeze_input(source, sample, args.alignment_method, args.start, args.sentences)
        run.update(sentence_count=selected, sample_sha256=digest)
        config = build_glossary.parser().parse_args([])
        vars(config).update(input=sample, output_dir=details, unit_types=args.unit_types,
                            methods=[ALGORITHM], mode=MODE, expansion_policy="lexically_covered_only",
                            include_occurrences=args.include_occurrences, require_both=args.require_both,
                            candidate_form=args.candidate_form, alignment_method=args.alignment_method, **SETTINGS)
        build_glossary.run(config, prepared_input=True)
        names = ("groups-simple.csv", "group-pairs.csv")
        for name in names:
            promote(details / ALGORITHM / name, path / name)
        summary_path = details / "3_complete-summary.json"
        export_sentence_candidates.run(Namespace(
            groups_csv=path / "groups-simple.csv", alignments=details / "sources.jsonl",
            output=path / "3_complete.jsonl", format="jsonl", candidate_form=args.candidate_form,
            include_occurrences=args.include_occurrences, require_both=args.require_both,
            overwrite=False, source_format="grouping_sources", summary_path=summary_path,
        ))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        write_json(summary_path, {**summary, "original_alignments": str(source)})
        run.update(primary_method=ALGORITHM, methods=[ALGORITHM],
                   public_files={ALGORITHM: {name: name for name in names}},
                   sentence_candidates="3_complete.jsonl")
        if args.batch_size:
            export_codex_batches(details / "codex", details, sample, args.unit_types, selected, args.batch_size)
            run["codex_review"] = "details/codex/codex_review.jsonl"
    return 0


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="v1.2 term-pairs.jsonl")
    parser.add_argument("--alignment-method", default="C_nltk_treebank")
    parser.add_argument("--sentences", "--max-records", type=int, default=0, help="문장 수. 기본값 0은 선택 방식 전체")
    parser.add_argument("--start", type=int, default=1, help="해당 정렬 방식에서 시작할 문장 순번")
    parser.add_argument("--batch-size", type=int, default=0, help="양수를 주면 details/codex에 검토 묶음 저장")
    parser.add_argument("--candidate-form", choices=["representative", "surface"], default="representative")
    parser.add_argument("--include-occurrences", action="store_true", help="3_complete에 원문 표기와 좌표 추가")
    parser.add_argument("--require-both", action="store_true", help="3_complete에 양쪽 후보가 있는 문장만 저장")
    parser.add_argument("--unit-types", nargs="+", choices=["nominal", "verbal"], default=["nominal", "verbal"])
    args = parser.parse_args()
    try:
        return execute(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
