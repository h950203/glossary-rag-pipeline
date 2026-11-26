#!/usr/bin/env python3
"""Compare EN→KO glossary matches and reference-based translation scores."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean


SPEC = importlib.util.spec_from_file_location(
    "lang_eval_rag_base", Path(__file__).resolve().with_name("lang_eval_v3.2-args.py")
)
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

REFERENCE_MODEL = "Unbabel/wmt22-comet-da"
LEXICAL_METRICS = ("bleu", "rouge1", "rouge2", "rougeL")
LEXICAL_FIELDS = tuple(field for metric in LEXICAL_METRICS
                       for field in ("no_rag_" + metric, "rag_" + metric, metric + "_delta"))
LEXICAL_TOKEN = re.compile(r"\w+|[^\w\s]")
CSV_FIELDS = (
    "row_type", "line", "sentences", "sentences_with_terms", "en", "no_rag_ko", "rag_ko", "reference_ko",
    "en_term_occurrences", "no_rag_matched_occurrences", "rag_matched_occurrences",
    "no_rag_term_rate_pct", "rag_term_rate_pct", "term_rate_delta_pp",
    "no_rag_comet", "rag_comet", "comet_delta", "terms_json",
    "glossary_entries", "conflicting_glossary_keys", "target_match", "comet_mode", "comet_model",
    *LEXICAL_FIELDS, "bleu_signature", "lexical_tokenizer",
)


def load_glossary(path, sentence_count):
    if path.suffix.lower() != ".jsonl":
        return base.load_terms(path, sentence_count), 0

    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                first = parse_glossary_row(line, number)
                break
        else:
            raise ValueError("용어집에 항목이 없습니다.")

        if "source_term" in first:
            return base.load_terms(path, sentence_count), 0

        entries, conflicts = {}, set()

        def add(row, number):
            en, ko = row.get("en"), row.get("ko")
            if not all(isinstance(term, str) and term.strip() for term in (en, ko)):
                raise ValueError(f"용어 JSONL {number}행에 비어 있지 않은 en, ko 문자열이 필요합니다.")
            key = base.normalize(en)
            previous = entries.setdefault(key, base.Term(en.strip(), (ko.strip(),)))
            if base.normalize(previous.targets[0], case_sensitive=True) != base.normalize(ko, case_sensitive=True):
                conflicts.add(key)

        add(first, number)
        for number, line in enumerate(handle, number + 1):
            if line.strip():
                add(parse_glossary_row(line, number), number)
    terms = [term for key, term in entries.items() if key not in conflicts]
    if not terms:
        raise ValueError("충돌하는 영어 표제어를 제외한 뒤 평가할 용어가 없습니다.")
    return terms, len(conflicts)


def parse_glossary_row(line, number):
    try:
        row = json.loads(line)
    except ValueError as error:
        raise ValueError(f"용어 JSONL {number}행의 JSON 오류: {error}") from error
    if not isinstance(row, dict):
        raise ValueError(f"용어 JSONL {number}행은 JSON 객체여야 합니다.")
    return row


def build_source_index(terms):
    try:
        import ahocorasick
    except ImportError as error:
        raise RuntimeError("영어 용어 검색에 pyahocorasick이 필요합니다.") from error

    global_ids, line_ids = {}, defaultdict(dict)
    index = ahocorasick.Automaton()
    for term_id, term in enumerate(terms):
        key = base.normalize(term.source)
        if term.line is None:
            global_ids[key] = term_id
        else:
            line_ids[term.line][key] = term_id
        index.add_word(key, key)
    index.make_automaton()
    return index, global_ids, line_ids


def word_char(char):
    return char.isalnum() or char == "_"


def find_source_terms(sentence, index, global_ids, line_ids):
    sentence = base.normalize(sentence)
    matches = []
    for end, key in index.iter(sentence):
        term_id = line_ids.get(key, global_ids.get(key))
        if term_id is None:
            continue
        start, stop = end - len(key) + 1, end + 1
        if start and word_char(key[0]) and word_char(sentence[start - 1]):
            continue
        if stop < len(sentence) and word_char(key[-1]) and word_char(sentence[stop]):
            continue
        matches.append((start, stop, term_id))

    counts, next_start = Counter(), 0
    for start, stop, term_id in sorted(matches, key=lambda hit: (hit[0], -hit[1], hit[2])):
        if start >= next_start:
            counts[term_id] += 1
            next_start = stop
    return counts


def target_forms(sentence, active, counts, mode):
    forms = defaultdict(list)
    normalized = base.normalize(sentence)
    for start, stop, term_id, phrase in base.spans(sentence, active, "target", mode, limits=counts):
        forms[term_id].append((phrase, normalized[start:stop]))
    return forms


def rate_fields(expected, no_rag, rag):
    return {
        "en_term_occurrences": expected,
        "no_rag_matched_occurrences": no_rag,
        "rag_matched_occurrences": rag,
        "no_rag_term_rate_pct": 100 * no_rag / expected if expected else None,
        "rag_term_rate_pct": 100 * rag / expected if expected else None,
        "term_rate_delta_pp": 100 * (rag - no_rag) / expected if expected else None,
    }


def detail_rates(details):
    expected, no_rag, rag = 0, 0, 0
    if not isinstance(details, list):
        raise ValueError("terms_json은 배열이어야 합니다.")
    for term in details:
        count = term.get("occurrences") if isinstance(term, dict) else None
        if type(count) is not int or count < 1:
            raise ValueError("terms_json의 occurrences는 양의 정수여야 합니다.")
        matches = []
        for key in ("no_rag_matched_ko", "rag_matched_ko"):
            values = term.get(key)
            if (not isinstance(values, list) or len(values) > count
                    or not all(isinstance(value, str) and value.strip() for value in values)):
                raise ValueError(f"terms_json의 {key}에는 출현 횟수 이하의 일치 표기를 넣어야 합니다.")
            matches.append(len(values))
        expected += count
        no_rag += matches[0]
        rag += matches[1]
    return rate_fields(expected, no_rag, rag)


def term_detail(term, occurrences, no_rag_forms, rag_forms):
    return {
        "en": term.source,
        "expected_ko": term.targets,
        "occurrences": occurrences,
        "no_rag_matched_ko": [phrase for phrase, _ in no_rag_forms],
        "rag_matched_ko": [phrase for phrase, _ in rag_forms],
        "no_rag_matched_surfaces": [surface for _, surface in no_rag_forms],
        "rag_matched_surfaces": [surface for _, surface in rag_forms],
    }


def compare_terms(sources, no_rag, rag, references, terms, mode):
    index, global_ids, line_ids = build_source_index(terms)
    rows = []
    for line, (source, no_rag_ko, rag_ko) in enumerate(zip(sources, no_rag, rag), 1):
        counts = find_source_terms(source, index, global_ids, line_ids.get(line, {}))
        active = [(term_id, terms[term_id]) for term_id in counts]
        no_rag_forms = target_forms(no_rag_ko, active, counts, mode)
        rag_forms = target_forms(rag_ko, active, counts, mode)
        details = [
            term_detail(term, counts[term_id], no_rag_forms.get(term_id, ()), rag_forms.get(term_id, ()))
            for term_id, term in active
        ]
        rows.append({
            "row_type": "sentence", "line": line, "en": source, "no_rag_ko": no_rag_ko, "rag_ko": rag_ko,
            "reference_ko": references[line - 1] if references is not None else None,
            **detail_rates(details),
            "terms_json": json.dumps(details, ensure_ascii=False),
        })
    return rows


def update_term_summary(summary, rows):
    totals = [sum(row[key] for row in rows) for key in
              ("en_term_occurrences", "no_rag_matched_occurrences", "rag_matched_occurrences")]
    summary.update(row_type="summary", sentences=len(rows),
                   sentences_with_terms=sum(row["en_term_occurrences"] > 0 for row in rows),
                   **rate_fields(*totals))


def lexical_tokens(text):
    return LEXICAL_TOKEN.findall(unicodedata.normalize("NFC", text))


def lcs_length(tokens, reference):
    previous = [0] * (len(reference) + 1)
    for token in tokens:
        current = [0]
        for index, ref_token in enumerate(reference, 1):
            if token == ref_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_f1(overlap, total):
    return 200 * overlap / total if total else 0.0


def rouge_scores(tokens, reference):
    unigram_overlap = sum((Counter(tokens) & Counter(reference)).values())
    bigrams = Counter(zip(tokens, tokens[1:]))
    reference_bigrams = Counter(zip(reference, reference[1:]))
    bigram_overlap = sum((bigrams & reference_bigrams).values())
    total = len(tokens) + len(reference)
    return {
        "rouge1": rouge_f1(unigram_overlap, total),
        "rouge2": rouge_f1(bigram_overlap, max(0, len(tokens) - 1) + max(0, len(reference) - 1)),
        "rougeL": rouge_f1(lcs_length(tokens, reference), total),
    }


def score_lexical(summary, rows, sides):
    has_reference = [bool(row["reference_ko"] and row["reference_ko"].strip()) for row in rows]
    if not any(has_reference):
        return False
    if not all(has_reference):
        raise ValueError("한국어 정답이 비어 있는 행이 있습니다.")
    try:
        from sacrebleu.metrics import BLEU
    except ImportError as error:
        raise RuntimeError('BLEU 계산에는 SacreBLEU가 필요합니다. python -m pip install "sacrebleu>=2.4,<3"') from error

    # Input has already been tokenized by lexical_tokens.
    bleu = BLEU(tokenize="none", force=True, effective_order=True)
    references = [lexical_tokens(row["reference_ko"]) for row in rows]
    reference_lines = [" ".join(tokens) for tokens in references]
    for side in sides:
        candidates = [lexical_tokens(row[side + "_ko"]) for row in rows]
        candidate_lines = [" ".join(tokens) for tokens in candidates]
        summary[side + "_bleu"] = bleu.corpus_score(candidate_lines, [reference_lines]).score
        for row, tokens, ref, sentence, ref_sentence in zip(rows, candidates, references, candidate_lines, reference_lines):
            row[side + "_bleu"] = bleu.sentence_score(sentence, [ref_sentence]).score
            for metric, score in rouge_scores(tokens, ref).items():
                row[side + "_" + metric] = score
        for metric in LEXICAL_METRICS[1:]:
            field = side + "_" + metric
            summary[field] = fmean(row[field] for row in rows)
    summary["bleu_signature"] = str(bleu.get_signature())
    summary["lexical_tokenizer"] = "unicode-word-punctuation"
    return True


def add_lexical_scores(summary, rows):
    if score_lexical(summary, rows, ("no_rag", "rag")):
        for result in (summary, *rows):
            for metric in LEXICAL_METRICS:
                result[metric + "_delta"] = result["rag_" + metric] - result["no_rag_" + metric]


def read_eval_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"row_type", "en", "no_rag_ko", "rag_ko", "reference_ko", "terms_json"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: 기존 비교 CSV의 문장·정답·terms_json 열이 필요합니다.")
        first = next(reader, None)
        if first is None or first["row_type"] != "summary":
            raise ValueError(f"{path}: 첫 데이터 행은 summary여야 합니다.")
        summary = {key: first[key] for key in CSV_FIELDS if key in first}
        rows = []
        for number, saved in enumerate(reader, 3):
            if saved["row_type"] != "sentence" or any(saved[key] is None for key in required):
                raise ValueError(f"{path}: CSV {number}행의 문장 데이터가 잘못됐습니다.")
            row = {key: saved[key] for key in CSV_FIELDS if key in saved}
            try:
                row.update(detail_rates(json.loads(row["terms_json"])))
            except ValueError as error:
                raise ValueError(f"{path}: CSV {number}행의 {error}") from error
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: 평가할 문장이 없습니다.")
    update_term_summary(summary, rows)
    return summary, rows


def ensure_xlmr_compatibility(model):
    from comet.encoders.xlmr import XLMREncoder
    from transformers import __version__ as transformers_version

    encoder = model.encoder
    if not isinstance(encoder, XLMREncoder) or int(transformers_version.split(".")[0]) < 5:
        return

    # Transformers 5 omits the empty pooler slot from the tuple expected by COMET.
    def forward(input_ids, attention_mask, **kwargs):
        outputs = encoder.model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True, return_dict=True)
        return {"sentemb": outputs.last_hidden_state[:, 0, :], "wordemb": outputs.last_hidden_state,
                "all_layers": outputs.hidden_states, "attention_mask": attention_mask}

    encoder.forward = forward


def score_comet(sources, references, translations_by_condition, args):
    try:
        import torch
        from comet import load_from_checkpoint
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("COMET 평가에 unbabel-comet과 PyTorch가 필요합니다.") from error

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. --device cpu로 실행하세요.")

    # Hugging Face snapshot paths locate hparams.yaml beside the checkpoint directory.
    model_id = str(args.comet_checkpoint.absolute()) if args.comet_checkpoint else REFERENCE_MODEL
    print(f"[COMET] {model_id}, device={device}", flush=True)
    try:
        if args.comet_checkpoint:
            checkpoint = model_id
        else:
            # COMET's legacy fallback masks the original Hub download error.
            snapshot = snapshot_download(repo_id=REFERENCE_MODEL, allow_patterns=["hparams.yaml", "checkpoints/model.ckpt"])
            checkpoint = str(Path(snapshot) / "checkpoints" / "model.ckpt")
        model = load_from_checkpoint(checkpoint)
    except Exception as error:
        raise RuntimeError(f"COMET 모델을 로드할 수 없습니다 ({type(error).__name__}): {error}") from error
    if not model.requires_references():
        raise ValueError("정답 번역을 입력으로 사용하는 COMET 체크포인트를 지정하세요.")
    ensure_xlmr_compatibility(model)

    samples = []
    for translations in translations_by_condition:
        for source, translation, reference in zip(sources, translations, references, strict=True):
            samples.append({"src": source, "mt": translation, "ref": reference})
    # Follow the v3.2 evaluator's direct batches and keep scores in input order.
    model.to(device)
    model.eval()
    scores = []
    with torch.no_grad():
        for batch_number, start in enumerate(range(0, len(samples), args.batch_size), 1):
            batch = samples[start:start + args.batch_size]
            inputs = model.prepare_sample(batch, stage="predict")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            scores.extend(model(**inputs).score.detach().cpu().tolist())
            if batch_number % 25 == 0 or start + len(batch) == len(samples):
                print(f"[COMET] {start + len(batch)}/{len(samples)}", flush=True)
    if len(scores) != len(samples) or not all(map(math.isfinite, scores)):
        raise RuntimeError("COMET이 모든 문장에 대해 유효한 점수를 반환하지 않았습니다.")
    size = len(sources)
    return [scores[i:i + size] for i in range(0, len(scores), size)], model_id


def calculate_comet(sources, no_rag, rag, references, args):
    (no_rag_scores, rag_scores), model_id = score_comet(sources, references, (no_rag, rag), args)
    return no_rag_scores, rag_scores, model_id


def comet_fields(no_rag, rag):
    return {"no_rag_comet": no_rag, "rag_comet": rag, "comet_delta": rag - no_rag}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, help="기존 평가 CSV의 terms_json과 정답을 사용하고 저장된 COMET 점수 재사용")
    parser.add_argument("--source-file", "--en-file", type=Path, help="영어 원문 TXT")
    parser.add_argument("--no-rag-file", type=Path, help="No-RAG 한국어 번역 TXT")
    parser.add_argument("--rag-file", type=Path, help="RAG 한국어 번역 TXT")
    parser.add_argument("--glossary", "--glossary-file", type=Path,
                        help="en/ko JSONL 또는 lang_eval_v3.2의 source_term/target_terms TSV/JSONL")
    parser.add_argument("--output-file", "--output-csv", type=Path, required=True, help="저장할 CSV 경로")
    parser.add_argument("--reference-file", type=Path, help="한국어 정답 TXT. 지정하면 COMET·BLEU·ROUGE 평가")
    parser.add_argument("--comet-checkpoint", type=Path, help="정답 번역을 사용하는 로컬 COMET 체크포인트")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8, help="COMET 배치 크기 (기본 8)")
    parser.add_argument("--target-match", choices=("ko-particle", "boundary", "substring"), default="ko-particle",
                        help="한국어 용어 매칭 방식. ko-particle은 조사 등을 허용하고 boundary는 독립 표기만 인정합니다.")
    return parser


def write_eval_csv(path, summary, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(summary)
        writer.writerows(rows)
    print(f"[CSV] {path}", flush=True)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size는 1 이상이어야 합니다.")
    text_inputs = (args.source_file, args.no_rag_file, args.rag_file, args.glossary)
    if args.input_csv:
        if any((*text_inputs, args.reference_file, args.comet_checkpoint)):
            parser.error("--input-csv에는 별도의 원문·번역·정답·용어집·COMET 체크포인트를 지정하지 않습니다.")
    elif not all(text_inputs):
        parser.error("--input-csv 또는 --source-file, --no-rag-file, --rag-file, --glossary를 지정하세요.")
    if args.comet_checkpoint and args.reference_file is None:
        parser.error("--comet-checkpoint를 사용하려면 --reference-file을 지정하세요.")
    if args.output_file.exists():
        parser.error(f"결과 파일이 이미 있습니다: {args.output_file}")

    try:
        if args.input_csv:
            summary, rows = read_eval_csv(args.input_csv)
            add_lexical_scores(summary, rows)
        else:
            summary, rows = evaluate_texts(args)
        write_eval_csv(args.output_file, summary, rows)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"오류: {error}\n")


def load_aligned_inputs(inputs):
    aligned = {label: base.legacy.load_aligned_text(path) for label, path in inputs.items()}
    counts = {label: len(lines) for label, lines in aligned.items()}
    if len(set(counts.values())) != 1:
        raise ValueError("입력 TXT 행 수가 다릅니다: " + ", ".join(f"{label}={n}" for label, n in counts.items()))
    if not aligned["en"]:
        raise ValueError("평가할 문장이 없습니다.")
    for label in ("en", "reference"):
        for line, sentence in enumerate(aligned.get(label, []), 1):
            if not sentence:
                raise ValueError(f"{label} TXT의 {line}행이 비어 있습니다.")
    return aligned


def evaluate_pair(sources, no_rag, rag, references, terms, conflicts, target_match, comet=None):
    rows = compare_terms(sources, no_rag, rag, references, terms, target_match)
    summary = {"glossary_entries": len(terms), "conflicting_glossary_keys": conflicts,
               "target_match": target_match}
    update_term_summary(summary, rows)
    total = summary["en_term_occurrences"]
    print(f"[Glossary] {len(terms)}개 등록, 충돌 {conflicts}개 제외, 영어 용어 출현 {total}건", flush=True)

    if references is None:
        summary["comet_mode"] = "skipped_no_reference"
        print("[평가] 한국어 정답 파일이 없어 용어 인식률만 저장합니다.", flush=True)
    else:
        add_lexical_scores(summary, rows)
        scores_no_rag, scores_rag, model_id = comet
        for row, no_rag_score, rag_score in zip(rows, scores_no_rag, scores_rag, strict=True):
            row.update(comet_fields(no_rag_score, rag_score))
        summary.update(comet_fields(fmean(scores_no_rag), fmean(scores_rag)))
        summary.update(comet_mode="reference_based", comet_model=model_id)
    return summary, rows


def evaluate_texts(args):
    inputs = {"en": args.source_file, "no-rag": args.no_rag_file, "rag": args.rag_file}
    if args.reference_file is not None:
        inputs["reference"] = args.reference_file
    aligned = load_aligned_inputs(inputs)
    sources, no_rag, rag, references = aligned["en"], aligned["no-rag"], aligned["rag"], aligned.get("reference")
    terms, conflicts = load_glossary(args.glossary, len(sources))
    comet = calculate_comet(sources, no_rag, rag, references, args) if references is not None else None
    return evaluate_pair(sources, no_rag, rag, references, terms, conflicts, args.target_match, comet)


if __name__ == "__main__":
    main()
