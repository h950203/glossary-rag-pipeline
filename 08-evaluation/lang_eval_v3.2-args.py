#!/usr/bin/env python3
"""COMET, 표준 chrF++, 사전 기반 용어 표기 충족률을 평가한다."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import re
import sys
import unicodedata
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("lang_eval_v32_legacy", ROOT / "lang_eval_v3.1-args.py")
legacy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = legacy
SPEC.loader.exec_module(legacy)

# Conservative suffix whitelist, not a morphological analyzer.
PARTICLES = ("은", "는", "이", "가", "을", "를", "의", "에", "에서", "에게", "께", "으로", "로",
             "와", "과", "도", "만", "부터", "까지", "보다", "처럼", "마다", "이나", "나", "이라도", "라도",
             "이며", "이고", "이다", "입니다", "이었다", "였다", "에서는", "에는", "으로는", "로는",
             "에도", "에서도", "으로도", "로도", "만을", "만이", "만으로", "까지는", "부터는")
DETAIL_FIELDS = ("line", "source_term", "target_terms", "expected_occurrences", "matched_occurrences",
                 "preferred_matched_occurrences", "reference_matched_occurrences", "matched_target_forms",
                 "success_rate", "source", "reference", "translation")


@dataclass(frozen=True)
class Term:
    source: str
    targets: tuple[str, ...]
    line: int | None = None


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def normalize(text, case_sensitive=False):
    text = " ".join(unicodedata.normalize("NFC", text).split())
    return text if case_sensitive else text.casefold()


def load_terms(path, sentence_count, case_sensitive=False):
    """TSV: source_term,target_terms[,line]. JSONL: same keys, target_terms list/string."""
    path = Path(path)
    if path.suffix.lower() == ".tsv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not {"source_term", "target_terms"}.issubset(reader.fieldnames or []):
                raise ValueError("용어 TSV에는 source_term, target_terms 헤더가 필요합니다.")
            records = list(reader)
    elif path.suffix.lower() == ".jsonl":
        records = []
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except ValueError as error:
                    raise ValueError(f"용어 JSONL {number}행의 JSON 오류: {error}") from error
    else:
        raise ValueError("용어 파일은 헤더 있는 .tsv 또는 .jsonl을 사용하세요.")
    terms, seen = [], set()
    for number, row in enumerate(records, 1):
        if not isinstance(row, dict) or not isinstance(row.get("source_term"), str):
            raise ValueError(f"용어 {number}: source_term 문자열이 필요합니다. word/meaning 설명 사전과는 형식이 다릅니다.")
        source = row["source_term"].strip()
        targets = row.get("target_terms")
        if isinstance(targets, str):
            targets = targets.split("|")
        if not source or not isinstance(targets, list) or not targets or not all(isinstance(t, str) and t.strip() for t in targets):
            raise ValueError(f"용어 {number}: 빈 source_term 또는 잘못된 target_terms입니다.")
        cleaned, target_seen = [], set()
        for target in targets:
            key = normalize(target, case_sensitive)
            if key not in target_seen:
                cleaned.append(target.strip())
                target_seen.add(key)
        line = row.get("line")
        if line in (None, ""):
            line = None
        elif isinstance(line, bool) or not (isinstance(line, int) or isinstance(line, str) and line.isdigit()):
            raise ValueError(f"용어 {number}: line은 1부터 시작하는 정수입니다.")
        else:
            line = int(line)
            if not 1 <= line <= sentence_count:
                raise ValueError(f"용어 {number}: line={line}이 입력 범위를 벗어납니다.")
        key = (normalize(source, case_sensitive), line)
        if key in seen:
            raise ValueError(f"중복 source_term/line: {source}, {line}. 허용 번역을 한 항목에 모으세요.")
        seen.add(key)
        terms.append(Term(source, tuple(cleaned), line))
    if not terms:
        raise ValueError("유효한 용어가 없습니다.")
    return terms


@lru_cache(maxsize=8192)
def term_pattern(phrase, mode):
    body = re.escape(phrase)
    if mode == "substring":
        return re.compile(body)
    left = r"(?<!\w)" if re.match(r"\w", phrase[0]) else ""
    right = ""
    if re.match(r"\w", phrase[-1]):
        if mode == "ko-particle":
            suffixes = "|".join(re.escape(s) for s in sorted(PARTICLES, key=len, reverse=True))
            right = rf"(?:{suffixes})?(?!\w)"
        else:
            right = r"(?!\w)"
    return re.compile(left + body + right)


def spans(text, entries, side, mode, case_sensitive=False, limits=None):
    """Leftmost-longest, non-overlapping matches; one span is credited only once."""
    text = normalize(text, case_sensitive)
    matches = []
    for index, term in entries:
        phrases = (term.source,) if side == "source" else term.targets
        for phrase in phrases:
            for found in term_pattern(normalize(phrase, case_sensitive), mode).finditer(text):
                matches.append((found.start(), found.end(), index, phrase))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2], item[3]))
    selected, end = [], -1
    alternatives = defaultdict(dict)
    if limits is not None:
        for begin, finish, index, phrase in matches:
            alternatives[(begin, finish)].setdefault(index, phrase)
    for item in matches:
        if item[0] >= end:
            selected.append(item)
            end = item[1]
    if limits is None:
        return selected
    # A target occurrence may be shared by source synonyms. Assign each span
    # once with per-term capacities, rather than crediting all to the first term.
    options = [alternatives[(hit[0], hit[1])] for hit in selected]
    assigned, owners = {}, defaultdict(set)
    for incoming in range(len(selected)):
        pending, visited_regions, parent_term = deque([incoming]), {incoming}, {}
        found = None
        while pending and found is None:
            region = pending.popleft()
            for index in options[region]:
                if index in parent_term:
                    continue
                parent_term[index] = region
                if len(owners[index]) < limits[index]:
                    found = index
                    break
                for occupied in sorted(owners[index]):
                    if occupied not in visited_regions:
                        visited_regions.add(occupied)
                        pending.append(occupied)
        while found is not None:
            region = parent_term[found]
            previous = assigned.get(region)
            if previous is not None:
                owners[previous].remove(region)
            assigned[region] = found
            owners[found].add(region)
            found = previous
    return [(selected[region][0], selected[region][1], index, options[region][index])
            for region, index in sorted(assigned.items())]


def evaluate_terms(sources, references, candidates, terms, mode, case_sensitive=False):
    global_terms = {normalize(t.source, case_sensitive): (i, t) for i, t in enumerate(terms) if t.line is None}
    specific = defaultdict(dict)
    for i, term in enumerate(terms):
        if term.line is not None:
            specific[term.line][normalize(term.source, case_sensitive)] = (i, term)
    details = []
    for line, (source, reference, candidate) in enumerate(zip(sources, references, candidates), 1):
        # A line-specific entry replaces the global entry for that source term.
        entries = list({**global_terms, **specific.get(line, {})}.values())
        source_hits = Counter(hit[2] for hit in spans(source, entries, "source", "boundary", case_sensitive))
        active = [(i, t) for i, t in entries if source_hits[i]]
        target_spans = spans(candidate, active, "target", mode, case_sensitive, limits=source_hits)
        target_hits = Counter(hit[2] for hit in target_spans)
        preferred_forms = {i: normalize(t.targets[0], case_sensitive) for i, t in active}
        preferred_hits = Counter(hit[2] for hit in target_spans
                                 if normalize(hit[3], case_sensitive) == preferred_forms[hit[2]])
        reference_hits = Counter(hit[2] for hit in spans(reference, active, "target", mode, case_sensitive, limits=source_hits))
        for index, term in active:
            total = source_hits[index]
            matched = min(total, target_hits[index])
            forms = [hit[3] for hit in target_spans if hit[2] == index][:total]
            details.append({"line": line, "source_term": term.source, "target_terms": "|".join(term.targets),
                            "expected_occurrences": total, "matched_occurrences": matched,
                            "preferred_matched_occurrences": min(total, preferred_hits[index]),
                            "reference_matched_occurrences": min(total, reference_hits[index]),
                            "matched_target_forms": json.dumps(forms, ensure_ascii=False),
                            "success_rate": 100 * matched / total,
                            "source": source, "reference": reference, "translation": candidate})
    total = sum(r["expected_occurrences"] for r in details)
    matched = sum(r["matched_occurrences"] for r in details)
    preferred = sum(r["preferred_matched_occurrences"] for r in details)
    reference_matched = sum(r["reference_matched_occurrences"] for r in details)
    summary = {"status": "measured" if total else "no_source_matches", "glossary_entries": len(terms),
               "eligible_sentence_term_pairs": len(details), "sentences_with_terms": len({r["line"] for r in details}),
               "expected_occurrences": total, "matched_occurrences": matched,
               "missed_occurrences": total - matched,
               "success_rate": 100 * matched / total if total else None,
               "preferred_success_rate": 100 * preferred / total if total else None,
               "reference_matched_occurrences": reference_matched,
               "reference_unmatched_occurrences": total - reference_matched,
               "counting": "source_occurrences_capped_nonoverlapping_target_matches_per_sentence",
               "target_match": mode, "source_match": "boundary", "case_sensitive": case_sensitive,
               "normalization": "NFC + whitespace collapse + optional casefold",
               "overlap_policy": "leftmost_longest_nonoverlapping; line-specific overrides global",
               "interpretation": "용어 표기 충족률이며 문맥상 번역 정확성/안전성을 보장하지 않습니다."}
    return summary, details


def calculate_chrfpp(candidates, references):
    try:
        import sacrebleu
        from sacrebleu.metrics import CHRF
    except ImportError as error:
        raise RuntimeError('SacreBLEU가 필요합니다: python -m pip install "sacrebleu>=2.4,<3"') from error
    metric = CHRF(char_order=6, word_order=2, beta=2, lowercase=False, whitespace=False, eps_smoothing=False)
    score = float(metric.corpus_score(candidates, [references]).score)
    return score, {"signature": str(metric.get_signature()), "sacrebleu_version": sacrebleu.__version__,
                   "char_order": 6, "word_order": 2, "beta": 2, "scale": "0..100"}


def calculate_comet(inputs, device):
    engine = legacy.load_engine()
    if device == "auto":
        device = "cuda" if engine.torch.cuda.is_available() else "cpu"
    if device == "cuda" and not engine.torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. --device cpu로 실행하세요.")
    print(f"[COMET] {engine.COMET_MODEL_NAME}, device={device}", flush=True)
    model = engine.load_from_checkpoint(engine.download_model(engine.COMET_MODEL_NAME))
    try:
        score = float(engine.calculate_comet(inputs["sources"], inputs["references"], inputs["candidates"], model, device))
    finally:
        del model
        if device == "cuda":
            engine.torch.cuda.empty_cache()
    return score, engine.COMET_MODEL_NAME, device


def append_record(path, record):
    """Model scoring is outside the lock; concurrent writers append safely on Linux."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {"schema_version": "3.2", "experiments": []}
        if not isinstance(data, dict) or not isinstance(data.get("experiments"), list):
            raise ValueError("누적 로그에 experiments 목록이 없습니다. 새 --output-log를 지정하세요.")
        data["experiments"].append(record)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(path)


def build_parser():
    parser = legacy.build_parser()
    parser.description = __doc__
    parser.set_defaults(output_log=str(ROOT / "log/evaluation_log_v32_args.json"))
    for action in parser._actions:
        if action.dest == "output_log":
            action.help = "누적 평가 JSON 경로 (기본: log/evaluation_log_v32_args.json)"
        elif action.dest == "check_only":
            action.help = "파일/행 대응/용어 대상만 검사하고 종료 (평가 모델·새 파일 없음)"
    terms = parser.add_mutually_exclusive_group(required=True)
    terms.add_argument("--terms-file", "--glossary-file", help="평가용 TSV/JSONL: source_term,target_terms[,line]")
    terms.add_argument("--skip-terms", action="store_true", help="용어 사전 없이 COMET/chrF++만 평가 (용어 점수=null)")
    parser.add_argument("--target-match", choices=("auto", "boundary", "ko-particle", "substring"), default="auto",
                        help="기본 auto: ko는 유한 조사 규칙, 다른 언어는 단어 경계. substring은 탐색용")
    parser.add_argument("--term-case-sensitive", action="store_true", help="용어 매칭의 대소문자를 구분")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = legacy.resolve_inputs(parser, args)
    terms_path = Path(args.terms_file).resolve() if args.terms_file else None
    if terms_path == inputs["output_log"]:
        parser.error("output-log와 terms-file은 다른 경로여야 합니다.")
    input_paths = [inputs[key] for key in ("source_path", "reference_path", "hypothesis_path")]
    if terms_path:
        input_paths.append(terms_path)
    hashes = {str(path): digest(path) for path in input_paths}
    mode = ("ko-particle" if args.target_lang == "ko" else "boundary") if args.target_match == "auto" else args.target_match
    terms = load_terms(terms_path, len(inputs["sources"]), args.term_case_sensitive) if terms_path else []
    if terms_path:
        terminology, details = evaluate_terms(inputs["sources"], inputs["references"], inputs["candidates"],
                                               terms, mode, args.term_case_sensitive)
    else:
        terminology, details = {"status": "skipped", "success_rate": None, "preferred_success_rate": None}, []
    if terminology["status"] != "measured":
        print(f"[주의] 용어 평가={terminology['status']}; 용어 성공률은 0이나 100이 아니라 null입니다.", flush=True)
    if terminology.get("reference_unmatched_occurrences", 0):
        print(f"[주의] 정답에서도 허용 표현이 확인되지 않은 용어 발생={terminology['reference_unmatched_occurrences']}회. "
              "문맥/허용 번역을 검토하세요. 평가 분모에서는 제외하지 않습니다.", flush=True)
    print(json.dumps({"source": str(inputs["source_path"]), "reference": str(inputs["reference_path"]),
                      "translation": str(inputs["hypothesis_path"]), "sentence_count": len(inputs["sources"]),
                      "empty_hypotheses": sum(not line for line in inputs["candidates"]),
                      "terms_file": str(terms_path) if terms_path else None, "terminology": terminology,
                      "output_log": str(inputs["output_log"])}, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return
    # Check an existing log before loading models, without changing it.
    if inputs["output_log"].exists():
        old = json.loads(inputs["output_log"].read_text(encoding="utf-8-sig"))
        if not isinstance(old, dict) or not isinstance(old.get("experiments"), list):
            raise ValueError("누적 로그 형식이 맞지 않습니다. 새 --output-log를 지정하세요.")
    chrfpp, chrf_config = calculate_chrfpp(inputs["candidates"], inputs["references"])
    print(f"[chrF++] {chrfpp:.6f} / {chrf_config['signature']}", flush=True)
    comet, model_name, device = calculate_comet(inputs, args.device)
    if not math.isfinite(chrfpp) or not math.isfinite(comet):
        raise RuntimeError("유한한 평가 점수가 아닙니다. 완료 결과로 저장하지 않습니다.")
    if any(digest(path) != hashes[str(path)] for path in input_paths):
        raise RuntimeError("평가 중 입력/용어 파일이 변경됐습니다. 완료 결과로 저장하지 않습니다.")
    now = datetime.now(timezone.utc)
    experiment_id = f"eval-v32-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    report_dir = inputs["output_log"].parent / (inputs["output_log"].stem + "_details") / experiment_id
    report_dir.mkdir(parents=True, exist_ok=False)
    term_details = report_dir / "terms.tsv"
    with term_details.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(details)
    policy = {"terms_sha256": hashes.get(str(terms_path)), "target_match": mode,
              "term_case_sensitive": args.term_case_sensitive, "chrfpp": chrf_config,
              "comet_model": model_name, "source_lang": args.source_lang, "target_lang": args.target_lang}
    code_hashes = {path.name: digest(path) for path in (
        Path(__file__), ROOT / "lang_eval_v3.1-args.py", ROOT / "lang_eval_v3.1.py")}
    comparison_key = {"source_sha256": hashes[str(inputs["source_path"])],
                      "reference_sha256": hashes[str(inputs["reference_path"])], "code_sha256": code_hashes, **policy}
    comparison_group = hashlib.sha256(json.dumps(comparison_key, sort_keys=True).encode()).hexdigest()
    record = {"id": experiment_id, "name": args.name or inputs["hypothesis_path"].stem,
              "script_version": "lang_eval_v3.2-args", "status": "completed", "comparison_group": comparison_group,
              "input": {"file": str(inputs["source_path"]), "reference_file": str(inputs["reference_path"]),
                        "terms_file": str(terms_path) if terms_path else None},
              "output": {"txt": str(inputs["hypothesis_path"]), "term_details": str(term_details)},
              "input_sha256": hashes, "code_sha256": code_hashes,
              "evaluation": {"source_lang": args.source_lang, "target_lang": args.target_lang, "status": "completed",
                             "evaluated_at": now.isoformat(), "sentence_count": len(inputs["sources"]),
                             "empty_hypotheses": sum(not line for line in inputs["candidates"]),
                             "reference_file": str(inputs["reference_path"]), "device": device,
                             "comet_model": model_name, "comet": comet, "chrfpp": chrfpp,
                             "chrfpp_config": chrf_config, "term_success_rate": terminology["success_rate"],
                             "preferred_term_success_rate": terminology["preferred_success_rate"],
                             "terminology": terminology}}
    append_record(inputs["output_log"], record)
    rate = terminology["success_rate"]
    print(f"[RESULT] COMET={comet:.6f}, chrF++={chrfpp:.6f}, "
          f"용어 성공률={f'{rate:.2f}%' if rate is not None else 'NA'}", flush=True)
    print(f"[DONE] {inputs['output_log']}\n[용어 상세] {term_details}", flush=True)


if __name__ == "__main__":
    main()
