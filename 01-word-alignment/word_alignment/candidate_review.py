"""BGE candidate rankings and human review traces."""
from contextlib import ExitStack
import fcntl
import json

from .lab_common import atomic_text, finish_pairs, pair_key, read_rows, write_line
from .term_alignment import intersect
from .term_export import SCHEMA

LANGS = ("en", "ko")
TOP_K = 3


def pair_for(row, en, ko, kind, score, margin):
    pair = {"kind": "word" if len(en[0]["token_ids"]) == len(ko[0]["token_ids"]) == 1 else "phrase",
            "unit_type": kind, "score": score,
            "evidence": {"source": "bge_bidirectional_top3_human_review", "margin": margin,
                         "shared_subword_boundary": en[1] or ko[1], "canonical_reuses_score": False}}
    for lang, (unit, _) in zip(LANGS, (en, ko)):
        pair.update({lang: unit["surface"], lang + "_span": unit["span"], lang + "_unit": unit,
                     lang + "_token_ids": unit["token_ids"], lang + "_canonical": unit["canonical"]})
    return finish_pairs(row, [pair])[0]


def alternatives(row, features, args, baseline):
    import numpy as np
    from .lab_bge import pool_span

    selected = {pair_key(p): p for p in baseline}
    pairs, pool_status = [], {}
    for kind in args.term_types:
        groups = []
        for lang, feature in zip(LANGS, features):
            found, vectors = [], []
            for unit in row["term_units"][lang]:
                if unit["type"] != kind:
                    continue
                vector, shared = pool_span(feature, *unit["span"])
                pool_status[unit["unit_id"]] = vector is not None
                if vector is not None:
                    found.append((unit, shared))
                    vectors.append(vector)
            groups.append((found, vectors))
        (ens, ev), (kos, kv) = groups
        if not ev or not kv or max(len(ev), len(kv)) > args.max_span_candidates:
            continue
        matrix = np.stack(ev) @ np.stack(kv).T
        # Stable sorting preserves argmax's first-index tie break.
        forward = np.argsort(-matrix, axis=1, kind="stable")
        reverse = np.argsort(-matrix, axis=0, kind="stable")
        forward_rank = np.argsort(forward, axis=1) + 1
        reverse_rank = np.argsort(reverse, axis=0) + 1
        indices = {(i, int(j)) for i, js in enumerate(forward) for j in js[:TOP_K]}
        indices.update((int(i), j) for j in range(len(kos)) for i in reverse[:TOP_K, j])
        for i, j in sorted(indices, key=lambda ij: (-float(matrix[ij]), *ij)):
            score = float(matrix[i, j])
            margins = []
            for scores, units, index in ((matrix[i], kos, j), (matrix[:, j], ens, i)):
                span = units[index][0]["span"]
                different = [float(val) for val, unit in zip(scores, units) if not intersect(unit[0]["span"], span)]
                margins.append(score - max(different) if different else 2.)
            margin = min(margins)
            pair = pair_for(row, ens[i], kos[j], kind, score, margin)
            en_rank, ko_rank = int(forward_rank[i, j]), int(reverse_rank[i, j])
            blockers = [
                p["pair_id"] for p in baseline if pair_key(p) != pair_key(pair)
                and any(intersect(p[lang + "_span"], pair[lang + "_span"]) for lang in LANGS)
            ]
            mutual = en_rank == ko_rank == 1
            passes = score >= args.min_similarity and margin >= args.margin
            if pair_key(pair) in selected:
                reason = "selected"
            elif not mutual:
                reason = "not_mutual_best"
            elif score < args.min_similarity:
                reason = "below_similarity"
            elif margin < args.margin:
                reason = "below_margin"
            elif blockers:
                reason = "overlap"
            else:
                reason = "not_selected"
            pairs.append({"pair": pair, "en_to_ko_rank": en_rank, "ko_to_en_rank": ko_rank,
                          "baseline_reason": reason, "overlaps_baseline": blockers,
                          "review_eligible": passes or reason == "selected"})
    return pairs, pool_status


def export_bundle(path, aligned, prefix):
    names = ("review_candidates.jsonl", "review_template.jsonl")
    files = {name: prefix + name for name in names}
    template = path / files["review_template.jsonl"]
    with ExitStack() as stack:
        candidates = stack.enter_context(atomic_text(path / files["review_candidates.jsonl"], overwrite=True))
        pending = stack.enter_context(atomic_text(template, overwrite=True))
        for row in read_rows(aligned):
            trace = row.get("term_candidate_review", {})
            units = row.get("term_units", {})
            pooled = trace.get("pooled", {})
            record = {"line_id": row["line_id"], "method": row["method"], "ko": row["ko"], "en": row["en"],
                      "source_status": row["status"], "analysis_signature": row.get("term_analysis", {}).get("signature"),
                      "term_diagnostics": row.get("term_diagnostics", {}), "baseline_pairs": row.get("term_pairs", []),
                      "candidates": trace.get("candidates", []),
                      "units": {lang: [{"unit_id": u["unit_id"], "surface": u["surface"], "span": u["span"],
                                       "type": u["type"], "pooled": pooled.get(u["unit_id"], False)}
                                       for u in units.get(lang, [])] for lang in ("ko", "en")}}
            write_line(candidates, record)
            write_line(pending, {"line_id": row["line_id"], "complete": False, "selected_pair_ids": []})
    return files


def approved_pairs(path, reviews, output, overwrite):
    pending = dict(reviews)
    rows = pairs = 0
    with atomic_text(output, overwrite=overwrite) as out:
        for row in read_rows(path):
            ident = row["line_id"]
            review = pending.pop(ident, None)
            if review is None or review["complete"] is not True:
                continue
            listed = {item["pair"]["pair_id"]: item["pair"] for item in row["candidates"] if item["review_eligible"]}
            ids = review["selected_pair_ids"]
            if (not isinstance(ids, list) or any(not isinstance(i, str) or i not in listed for i in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError(f"{ident}: 해당 문장의 검토 가능 pair_id를 중복 없이 사용하세요")
            chosen = sorted((listed[i] for i in ids), key=pair_key)
            write_line(out, {"schema": SCHEMA, "line_id": ident, "method": row["method"],
                             "ko": row["ko"], "en": row["en"], "pairs": chosen,
                             "status": "aligned" if chosen else "no_term_alignment",
                             "source_status": row["source_status"], "term_diagnostics": row["term_diagnostics"],
                             "analysis_signature": row["analysis_signature"]})
            rows += 1
            pairs += len(chosen)
        if pending:
            raise ValueError(f"검토 입력에 이 실행의 문장이 아닌 ID가 있습니다: {min(pending)}")
        if not rows:
            raise ValueError("완료 표시한 문장이 없습니다")
    return rows, pairs


def approve_run(root, review_file, method=None, overwrite=False):
    root, review_file = root.resolve(), review_file.resolve()
    with (root / "details/.pipeline.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("정렬 실행이 끝난 뒤 승인하세요") from exc
        meta = json.loads((root / "details/run.json").read_text(encoding="utf-8"))
        if meta.get("project") != "word-alignment-v1.3" or meta.get("status") != "complete":
            raise ValueError("완료된 word-alignment v1.3 실행 폴더가 필요합니다")
        method = method or meta["primary_method"]
        candidate_file = root / meta["public_files"][method]["review_candidates.jsonl"]
        output = root / (("" if method == meta["primary_method"] else method + "-") + "term-pairs.jsonl")
        if review_file == output.resolve():
            raise ValueError("승인 입력과 출력은 다른 파일이어야 합니다")
        reviews = {}
        for row in read_rows(review_file):
            ident = row["line_id"]
            if type(ident) is not int or ident in reviews or type(row["complete"]) is not bool:
                raise ValueError(f"잘못되거나 중복된 검토 행: {ident}")
            reviews[ident] = row
        rows, pairs = approved_pairs(candidate_file, reviews, output, overwrite)
        print(f"저장: {output} · 검토 완료 문장 {rows:,}개 · 승인 쌍 {pairs:,}개")
    return output


