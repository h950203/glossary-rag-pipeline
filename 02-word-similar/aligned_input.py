"""Observed-pair input and bilingual provenance for monolingual lexical families."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math


LANGUAGES = ("ko", "en")


def read_input(path, limit):
    fingerprint = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    rows = []
    raw = {lang: defaultdict(set) for lang in LANGUAGES}
    identities = set()
    with path.open(encoding="utf-8-sig") as reader:
        for number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSON 객체가 필요합니다")
                if value.get("schema") != "term-pairs-v1.1":
                    raise ValueError("word-alignment_v1.1의 term-pairs.jsonl을 입력하세요(기존 simple JSONL은 v1에서 처리)")
                line_id = value.get("line_id")
                if isinstance(line_id, bool) or not isinstance(line_id, int) or line_id < 1:
                    raise ValueError("line_id는 양의 정수여야 합니다")
                method = value.get("method", "unknown")
                if not isinstance(method, str) or not method.strip():
                    raise ValueError("method는 비어 있지 않은 문자열이어야 합니다")
                status = value.get("status", "unknown")
                if not isinstance(status, str):
                    raise ValueError("status는 문자열이어야 합니다")
                identity = (method, line_id)
                if identity in identities:
                    raise ValueError(f"같은 method/line_id가 중복되었습니다: {identity}")
                identities.add(identity)
                for lang in LANGUAGES:
                    if not isinstance(value.get(lang), str):
                        raise ValueError(f"{lang} 원문 문자열이 필요합니다")
                pairs = value.get("pairs")
                if not isinstance(pairs, list):
                    raise ValueError("pairs 배열이 필요합니다")
                for index, pair in enumerate(pairs, 1):
                    if not isinstance(pair, dict):
                        raise ValueError(f"pairs[{index}]는 JSON 객체여야 합니다")
                    for lang in LANGUAGES:
                        surface = pair.get(lang)
                        if not isinstance(surface, str) or not surface.strip():
                            raise ValueError(f"pairs[{index}].{lang} 용어 문자열이 필요합니다")
                        if surface not in value[lang]:
                            raise ValueError(f"pairs[{index}].{lang} 용어가 원문에 없습니다: {surface!r}")
                    score = pair.get("score")
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        raise ValueError(f"pairs[{index}].score는 유한한 숫자여야 합니다")
                    if pair.get("kind") not in {"word", "phrase"}:
                        raise ValueError(f"pairs[{index}].kind는 word 또는 phrase여야 합니다")
                    unit_type = pair.get("unit_type")
                    if unit_type not in {"nominal", "verbal"}:
                        raise ValueError(f"pairs[{index}].unit_type이 필요합니다")
                    for lang in LANGUAGES:
                        unit = pair.get(lang + "_unit")
                        if not isinstance(unit, dict) or unit.get("type") != unit_type or unit.get("shape_valid") is not True:
                            raise ValueError(f"pairs[{index}].{lang}_unit 분석 정보가 필요합니다")
                        start, end = unit.get("span", [None,None])
                        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(value[lang]):
                            raise ValueError("용어의 원문 좌표가 잘못되었습니다")
                        if unit.get("surface") != pair[lang] or value[lang][start:end] != pair[lang]:
                            raise ValueError("용어 좌표와 원문이 다릅니다")
                        if pair.get(lang + "_span") != [start,end]:
                            raise ValueError("대응쌍 좌표와 분석 좌표가 다릅니다")
                        tokens = unit.get("tokens")
                        if not isinstance(tokens,list) or not tokens or any(not isinstance(t,str) or not t.strip() for t in tokens):
                            raise ValueError("정규화 토큰 배열이 필요합니다")
                        if (unit.get("normalized", " ".join(tokens)) != " ".join(tokens)
                                or not isinstance(unit.get("canonical"), str) or not unit["canonical"].strip()
                                or pair.get(lang+"_canonical") != unit["canonical"]):
                            raise ValueError("기본형과 정규화 토큰이 다릅니다")
                        if unit.get("analysis_schema") != "term-units-v1.1" or not unit.get("analysis_signature"):
                            raise ValueError("지원하지 않는 분석 스키마입니다")
                        if (not isinstance(unit.get("pos"),list) or not unit["pos"]
                                or any(not isinstance(tag,str) for tag in unit["pos"])):
                            raise ValueError("원문 품사 배열이 필요합니다")
                        if unit_type == "verbal" and len(tokens) != 1:
                            raise ValueError("동사는 하나의 전체 기본형으로 전달해야 합니다")
                        if unit["analysis_signature"] != value.get("analysis_signature"):
                            raise ValueError("행·용어 분석 서명이 다릅니다")
                        raw[lang][(unit_type, pair[lang], tuple(tokens))].add(number)
                rows.append({"source_line": number, "line_id": line_id, "row_idx": line_id,
                             "method": method, "status": status,
                             "schema": value["schema"], "analysis_signature": value.get("analysis_signature"),
                             "source_status": value.get("source_status"), "term_diagnostics": value.get("term_diagnostics", {}),
                             "ko_sentence": value["ko"], "en_sentence": value["en"],
                             "ko_sent_id": value.get("ko_sent_id"), "en_sent_id": value.get("en_sent_id"),
                             "alignment_pairs": pairs})
            except (ValueError, TypeError) as exc:
                raise ValueError(f"입력 {number}행: {exc}") from exc
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("입력에 유효한 문장 행이 없습니다")
    if not any(raw[lang] for lang in LANGUAGES):
        raise ValueError("입력에 관측된 용어쌍이 없습니다(pairs가 모두 비어 있습니다)")
    return rows, raw, fingerprint.hexdigest()


def input_summary(rows, raw):
    return {"sentence_rows": len(rows),
            "rows_with_pairs": sum(bool(row["alignment_pairs"]) for row in rows),
            "rows_without_pairs": sum(not row["alignment_pairs"] for row in rows),
            "pair_occurrences": sum(len(row["alignment_pairs"]) for row in rows),
            "unique_surface_pairs": len({(p["ko"], p["en"]) for r in rows for p in r["alignment_pairs"]}),
            "unique_surfaces": {lang: len(raw[lang]) for lang in LANGUAGES},
            "upstream_methods": dict(Counter(row["method"] for row in rows)),
            "statuses": dict(Counter(row["status"] for row in rows)),
            "kinds": dict(Counter(p["kind"] for row in rows for p in row["alignment_pairs"]))}


def build_pair_inventory(rows, all_terms, sample_count):
    surface_ids, by_id = {}, {}
    for lang in LANGUAGES:
        surface_ids[lang] = {}
        for term in all_terms[lang].values():
            by_id[term["term_id"]] = term
            term.update(alignment_pair_occurrences=0, sample_alignment_pairs=[],
                        _alignment_rows=set(), _aligned_partner_ids=set(), _sample_alignment_rows=set(), _partner_counts=Counter())
            for variant in term["surface_variants"]:
                surface_ids[lang][(term["unit_type"], variant["surface"], tuple(term["tokens"]))] = term["term_id"]
    merged = {}
    for row in rows:
        for index, pair in enumerate(row["alignment_pairs"], 1):
            ko_id, en_id = (surface_ids[lang][(pair["unit_type"], pair[lang], tuple(pair[lang+"_unit"]["tokens"]))] for lang in LANGUAGES)
            key = (ko_id, en_id)
            if key not in merged:
                pair_id = "pair_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
                merged[key] = {"pair_id": pair_id, "ko_term_id": ko_id, "en_term_id": en_id,
                               "ko": by_id[ko_id]["term"], "en": by_id[en_id]["term"],
                               "occurrence_count": 0, "score_min": pair["score"], "score_max": pair["score"],
                               "sample_sources": [], "translation_verified": False, "unit_type": pair["unit_type"],
                               "_rows": set(), "_score_sum": 0.0, "_kinds": Counter(), "_variants": {}}
            target = merged[key]
            target["occurrence_count"] += 1
            target["_rows"].add(row["source_line"])
            target["_score_sum"] += pair["score"]
            target["score_min"] = min(target["score_min"], pair["score"])
            target["score_max"] = max(target["score_max"], pair["score"])
            target["_kinds"][pair["kind"]] += 1
            surface_pair = (pair["ko"], pair["en"])
            variant = target["_variants"].setdefault(surface_pair, {"ko": pair["ko"], "en": pair["en"],
                                                                  "occurrence_count": 0, "_rows": set()})
            variant["occurrence_count"] += 1
            variant["_rows"].add(row["source_line"])
            sample = {"source_line": row["source_line"], "line_id": row["line_id"],
                      "method": row["method"], "pair_index": index, **{k:pair[k] for k in ("ko","en","kind","score")}}
            if len(target["sample_sources"]) < sample_count:
                target["sample_sources"].append(sample)
            for own_id, partner_id in ((ko_id, en_id), (en_id, ko_id)):
                term = by_id[own_id]
                term["alignment_pair_occurrences"] += 1
                term["_partner_counts"][partner_id] += 1
                term["_alignment_rows"].add(row["source_line"])
                term["_aligned_partner_ids"].add(partner_id)
                if (len(term["sample_alignment_pairs"]) < sample_count
                        and row["source_line"] not in term["_sample_alignment_rows"]):
                    term["sample_alignment_pairs"].append({"pair_id": target["pair_id"], **sample})
                    term["_sample_alignment_rows"].add(row["source_line"])
    for term in by_id.values():
        term["alignment_sentence_frequency"] = len(term["_alignment_rows"])
        term["aligned_partner_count"] = len(term["_aligned_partner_ids"])
        if term["alignment_sentence_frequency"] != term["candidate_sentence_frequency"]:
            raise ValueError(f"정규화 후보와 원본 대응쌍 출처가 다릅니다: {term['term_id']}")
    output = []
    for key, target in sorted(merged.items()):
        record = {k: v for k, v in target.items() if not k.startswith("_")}
        record.update(sentence_frequency=len(target["_rows"]),
                      score_mean=target["_score_sum"] / target["occurrence_count"],
                      kind_counts=dict(target["_kinds"]), surface_pairs=[])
        for surface_pair, value in sorted(target["_variants"].items()):
            record["surface_pairs"].append({k: v for k, v in value.items() if not k.startswith("_")}
                                            | {"sentence_frequency": len(value["_rows"])})
        output.append(record)
    return output


def export_pair_memberships(path, pairs, memberships, terms):
    """Join both existing memberships to each observed pair; never cross-product."""
    counts = Counter({"both_assigned": 0, "partially_assigned": 0, "unassigned": 0})
    observed = {(p["ko_term_id"], p["en_term_id"]) for p in pairs}
    columns = ("pair_id", "unit_type", "ko", "en", "ko_representative", "en_representative", "assignment_status",
               "representatives_observed_as_pair", "ko_status", "en_status", "ko_quality_flags", "en_quality_flags",
               "sentence_frequency", "occurrence_count", "score_mean", "surface_pairs")
    with (path / "pair-memberships.jsonl").open("w", encoding="utf-8") as jsonl, \
         (path / "group-pairs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for pair in pairs:
            sides = {}
            for lang in LANGUAGES:
                term_id = pair[lang + "_term_id"]
                member = memberships[term_id]
                representative_id = member["representative_term_id"]
                sides[lang] = {"term_id": term_id, "term": terms[term_id]["term"],
                               "representative_term_id": representative_id,
                               "representative": terms[representative_id]["term"] if representative_id else None,
                               "is_core_candidate": member["is_core_candidate"], "status": member["status"],
                               "quality_flags": member["quality_flags"],
                               "lexically_covered": member["lexically_covered"]}
            assigned = sum(side["representative_term_id"] is not None for side in sides.values())
            status = ("unassigned", "partially_assigned", "both_assigned")[assigned]
            representative_pair = tuple(sides[lang]["representative_term_id"] for lang in LANGUAGES)
            seen = representative_pair in observed
            counts[status] += 1
            record = {"pair_id": pair["pair_id"], "unit_type": pair["unit_type"], **sides, "assignment_status": status,
                      "representatives_observed_as_pair": seen,
                      "sentence_frequency": pair["sentence_frequency"], "occurrence_count": pair["occurrence_count"],
                      "score_mean": pair["score_mean"], "surface_pairs": pair["surface_pairs"],
                      "sample_sources": pair["sample_sources"], "translation_verified": False,
                      "can_omit_from_glossary": None}
            jsonl.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            writer.writerow({"pair_id": pair["pair_id"], "unit_type": pair["unit_type"], "ko": pair["ko"], "en": pair["en"],
                             "ko_representative": sides["ko"]["representative"],
                             "en_representative": sides["en"]["representative"], "assignment_status": status,
                             "representatives_observed_as_pair": seen,
                             "ko_status": sides["ko"]["status"], "en_status": sides["en"]["status"],
                             "ko_quality_flags": json.dumps(sides["ko"]["quality_flags"], ensure_ascii=False),
                             "en_quality_flags": json.dumps(sides["en"]["quality_flags"], ensure_ascii=False),
                             "sentence_frequency": pair["sentence_frequency"], "occurrence_count": pair["occurrence_count"],
                             "score_mean": pair["score_mean"],
                             "surface_pairs": json.dumps(pair["surface_pairs"], ensure_ascii=False)})
    return {"normalized_pairs": len(pairs), "pair_occurrences": sum(p["occurrence_count"] for p in pairs), **counts}
