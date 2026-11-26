#!/usr/bin/env python3
"""관측된 번역 대응쌍의 표현만으로 복합 기본어와 중복 없는 용어 그룹을 구성합니다."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import unicodedata
from nlp_tokens import load_rules


from quality import DEFAULT_RULES, shape_flags, initialize_quality, record_occurrence, finalize_quality, compatible

ROOT = Path(__file__).resolve().parent
METHODS = ("3A_cvalue", "3B_evidence")
LANGUAGES = ("ko", "en")
BOUNDARY = "\x00"
END = object()
EN_TOKEN = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*|[^\s]", re.UNICODE)
STOP = {
    "ko": set("의 및 또는 등 것 수 본 해당 상기 이 그 저 후 전 중 때 대한 위한 관한 의한 통한 관련".split()),
    "en": set("a an the of and or to in on at by with for from as is are was were be been being this that these those".split()),
}
NOUN_TAGS = {"NNG", "NNP", "SL", "SH"}
NOMINAL_TAGS = NOUN_TAGS | {"NNB", "NR", "SN", "XPN", "XSN", "MM"}
SHARED_FILES = ("terms.jsonl", "candidate-scores.csv", "sources.jsonl", "core-comparison.csv",
                "comparison.csv", "summary.json", "summary.md", "rules.json", "normalization-links.csv",
                "boundary-review.jsonl", "sense-review.jsonl", "aligned-pairs.jsonl", "input-summary.json", "corpus-audit.jsonl")
METHOD_FILES = ("core_candidates.jsonl", "core_hierarchy.jsonl", "families.jsonl", "memberships.jsonl",
                "groups-simple-ko.json", "groups-simple-en.json", "groups-simple.csv",
                "group-statistics.csv", "omission_review.jsonl", "review_terms.jsonl",
                "unassigned_terms.jsonl", "summary.json", "summary.md", "excluded_members.jsonl", "excluded-members.csv",
                "pair-memberships.jsonl", "group-pairs.csv", "rare_compounds_review.jsonl")


def normalize(text):
    return unicodedata.normalize("NFC", text).casefold()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, values):
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def lexical(form):
    return any(char.isalnum() for char in form)


def build_trie(patterns):
    root = {}
    for pattern in patterns:
        node = root
        for token in pattern:
            node = node.setdefault(token, {})
        node[END] = pattern
    return root


def matches(sequence, trie):
    for start in range(len(sequence)):
        node = trie
        for end in range(start, len(sequence)):
            node = node.get(sequence[end])
            if node is None:
                break
            if END in node:
                yield start, end + 1, node[END]


def entropy(counter):
    total = sum(counter.values())
    return -sum((n / total) * math.log2(n / total) for n in counter.values()) if total else 0.0


def compute_scores(terms, frequencies, token_total, frequency_scope="corpus"):
    plausible = {parts for parts, term in terms.items() if term["shape_eligible"] and term["occurrence_frequency"] > 0}
    for longer in plausible:
        if len(longer) < 2:
            continue
        contained = {longer[start:end] for start in range(len(longer)) for end in range(start + 1, len(longer) + 1)
                     if end - start < len(longer) and longer[start:end] in plausible}
        for shorter in contained:
            terms[shorter]["_containing"].append(longer)
    for parts, term in terms.items():
        frequency = (term["aligned_containment_occurrence_frequency"] if frequency_scope == "observed"
                     else term["occurrence_frequency"])
        containers = term["_containing"]
        nested_mean = sum(frequencies[p] for p in containers) / len(containers) if containers else 0.0
        adjusted = max(0.0, frequency - nested_mean)
        # Original C-value uses log2(length), hence single-token values are zero.
        cvalue = math.log2(len(parts)) * adjusted
        splits = []
        if frequency and len(parts) > 1 and token_total > frequency:
            for cut in range(1, len(parts)):
                left, right = frequencies.get(parts[:cut], 0), frequencies.get(parts[cut:], 0)
                if left and right:
                    pmi = math.log2(frequency * token_total / (left * right))
                    npmi = pmi / -math.log2(frequency / token_total)
                    splits.append({"cut": cut, "pmi": round(pmi, 8), "npmi": round(max(-1.0, min(1.0, npmi)), 8)})
        term.update(c_value=round(cvalue, 8), adjusted_frequency=round(adjusted, 8),
            scoring_occurrence_frequency=frequency,
            association_scope="aligned_expression_token_stream" if frequency_scope == "observed" else "original_sentences",
            nesting_type_count=len(containers), nesting_mean_frequency=round(nested_mean, 8),
            nesting_samples=[terms[p]["term_id"] for p in sorted(containers)[:10]],
            min_npmi=min((item["npmi"] for item in splits), default=None), association_splits=splits,
            left_context_types=len(term["_left"]), right_context_types=len(term["_right"]),
            context_types=len(set(term["_left"]) | set(term["_right"])),
            left_context_entropy=round(entropy(term["_left"]), 8),
            right_context_entropy=round(entropy(term["_right"]), 8),
            context_samples={"left": term["_left"].most_common(5), "right": term["_right"].most_common(5)})


def public(term):
    return {key: value for key, value in term.items() if not key.startswith("_")}


def core_decision(term, method, args, protected=False):
    if protected:
        return (True, "user_protected") if term["occurrence_frequency"] else (False, "protected_unobserved_review")
    if not term.get("quality_eligible", True):
        return False, "quality:" + ",".join(term["quality_flags"])
    if term.get("normalized") in getattr(args, "active_rules", DEFAULT_RULES)["generic_anchors"].get(term.get("language"), []):
        return False, "generic_anchor_review"
    if not term["shape_eligible"]:
        return False, "shape_or_length_review"
    if term.get("boundary_rescue") and term["sentence_frequency"] >= args.min_atomic_sentences:
        return True, "boundary_completion_core"
    if term["token_count"] == 1:
        enough = term["sentence_frequency"] >= args.min_atomic_sentences
        return enough, "atomic_core" if enough else "low_atomic_support"
    extension_supported = (term.get("frequency_basis") == "exact_observed_pairs"
                           and term.get("aligned_containment_sentence_frequency", 0) >= args.min_compound_sentences)
    if term["sentence_frequency"] < args.min_compound_sentences and not extension_supported:
        return False, "low_compound_support"
    if term["c_value"] < args.min_cvalue:
        return False, "low_cvalue"
    if method == "3B_evidence":
        if term["min_npmi"] is None or term["min_npmi"] < args.min_npmi:
            return False, "weak_association"
        if term["context_types"] < args.min_context_types:
            return False, "limited_context"
    if term["sentence_frequency"] < args.min_compound_sentences:
        return True, "aligned_extension_support"
    return True, "compound_cvalue" if method == "3A_cvalue" else "compound_evidence"


def core_matches(parts, core_trie):
    return list(matches(parts, core_trie))


def choose_match(found, cores, lang):
    if not found:
        return None
    # Compound cores outrank atomic components. Among compounds prefer the most
    # specific span; C-value breaks ties rather than the frequency of a component.
    return min(found, key=lambda hit: (len(hit[2]) == 1, -len(hit[2]),
        -cores[hit[2]]["c_value"] if len(hit[2]) > 1 else 0,
        -(cores[hit[2]].get("dominant_partner_share") or 0),
        -cores[hit[2]].get("alignment_sentence_frequency", 0), hit[0] != 0, hit[0], hit[2]))


def decompose(parts, cores, trie, lang, allowed=None):
    starts = defaultdict(list)
    for start, end, core in matches(parts, trie):
        if allowed is None or core in allowed:
            starts[start].append((end, core))
    states = {len(parts): (0, [])}
    for start in reversed(range(len(parts))):
        covered, tail = states[start + 1]
        kind = "function" if parts[start] in STOP[lang] else "uncovered"
        best = (covered, [{"kind": kind, "tokens": [parts[start]]}] + tail)
        for end, core in starts[start]:
            covered, tail = states[end]
            candidate = (end - start + covered,
                [{"kind": "core", "term_id": cores[core]["term_id"], "term": cores[core]["term"],
                  "tokens": list(core)}] + tail)
            if (candidate[0], -len(candidate[1])) > (best[0], -len(best[1])):
                best = candidate
        states[start] = best
    segments = states[0][1]
    complete = not any(segment["kind"] == "uncovered" for segment in segments)
    return segments, complete


def build_method(terms, lang, method, args, protected):
    decisions = {parts: core_decision(term, method, args, parts in protected) for parts, term in terms.items()}
    cores = {parts: terms[parts] for parts, (selected, _) in decisions.items() if selected}
    trie = build_trie(cores)
    parents = {}
    for parts in cores:
        # Multiword cores remain independent of atomic words; only a proper
        # compound subcore may be a hierarchical parent.
        found = [hit for hit in core_matches(parts, trie) if 1 < len(hit[2]) < len(parts)
                 and compatible(cores[parts], cores[hit[2]], args.active_rules)]
        choice = choose_match(found, cores, lang)
        parents[parts] = choice[2] if choice else None
    roots = {}
    for parts in cores:
        current = parts
        while parents[current] is not None:
            current = parents[current]
        roots[parts] = current
    records, groups = [], defaultdict(list)
    for parts, term in terms.items():
        selected, selection_reason = decisions[parts]
        all_found = core_matches(parts, trie)
        found = [hit for hit in all_found if compatible(term, cores[hit[2]], args.active_rules)]
        suggested = choose_match(found, cores, lang)
        included = term["quality_eligible"] or (selected and selection_reason == "user_protected")
        if not included:
            found = []
        chosen = (0, len(parts), parts) if selected else choose_match(found, cores, lang)
        anchor = chosen[2] if chosen else None
        root = roots[anchor] if anchor else None
        segments, complete = decompose(parts, cores, trie, lang, {hit[2] for hit in found})
        if selected:
            status = "preserved_core"
        elif (term["shape_eligible"] and term["sentence_frequency"] >= args.min_compound_sentences
              and included and complete and any(s["kind"] == "core" for s in segments)):
            status = "omission_review_candidate"
        else:
            status = "separate_entry_review"
        record = {"term_id": term["term_id"], "language": lang, "term": term["term"],
            "normalized": term["normalized"], "token_count": len(parts), "unit_type": term["unit_type"],
            "is_core_candidate": selected, "selection_reason": selection_reason,
            "quality_status": term["quality_status"], "quality_flags": term["quality_flags"],
            "sense_tags": term["sense_tags"], "included_in_clean_groups": included,
            "suggested_core_id": cores[suggested[2]]["term_id"] if suggested else None,
            "blocked_semantic_core_ids": sorted({cores[hit[2]]["term_id"] for hit in all_found if hit not in found and not compatible(term, cores[hit[2]], args.active_rules)}),
            "immediate_core_id": cores[anchor]["term_id"] if anchor else None,
            "representative_term_id": cores[root]["term_id"] if root else None,
            "other_matching_core_ids": sorted({cores[hit[2]]["term_id"] for hit in found
                                                if hit[2] != anchor}),
            "status": status, "decomposition": segments, "lexically_covered": complete,
            "rare_compound_review": (len(parts) > 1 and term["quality_eligible"]
                                     and term["sentence_frequency"] < args.min_compound_sentences),
            "alignment_sentence_frequency": term.get("alignment_sentence_frequency"),
            "frequency_basis": term.get("frequency_basis", "original_sentences"),
            "dominant_partner_share": term.get("dominant_partner_share"),
            "translation_verified": False, "can_omit_from_glossary": None}
        if root is not None and root != parts and not complete:
            # Apply the covered-expansion policy before any membership or pair export.
            record.update(representative_term_id=None, immediate_core_id=None,
                          status="separate_entry_review", selection_reason="uncovered_tokens_in_expansion")
            root = None
        records.append(record)
        if root:
            groups[root].append(record)
    families = []
    for root, members in sorted(groups.items()):
        members.sort(key=lambda r: (r["term_id"] != cores[root]["term_id"], r["token_count"], r["normalized"]))
        families.append({"language": lang, "unit_type": cores[root]["unit_type"], "representative": public(cores[root]),
            "members": members, "member_count": len(members),
            "compound_core_count": sum(m["is_core_candidate"] and m["token_count"] > 1 for m in members),
            "semantic_relation_verified": False})
    hierarchy = [{"term_id": cores[p]["term_id"], "language": lang, "term": cores[p]["term"],
        "unit_type": cores[p]["unit_type"], "parent_core_id": cores[parents[p]]["term_id"] if parents[p] else None,
        "family_representative_id": cores[roots[p]]["term_id"], "relation": "lexical_containment"}
        for p in cores]
    return records, families, hierarchy


def export_method(path, terms_by_id, records, families, hierarchy, method):
    path.mkdir()
    core_ids = {r["term_id"] for r in records if r["is_core_candidate"]}
    enriched = ({**public(terms_by_id[r["term_id"]]), **r} for r in records if r["term_id"] in core_ids)
    write_jsonl(path / "core_candidates.jsonl", enriched)
    write_jsonl(path / "core_hierarchy.jsonl", hierarchy)
    write_jsonl(path / "families.jsonl", families)
    write_jsonl(path / "memberships.jsonl", records)
    for name, predicate in (
        ("omission_review.jsonl", lambda r: r["status"] == "omission_review_candidate"),
        ("review_terms.jsonl", lambda r: r["status"] == "separate_entry_review"),
        ("unassigned_terms.jsonl", lambda r: r["representative_term_id"] is None),
        ("rare_compounds_review.jsonl", lambda r: r["rare_compound_review"])):
        write_jsonl(path / name, ({**public(terms_by_id[r["term_id"]]), **r} for r in records if predicate(r)))
    excluded = [{**public(terms_by_id[r["term_id"]]), **r} for r in records if not r["included_in_clean_groups"]]
    write_jsonl(path / "excluded_members.jsonl", excluded)
    write_csv(path / "excluded-members.csv", ("language", "term", "normalized", "quality_flags", "suggested_core_id"),
              ({key: json.dumps(r[key], ensure_ascii=False) if key == "quality_flags" else r[key]
                for key in ("language", "term", "normalized", "quality_flags", "suggested_core_id")} for r in excluded))
    for lang in LANGUAGES:
        write_json(path / f"groups-simple-{lang}.json", {
            kind: {family["representative"]["term"]: [m["term"] for m in family["members"]]
                   for family in families if family["language"] == lang and family["unit_type"] == kind}
            for kind in ("nominal","verbal")})
    simple_rows = []
    for family in families:
        representative = family["representative"]
        simple_rows.append({"language": family["language"], "unit_type": family["unit_type"], "representative": representative["term"],
            "normalized": representative["normalized"], "word_count": representative["word_count"],
            "token_count": representative["token_count"], "character_count": representative["character_count"],
            "member_count": len(family["members"]), "extension_count": len(family["members"]) - 1,
            "compound_core_count": family["compound_core_count"],
            "members": json.dumps([m["term"] for m in family["members"]], ensure_ascii=False)})
    write_csv(path / "groups-simple.csv", ("language", "unit_type", "representative", "normalized", "word_count", "token_count",
        "character_count", "member_count", "extension_count", "compound_core_count", "members"), simple_rows)
    distributions = []
    for lang in LANGUAGES:
        for kind in ("nominal","verbal"):
            for unit in ("word_count", "token_count", "character_count"):
                buckets = defaultdict(lambda: [0, 0])
                for row in simple_rows:
                    if row["language"] == lang and row["unit_type"] == kind:
                        buckets[row[unit]][0] += 1
                        buckets[row[unit]][1] += row["member_count"]
                distributions.extend({"language": lang, "unit_type": kind, "length_unit": unit, "length": length,
                    "representatives": values[0], "members": values[1]} for length, values in sorted(buckets.items()))
    write_csv(path / "group-statistics.csv", ("language", "unit_type", "length_unit", "length", "representatives", "members"), distributions)
    summary = {"method": method, "languages": {}}
    for lang in LANGUAGES:
        local = [r for r in records if r["language"] == lang]
        local_families = [f for f in families if f["language"] == lang]
        summary["languages"][lang] = {"terms": len(local), "core_candidates": sum(r["is_core_candidate"] for r in local),
            "compound_cores": sum(r["is_core_candidate"] and r["token_count"] > 1 for r in local),
            "families": len(local_families), "singleton_families": sum(f["member_count"] == 1 for f in local_families),
            "unassigned_terms": sum(r["representative_term_id"] is None for r in local),
            "statuses": dict(Counter(r["status"] for r in local)), "confirmed_omissions": 0,
            "excluded_from_clean_groups": sum(not r["included_in_clean_groups"] for r in local),
            "semantic_conflict_links": sum(len(r["blocked_semantic_core_ids"]) for r in local)}
    write_json(path / "summary.json", summary)
    lines = [f"# {method}", "", "| 언어 | 전체 후보 | 기본어 후보 | 복합 기본어 | 그룹 | 단독 그룹 | 미배정 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for lang, stats in summary["languages"].items():
        lines.append(f"| {lang} | {stats['terms']} | {stats['core_candidates']} | {stats['compound_cores']} | {stats['families']} | {stats['singleton_families']} | {stats['unassigned_terms']} |")
    lines += ["", "그룹화는 어휘 포함 관계이며 의미 동등성·번역 합성 가능성을 보장하지 않습니다."]
    (path / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


