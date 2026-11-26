"""Statistics over selected alignment expressions, without sentence NLP."""
from collections import Counter
import re

from quality import senses


WORD = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)


def observed_statistics(terms, rows, lang, max_tokens, sample_count, rules):
    """Keep exact pair support separate from containment in aligned expressions.

    The association corpus consists of the aligned expressions, separated by
    boundaries. A phrase may contribute to component counts, but never creates
    a new candidate. Repeated normalized terms in one row count once for sentence
    support, while each actual pair remains an occurrence.
    """
    surface_map = {}
    for term in terms.values():
        term.update(sentence_frequency=term["alignment_sentence_frequency"],
                    occurrence_frequency=term["alignment_pair_occurrences"],
                    sample_occurrences=[], _left=Counter(), _right=Counter(), _containing=[],
                    _containment_rows=set(), _sample_rows=set(), _source_senses={},
                    _ambiguous_spans=0, _literal_sources=0)
        for variant in term["surface_variants"]:
            surface_map[tuple(term["tokens"])] = term
    frequencies = Counter()
    token_total = 0
    for parts, source in terms.items():
        weight = source["alignment_pair_occurrences"]
        token_total += sum(any(c.isalnum() for c in part) for part in parts) * weight
        for start in range(len(parts)):
            for end in range(start + 1, min(len(parts), start + max_tokens) + 1):
                if not any(c.isalnum() for c in parts[end - 1]):
                    break
                pattern = parts[start:end]
                frequencies[pattern] += weight
                target = terms.get(pattern)
                if target is None:
                    continue
                target["_containment_rows"].update(source["_candidate_rows"])
                if start and any(c.isalnum() for c in parts[start - 1]):
                    target["_left"][parts[start - 1]] += weight
                if end < len(parts) and any(c.isalnum() for c in parts[end]):
                    target["_right"][parts[end]] += weight
    for row in rows:
        sentence = row[lang + "_sentence"]
        parallel = row[("en" if lang == "ko" else "ko") + "_sentence"]
        for index, pair in enumerate(row["alignment_pairs"], 1):
            surface = pair[lang]
            term = surface_map[tuple(pair[lang+"_unit"]["tokens"])]
            positions = [pair[lang+"_unit"]["span"][0]]
            # Simple JSONL has no offsets. Multiple literal occurrences are
            # ambiguous; do not guess one and call it the original aligned span.
            unique = len(positions) == 1
            term["_literal_sources"] += 1
            term["_ambiguous_spans"] += int(not unique)
            if unique:
                start, end = positions[0], positions[0] + len(surface)
                left = list(WORD.finditer(sentence[max(0, start - 60):start]))
                right = WORD.search(sentence[end:end + 60])
                if left:
                    term["_left"][left[-1].group().casefold()] += 1
                if right:
                    term["_right"][right.group().casefold()] += 1
                context = sentence[max(0, start - 50):end + 50] + " " + parallel
            else:
                context = sentence + " " + parallel
            for key, label in senses(term["tokens"], lang, context, rules).items():
                term["_source_senses"].setdefault(key, Counter())[label] += 1
            if len(term["sample_occurrences"]) < sample_count and row["source_line"] not in term["_sample_rows"]:
                term["sample_occurrences"].append({"source_line": row["source_line"],
                    "line_id": row["line_id"], "pair_index": index, "text": surface,
                    "character_span": [positions[0], positions[0] + len(surface)] if unique else None,
                    "span_basis": "upstream_character_span"})
                term["_sample_rows"].add(row["source_line"])
    for parts, term in terms.items():
        term.update(aligned_containment_sentence_frequency=len(term["_containment_rows"]),
                    aligned_containment_occurrence_frequency=frequencies[parts],
                    frequency_basis="exact_observed_pairs",
                    context_basis="aligned_expression_components_and_literal_neighbors",
                    corpus_sentence_frequency=None, corpus_occurrence_frequency=None,
                    corpus_quality_flags=[], corpus_audit_enabled=False)
    return frequencies, token_total


def finalize_observed_quality(terms, lang, args, rules):
    """Use expression POS and supplied literal evidence, not a re-tokenization veto."""
    for parts, term in terms.items():
        flags = list(term["candidate_flags"])
        if term["over_token_limit"]:
            flags.append("over_token_limit")
        tags = senses(parts, lang, rules=rules)
        for key, counts in term["_source_senses"].items():
            label, count = counts.most_common(1)[0]
            tags[key] = label if label != "unresolved" and count / sum(counts.values()) >= args.min_sense_purity else "unresolved"
        if "unresolved" in tags.values():
            flags.append("ambiguous_sense_review")
        term.update(quality_flags=sorted(set(flags)), quality_eligible=not flags,
                    quality_status="review" if flags else "clean_candidate",
                    quality_basis="upstream_unit_grammar_and_observed_evidence",
                    sense_tags=tags, sense_evidence={k: dict(v) for k, v in term["_source_senses"].items()},
                    source_clean_frequency=None, source_clean_ratio=None,
                    source_closed_word_ratio=None, attached_content_ratio=None,
                    completion_target_id=None, completion_ratio=None, boundary_rescue=False,
                    boundary_warnings=["ambiguous_source_span"] if term["_ambiguous_spans"] else [],
                    ambiguous_source_span_count=term["_ambiguous_spans"])


def attach_bilingual_evidence(all_terms):
    by_id = {t["term_id"]: t for local in all_terms.values() for t in local.values()}
    for term in by_id.values():
        counts = term["_partner_counts"]
        total = sum(counts.values())
        top = sorted(counts, key=lambda partner: (-counts[partner], partner))
        term["dominant_partner_share"] = counts[top[0]] / total if total else None
        term["dominant_partner_term_id"] = top[0] if top else None
        term["clean_compound_partner_occurrences"] = sum(
            count for partner, count in counts.items()
            if by_id[partner]["token_count"] > 1 and by_id[partner]["quality_eligible"])
        term["bilingual_evidence_verified"] = False
