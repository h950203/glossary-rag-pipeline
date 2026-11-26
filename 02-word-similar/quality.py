"""Interpretable candidate quality, boundary completion and sense-link guards."""
from collections import Counter
import re

from nlp_tokens import load_rules

DEFAULT_RULES = load_rules()
EN_ALLOWED = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VBG", "VBN", "IN", "DT", "CD", "POS"}
KO_ALLOWED = {"NNG", "NNP", "SL", "SH", "NNB", "NR", "SN", "XPN", "XSN", "MM"}
KO_HEAD = {"NNG", "NNP", "SL", "SH"}


def shape_flags(parsed, lang, rules=None):
    rules = rules or DEFAULT_RULES
    if not parsed or any(not any(c.isalnum() for c in t[0]) for t in parsed):
        return ["punctuation_or_empty"]
    parts, tags = [t[0] for t in parsed], [t[1] for t in parsed]
    flags = []
    if lang == "en":
        if not tags[-1].startswith("NN"):
            flags.append("non_nominal_head")
        if any(tag not in EN_ALLOWED for tag in tags):
            flags.append("clause_or_non_nominal_structure")
    elif tags[-1] not in KO_HEAD or any(tag not in KO_ALLOWED for tag in tags):
        flags.append("non_nominal_or_incomplete_boundary")
    if parts[0] in rules["bad_starts"][lang] or parts[-1] in rules["bad_ends"][lang]:
        flags.append("discourse_boundary")
    normalized = " ".join(parts)
    if normalized in rules["bad_expressions"][lang]:
        flags.append("incomplete_or_contextual_expression")
    if len(parts) > 1 and parts[0] in rules["incomplete_unit_prefixes"][lang]:
        flags.append("incomplete_unit_prefix")
    if any(re.search(pattern, normalized, re.I) for pattern in rules["meta_patterns"][lang]):
        flags.append("research_meta_phrase")
    if not any(any(c.isalpha() for c in p) for p in parts):
        flags.append("numeric_only")
    return sorted(set(flags))


def senses(parts, lang, context="", rules=None):
    rules = rules or DEFAULT_RULES
    text = " ".join(parts)
    out = {}
    for rule in rules["sense_rules"]:
        if rule["language"] != lang or not re.search(rule["trigger"], text, re.I):
            continue
        # The expression itself has priority over surrounding paper vocabulary.
        own = [label for label, patterns in rule["labels"].items() if any(re.search(p, text, re.I) for p in patterns)]
        contextual = [label for label, patterns in rule["labels"].items() if any(re.search(p, context, re.I) for p in patterns)]
        found = own or contextual
        out[rule["key"]] = found[0] if len(found) == 1 else "unresolved" if len(found) > 1 else rule["default"]
    return out


def compatible(child, anchor, rules=None, check_scope=True):
    rules = rules or DEFAULT_RULES
    left, right = child.get("sense_tags", {}), anchor.get("sense_tags", {})
    for key in left.keys() & right.keys():
        if left[key] != right[key] or "unresolved" in (left[key], right[key]):
            return False
    if check_scope and child.get("tokens") and anchor.get("tokens"):
        head, parent_head = child["tokens"][-1], anchor["tokens"][-1]
        if head != parent_head and head in rules["independent_heads"].get(child.get("language"), []):
            return False
    return True


def initialize_quality(term):
    term.update(_source_clean=0, _closed_word=0, _left_attached=Counter(), _right_attached=Counter(),
                _source_senses={}, source_pos_heads={})


def record_occurrence(term, parsed, start, end, sentence, parallel, lang, rules):
    clean = not shape_flags(parsed[start:end], lang, rules)
    # Reject units cut out of rate expressions such as 5회 이상/주 섭취.
    before = sentence[max(0, parsed[start][2] - 20):parsed[start][2]]
    if re.search(r"(?:회|times|\d)\s*(?:이상|이하)?\s*/\s*$", before, re.I):
        clean = False
    term["_source_clean"] += int(clean)
    head = parsed[end - 1][1]
    term["source_pos_heads"][head] = term["source_pos_heads"].get(head, 0) + 1
    if lang == "ko":
        left_joined = start and parsed[start - 1][3] == parsed[start][2] and parsed[start - 1][1] in KO_ALLOWED
        right_joined = end < len(parsed) and parsed[end - 1][3] == parsed[end][2] and parsed[end][1] in KO_ALLOWED
        if left_joined:
            term["_left_attached"][parsed[start - 1][0]] += 1
        if right_joined:
            term["_right_attached"][parsed[end][0]] += 1
        surface = sentence[parsed[start][2]:parsed[end - 1][3]]
        term["_closed_word"] += int(not left_joined and not right_joined and not any(c.isspace() for c in surface))
    context = sentence[max(0, parsed[start][2] - 50):min(len(sentence), parsed[end - 1][3] + 50)] + " " + parallel
    for key, label in senses(term["tokens"], lang, context, rules).items():
        term["_source_senses"].setdefault(key, Counter())[label] += 1


def finalize_quality(terms, lang, args, rules):
    for parts, term in terms.items():
        frequency = term["occurrence_frequency"]
        flags = list(term.get("candidate_flags", []))
        ratio = term["_source_clean"] / frequency if frequency else 0.0
        if not frequency:
            flags.append("unobserved_token_sequence" if not term["over_token_limit"] else "over_token_limit")
        elif ratio < args.min_clean_ratio:
            flags.append("low_nominal_source_support")
        attached = max(sum(term["_left_attached"].values()), sum(term["_right_attached"].values())) / frequency if frequency else 0
        if lang == "ko" and frequency >= args.min_atomic_sentences and attached >= args.fixed_extension_ratio:
            flags.append("inside_lexical_word")
        term.update(source_clean_frequency=term["_source_clean"], source_clean_ratio=round(ratio, 6),
                    source_closed_word_ratio=round(term["_closed_word"] / frequency, 6) if frequency else 0.0,
                    attached_content_ratio=round(attached, 6), sense_tags=senses(parts, lang, rules=rules),
                    sense_evidence={key: dict(counts) for key, counts in term["_source_senses"].items()})
        for key, counts in term["_source_senses"].items():
            ranked = counts.most_common()
            if ranked:
                label, count = ranked[0]
                if label == "unresolved" or count / sum(counts.values()) < args.min_sense_purity:
                    term["sense_tags"][key] = "unresolved"
                else:
                    term["sense_tags"][key] = label
        if "unresolved" in term["sense_tags"].values():
            flags.append("ambiguous_sense_review")
        term["quality_flags"] = sorted(set(flags))
        term["completion_target_id"] = None
        term["completion_ratio"] = None
        term["boundary_rescue"] = False
        term["boundary_warnings"] = []
    # A statistical closure guard, not a universal semantic completeness test.
    targets = {}
    for parts, term in terms.items():
        frequency = term["occurrence_frequency"]
        if frequency < args.min_atomic_sentences or not term["shape_eligible"]:
            continue
        choices = []
        for longer in term["_containing"]:
            extension = terms[longer]
            if (extension["shape_eligible"] and extension["source_clean_ratio"] >= args.min_clean_ratio
                    and not extension["quality_flags"] and compatible(term, extension, rules, check_scope=False)
                    and extension["occurrence_frequency"] >= args.min_atomic_sentences):
                ratio = extension["occurrence_frequency"] / frequency
                if ratio >= args.fixed_extension_ratio:
                    choices.append((longer, ratio))
        if choices:
            if "inside_lexical_word" in term["quality_flags"]:
                # Finish a split lexical word before absorbing additional free
                # modifiers. Remaining attached fragments are excluded above.
                best, ratio = min(choices, key=lambda item: (terms[item[0]]["source_closed_word_ratio"] < args.fixed_extension_ratio,
                                                             len(item[0]), -item[1], item[0]))
            else:
                best, ratio = min(choices, key=lambda item: (-len(item[0]), -item[1], item[0]))
            term["completion_target_id"], term["completion_ratio"] = terms[best]["term_id"], round(ratio, 6)
            if (len(parts) == 1 or term["source_closed_word_ratio"] >= args.fixed_extension_ratio) and term["attached_content_ratio"] < args.fixed_extension_ratio:
                # A frequent collocation alone must not erase complete atomic
                # nouns such as regression and analysis.
                term["boundary_warnings"].append("fixed_collocation_warning")
            else:
                term["quality_flags"] = sorted(set(term["quality_flags"] + ["fixed_extension_review"]))
                targets[parts] = best
    for parts in targets:
        end, seen = targets[parts], {parts}
        while end in targets and end not in seen:
            seen.add(end)
            end = targets[end]
        if not terms[end]["quality_flags"]:
            terms[end]["boundary_rescue"] = True
    for term in terms.values():
        term["quality_eligible"] = not term["quality_flags"]
        term["quality_status"] = "clean_candidate" if term["quality_eligible"] else "review"
