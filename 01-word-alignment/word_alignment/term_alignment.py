"""Re-score grammar-derived source spans with the same contextual BGE vectors."""
from . import core
from .lab_common import finish_pairs


def intersect(a, b):
    return a[0] < b[1] and b[0] < a[1]


def align_term_units(row, features, args):
    import numpy as np
    from .lab_bge import pool_span
    proposals, diagnostics = [], {}
    for kind in args.term_types:
        groups = []
        for lang, encoded in zip(("en", "ko"), features):
            values, vectors = [], []
            for unit in row.get("term_units", {}).get(lang, []):
                if unit["type"] != kind:
                    continue
                vector, shared = pool_span(encoded, *unit["span"])
                if vector is not None:
                    values.append((unit, shared)); vectors.append(vector)
            groups.append((values, vectors))
        (es, ev), (ks, kv) = groups
        diagnostics[kind] = {"en_units": len(es), "ko_units": len(ks)}
        if not ev or not kv:
            continue
        if max(len(ev), len(kv)) > args.max_span_candidates:
            diagnostics[kind]["status"] = "too_many_candidates"
            continue
        matrix = np.stack(ev) @ np.stack(kv).T
        forward, reverse = matrix.argmax(axis=1), matrix.argmax(axis=0)
        for i, j in enumerate(forward.tolist()):
            score = float(matrix[i, j])
            if reverse[j] != i or score < args.min_similarity:
                continue
            en, en_shared = es[i]; ko, ko_shared = ks[j]
            margins = []
            for scores, values, selected in ((matrix[i], ks, j), (matrix[:, j], es, i)):
                alternatives = [float(value) for index, value in enumerate(scores)
                                if not intersect(values[index][0]["span"], values[selected][0]["span"])]
                margins.append(float(scores[selected]) - max(alternatives) if alternatives else 2.)
            margin = min(margins)
            if margin < args.margin:
                continue
            pair = {"kind": "word" if len(en["token_ids"]) == len(ko["token_ids"]) == 1 else "phrase",
                    "score": score, "unit_type": kind,
                    "evidence": {"source": "bge_term_unit_mutual_best", "margin": margin,
                                 "shared_subword_boundary": en_shared or ko_shared,
                                 "canonical_reuses_score": False}}
            for lang, unit in (("en", en), ("ko", ko)):
                pair.update({lang: unit["surface"], lang+"_span": unit["span"],
                             lang+"_token_ids": unit["token_ids"], lang+"_unit": unit,
                             lang+"_canonical": unit["canonical"]})
            proposals.append(pair)
    accepted = []
    for pair in sorted(proposals, key=lambda p: (-p["score"], -len(p["ko_unit"]["tokens"])-len(p["en_unit"]["tokens"]), p["en_span"], p["ko_span"])):
        if any(intersect(pair[lang+"_span"], old[lang+"_span"]) for old in accepted for lang in ("en", "ko")):
            continue
        accepted.append(pair)
    return finish_pairs(row, accepted), diagnostics
