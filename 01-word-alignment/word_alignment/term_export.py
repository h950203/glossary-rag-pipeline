"""The typed, offset-preserving input contract for word-similar v1.1."""
from contextlib import ExitStack
import csv
from .lab_common import atomic_text, read_rows, write_line

SCHEMA = "term-pairs-v1.1"


def review_rows(row):
    """Preserve general pairs that were trimmed, untyped, or not selected."""
    terms = row.get("term_pairs", [])
    for pair in row.get("pairs", []):
        if any(all(pair[lang+"_span"] == new[lang+"_span"] for lang in ("en","ko")) for new in terms):
            continue
        units = {lang: [u for u in row.get("term_units",{}).get(lang,[])
                        if u["span"] == pair[lang+"_span"]] for lang in ("en","ko")}
        related = [new["pair_id"] for new in terms if all(
            pair[lang+"_span"][0] <= new[lang+"_span"][0] < new[lang+"_span"][1] <= pair[lang+"_span"][1]
            for lang in ("en","ko"))]
        if related:
            reason = "resegmented_as_term_pair"
        elif not units["en"] or not units["ko"]:
            reason = "incomplete_boundary_or_untyped_span"
        elif not ({u["type"] for u in units["en"]} & {u["type"] for u in units["ko"]}):
            reason = "unit_type_mismatch"
        else:
            reason = "not_selected_as_term_pair"
        yield {"schema":"term-pair-review-v1.1", "line_id":row["line_id"], "method":row["method"],
               "ko_sentence":row["ko"], "en_sentence":row["en"], "original_pair":pair,
               "reason":reason, "related_term_pair_ids":related,
               "exact_unit_types":{lang:sorted({u["type"] for u in values}) for lang,values in units.items()},
               "translation_verified":False}


def export_terms(root, manifest, methods):
    artifacts = {}
    for method in methods:
        source = root / manifest["stages"][method]["path"]
        target = source.parent
        count, pair_count = 0, 0
        with ExitStack() as stack:
            jsonl = stack.enter_context(atomic_text(target / "term-pairs.jsonl", overwrite=True))
            text = stack.enter_context(atomic_text(target / "term-pairs-readable.txt", overwrite=True))
            stream = stack.enter_context(atomic_text(target / "term-pairs.csv", overwrite=True))
            review = stack.enter_context(atomic_text(target / "term-pairs-review.jsonl", overwrite=True))
            table = csv.writer(stream)
            table.writerow(["line_id", "type", "ko_surface", "en_surface", "ko_canonical", "en_canonical", "score"])
            for row in read_rows(source):
                pairs = row.get("term_pairs", [])
                record = {"schema": SCHEMA, "line_id": row["line_id"], "method": method,
                          "ko": row["ko"], "en": row["en"], "pairs": pairs,
                          "status": "aligned" if pairs else "no_term_alignment",
                          "source_status": row["status"], "term_diagnostics": row.get("term_diagnostics", {}),
                          "analysis_signature": row.get("term_analysis", {}).get("signature")}
                write_line(jsonl, record)
                for held in review_rows(row):
                    write_line(review, held)
                text.write(f"\n[{row['line_id']}]\nKO: {row['ko']}\nEN: {row['en']}\n")
                for pair in pairs:
                    text.write(f"  {pair['ko_canonical']} ↔ {pair['en_canonical']} | {pair['unit_type']} | {pair['score']:.3f}\n"
                               f"    원문: {pair['ko']} ↔ {pair['en']}\n")
                    table.writerow([row["line_id"], pair["unit_type"], pair["ko"], pair["en"],
                                    pair["ko_canonical"], pair["en_canonical"], pair["score"]])
                count += 1; pair_count += len(pairs)
        from .lab_common import file_hash
        artifacts[method] = {"path": str((target / "term-pairs.jsonl").relative_to(root)),
                             "sha256": file_hash(target / "term-pairs.jsonl"),
                             "sentences": count, "pairs": pair_count, "schema": SCHEMA}
        artifacts[method]["review_sha256"] = file_hash(target / "term-pairs-review.jsonl")
        print(f"용어형 대응 저장: {target / 'term-pairs.jsonl'}; {pair_count}쌍", flush=True)
    manifest["term_outputs"] = {**manifest.get("term_outputs", {}), **artifacts}
