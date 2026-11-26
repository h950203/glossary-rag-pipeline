"""
Bilingual Term Candidate Mapping Runner.
Executes BGE-M3 candidate mapping directly on external candidate files without Kiwi/NLTK regeneration.

Stages executed:
4. BGE-M3 Dense Multilingual Embedding Mapping (MNN & NMS Span Masking)
4-1. 1-Word Terminology Dictionary (1:N Polysemy/Synonym Clusters)
     (4-1-1: A언어 1단어, B언어 2단어 이상 -> A언어 1단어 클러스터 번역어로 등록)
4-2. Compound Terminology Dictionary (양쪽 2단어 이상) & Recursive Hierarchical Compositionality Filter

Default I/O:
    Input:  co-work/candidates.jsonl (or specified via --candidates_file)
    Output: co-work/output/
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

# Add Term_Mapping root to sys.path if not present
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from models import CandidateBundle, CandidatePair, RawTermPair, UnmappedSentence
from pipeline_lightweight.cluster_builder import ClusterBuilder
from pipeline_lightweight.compositionality_filter import CompositionalityFilter
from pipeline_lightweight.domain_input import export_domain_input
from pipeline_lightweight.term_mapper import TermMapper


def _extract_cand_strings(raw_cands: Any) -> List[str]:
    """Extracts cleaned string candidates from list of strings or dicts."""
    if not isinstance(raw_cands, list):
        return []
    result: List[str] = []
    for item in raw_cands:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
        elif isinstance(item, dict):
            val = item.get("text") or item.get("term") or item.get("surface") or ""
            if isinstance(val, str) and val.strip():
                result.append(val.strip())
    return list(dict.fromkeys(result))


def parse_external_record(
    line_dict: Dict[str, Any], idx: int
) -> Tuple[CandidateBundle, List[str], List[str]]:
    """Parses a single external candidate record into CandidateBundle and candidate lists."""
    ko_text = (
        line_dict.get("ko_sentence")
        or line_dict.get("a_text")
        or line_dict.get("ko_text")
        or line_dict.get("ko")
        or ""
    )
    en_text = (
        line_dict.get("en_sentence")
        or line_dict.get("b_text")
        or line_dict.get("en_text")
        or line_dict.get("en")
        or ""
    )

    line_num = line_dict.get("line_id") or idx
    ko_id = (
        line_dict.get("ko_sent_id")
        or line_dict.get("a_sent_id")
        or line_dict.get("sent_id_ko")
        or f"KO_{int(line_num):04d}"
    )
    en_id = (
        line_dict.get("en_sent_id")
        or line_dict.get("b_sent_id")
        or line_dict.get("sent_id_en")
        or f"EN_{int(line_num):04d}"
    )

    score_val = (
        line_dict.get("alignment_score")
        or line_dict.get("score")
        or line_dict.get("top1_score")
        or 1.0
    )
    score = float(score_val)

    tier = str(
        line_dict.get("alignment_tier")
        or line_dict.get("tier")
        or "MATCHED"
    )

    raw_ko_cands = (
        line_dict.get("ko_candidates")
        or line_dict.get("a_candidates")
        or line_dict.get("korean_candidates")
        or []
    )
    raw_en_cands = (
        line_dict.get("en_candidates")
        or line_dict.get("b_candidates")
        or line_dict.get("english_candidates")
        or []
    )

    ko_cands = _extract_cand_strings(raw_ko_cands)
    en_cands = _extract_cand_strings(raw_en_cands)

    candidates: List[CandidatePair] = []
    if en_text:
        candidates.append(
            CandidatePair(
                b_sent_id=en_id,
                b_text=en_text,
                score=score,
                rank=1,
            )
        )

    bundle = CandidateBundle(
        a_sent_id=ko_id,
        a_text=ko_text,
        top1_score=score,
        alignment_tier=tier,
        is_aligned_candidate=True,
        candidates=candidates,
    )
    return bundle, ko_cands, en_cands


def load_external_dataset(
    file_path: Path, max_count: Optional[int] = None
) -> Tuple[List[CandidateBundle], List[Tuple[List[str], List[str]]]]:
    """Loads and validates external candidate dataset (supports both JSON array and JSONL)."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {file_path}\n"
            f"Please place your candidates file at '{file_path}' or pass --candidates_file <path>."
        )

    bundles: List[CandidateBundle] = []
    bundle_cands: List[Tuple[List[str], List[str]]] = []

    with file_path.open("r", encoding="utf-8") as f:
        first_char = ""
        while True:
            ch = f.read(1)
            if not ch or not ch.isspace():
                first_char = ch
                break
        f.seek(0)

        if first_char == "[":
            records = json.load(f)
            for idx, data in enumerate(records, 1):
                if not isinstance(data, dict):
                    continue
                bundle, ko_c, en_c = parse_external_record(data, idx)
                bundles.append(bundle)
                bundle_cands.append((ko_c, en_c))
                if max_count is not None and len(bundles) >= max_count:
                    break
        else:
            for idx, line in enumerate(f, 1):
                line_str = line.strip()
                if not line_str:
                    continue
                data = json.loads(line_str)
                bundle, ko_c, en_c = parse_external_record(data, idx)
                bundles.append(bundle)
                bundle_cands.append((ko_c, en_c))
                if max_count is not None and len(bundles) >= max_count:
                    break

    return bundles, bundle_cands


def save_aligned_bundles_jsonl(bundles: List[CandidateBundle], out_path: Path) -> None:
    """Saves candidate bundles in standard JSONL format for downstream provenance loading."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for b in bundles:
            rec = {
                "a_sent_id": b.a_sent_id,
                "a_text": b.a_text,
                "top1_score": round(b.top1_score, 4),
                "alignment_tier": b.alignment_tier,
                "is_aligned_candidate": b.is_aligned_candidate,
                "candidates": [
                    {
                        "b_sent_id": c.b_sent_id,
                        "b_text": c.b_text,
                        "score": round(c.score, 4),
                        "rank": c.rank,
                    }
                    for c in b.candidates
                ],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def execute_mapping(
    bundles: List[CandidateBundle],
    bundle_cands: List[Tuple[List[str], List[str]]],
    term_mapper: TermMapper,
) -> Tuple[List[RawTermPair], List[RawTermPair], List[UnmappedSentence]]:
    """Maps candidate phrases using dense embeddings and isolates defective sentences."""
    all_standard_pairs: List[RawTermPair] = []
    all_polysemy_pairs: List[RawTermPair] = []
    defective_sentences: List[UnmappedSentence] = []
    total = len(bundles)

    is_ko_compound = lambda text: len(text.strip().split()) >= 2

    for idx, (b, (ko_c, en_c)) in enumerate(zip(bundles, bundle_cands), 1):
        if idx % 1000 == 0 or idx == total:
            print(f"    Mapping progress: {idx:,} / {total:,} ({idx/total*100:.1f}%)")

        top1 = b.candidates[0] if b.candidates else None
        if top1 is None:
            continue

        std_p, poly_p, uncov_comp = term_mapper.map_bundle(
            bundle=b,
            ko_cands=ko_c,
            en_cands=en_c,
            is_ko_compound_fn=is_ko_compound,
        )
        all_standard_pairs.extend(std_p)
        all_polysemy_pairs.extend(poly_p)

        if uncov_comp:
            defective_sentences.append(
                UnmappedSentence(
                    source_a_id=b.a_sent_id,
                    source_b_id=top1.b_sent_id,
                    a_text=b.a_text,
                    b_text=top1.b_text,
                    labse_score=top1.score,
                    uncovered_compounds=uncov_comp,
                )
            )

    return all_standard_pairs, all_polysemy_pairs, defective_sentences


def merge_polysemy_clusters(paths: List[Path], out_path: Path) -> Path:
    """Merges multiple polysemy_clusters.jsonl files by summing frequencies per anchor and translation."""
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = (rec["canonical_term"], rec["anchor_lang"])
                if key not in merged:
                    merged[key] = {
                        "anchor_term": rec["anchor_term"],
                        "anchor_lang": rec["anchor_lang"],
                        "canonical_term": rec["canonical_term"],
                        "total_frequency": 0,
                        "translations": defaultdict(lambda: {"frequency": 0, "sample_sources": []}),
                    }
                entry = merged[key]
                entry["total_frequency"] += int(rec.get("total_frequency", 0))
                for tr in rec.get("translations", []):
                    t_term = tr["term"]
                    t_entry = entry["translations"][t_term]
                    t_entry["frequency"] += int(tr.get("frequency", 0))
                    for src in tr.get("sample_sources", []):
                        if src not in t_entry["sample_sources"] and len(t_entry["sample_sources"]) < 5:
                            t_entry["sample_sources"].append(src)

    sorted_items = sorted(
        merged.items(), key=lambda x: (x[1]["total_frequency"], x[0][0]), reverse=True
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, data in sorted_items:
            sorted_trs = sorted(
                data["translations"].items(), key=lambda x: x[1]["frequency"], reverse=True
            )
            tr_list = [
                {
                    "term": term,
                    "frequency": info["frequency"],
                    "sample_sources": info["sample_sources"],
                }
                for term, info in sorted_trs
            ]
            out_rec = {
                "anchor_term": data["anchor_term"],
                "anchor_lang": data["anchor_lang"],
                "canonical_term": data["canonical_term"],
                "total_frequency": data["total_frequency"],
                "distinct_translations": len(tr_list),
                "translations": tr_list,
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    return out_path


BOOLEAN_OPERATORS = {"AND", "OR", "NOT"}
MASKING_TOKENS_2CHAR = {"OO"}


def is_excluded_acronym(acr: str) -> bool:
    """Returns True if the token is a Boolean search operator or anonymization masking string."""
    u = acr.strip().upper()
    if u in BOOLEAN_OPERATORS or u in MASKING_TOKENS_2CHAR:
        return True
    if len(u) >= 3 and len(set(u)) == 1 and u.isalpha() and u != "III":
        return True
    return False


def filter_acronym_file(path: Path) -> Tuple[List[Dict[str, Any]], List[Tuple[str, int]]]:
    """Filters out Boolean operators and anonymization masking tokens from an acronym_terms.jsonl file in-place."""
    if not path.exists():
        return [], []
    kept: List[Dict[str, Any]] = []
    removed: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            acr = rec.get("acronym", "")
            if is_excluded_acronym(acr):
                removed.append((acr, int(rec.get("frequency", 0))))
            else:
                kept.append(rec)
    with path.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return kept, removed


def merge_acronym_terms(paths: List[Path], out_path: Path) -> Path:
    """Merges multiple acronym_terms.jsonl files by summing frequencies (excluding operators/masking)."""
    merged: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                acr = rec["acronym"]
                if is_excluded_acronym(acr):
                    continue
                if acr not in merged:
                    merged[acr] = {
                        "acronym": acr,
                        "surface_a": acr,
                        "surface_b": acr,
                        "frequency": 0,
                        "sample_sources": [],
                        "category": "acronym",
                    }
                merged[acr]["frequency"] += int(rec.get("frequency", 0))
                for src in rec.get("sample_sources", []):
                    if src not in merged[acr]["sample_sources"] and len(merged[acr]["sample_sources"]) < 5:
                        merged[acr]["sample_sources"].append(src)

    sorted_items = sorted(merged.values(), key=lambda x: (x["frequency"], x["acronym"]), reverse=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in sorted_items:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path


def main() -> None:
    cowork_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Map bilingual term candidates with BGE-M3 and check compound compositionality."
    )
    default_cand = (
        cowork_dir / "candidates.json"
        if (cowork_dir / "candidates.json").exists()
        else cowork_dir / "candidates.jsonl"
    )
    parser.add_argument(
        "--candidates_file",
        type=str,
        default=str(default_cand),
        help="Path to external candidate JSON/JSONL file (default: co-work/candidates.json or candidates.jsonl).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(cowork_dir / "output"),
        help="Directory to save mapping and compositionality artifacts (default: co-work/output).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda' or 'cpu'). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--bge_model", default="BAAI/bge-m3",
        help="Hugging Face model ID or a local BGE-M3 model directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for BGE-M3 candidate phrase encoding.",
    )
    parser.add_argument(
        "--min_trans_freq",
        type=int,
        default=5,
        help="Minimum translation frequency for generic lexicon in compositionality filter (default: 5).",
    )
    parser.add_argument(
        "--skip_compositionality_filter",
        action="store_true",
        default=False,
        help="Skip Stage 4-2 recursive hierarchical compositionality filtering.",
    )
    parser.add_argument(
        "--external_polysemy_path",
        type=str,
        default=None,
        help="Optional path to an external polysemy_clusters.jsonl to use as the base 1-word lexicon.",
    )
    parser.add_argument(
        "--external_acronym_path",
        type=str,
        default=None,
        help="Optional path to an external acronym_terms.jsonl to use with compositionality filter.",
    )
    parser.add_argument(
        "--merge_external_polysemy",
        action="store_true",
        default=False,
        help="When set with --external_polysemy_path, merges self + external polysemy (and acronym) dictionaries.",
    )
    parser.add_argument(
        "--comp_out_dir",
        type=str,
        default=None,
        help="Optional custom output directory (absolute or relative to output_dir) for compositionality filter results.",
    )
    parser.add_argument(
        "--compositionality_only",
        action="store_true",
        default=False,
        help="Run only Stage 4-2 compositionality filter on existing compound_terms.jsonl in output_dir.",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=None,
        help="Maximum sentence bundles to process (for quick testing).",
    )
    args = parser.parse_args()

    cand_path = Path(args.candidates_file)
    if not cand_path.is_absolute():
        if cand_path.exists():
            cand_path = cand_path.resolve()
        elif (cowork_dir / args.candidates_file).exists():
            cand_path = (cowork_dir / args.candidates_file).resolve()
        else:
            cand_path = (base_dir / args.candidates_file).resolve()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        if str(args.output_dir).startswith("co-work"):
            out_dir = (base_dir / args.output_dir).resolve()
        else:
            out_dir = (cowork_dir / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    compounds_out = out_dir / "compound_terms.jsonl"
    poly_out = out_dir / "polysemy_clusters.jsonl"
    acronym_out = out_dir / "acronym_terms.jsonl"
    saved_bundles_path = out_dir / "aligned_bundles.jsonl"

    if not args.compositionality_only:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

        print("================================================================================")
        print("🚀 Bilingual Term Candidate Mapping & Compositionality")
        print(f"   Input Candidates: {cand_path}")
        print(f"   Output Directory: {out_dir}")
        print(f"   Device:           {device}")
        print("================================================================================")

        print("\n▶ [1/4] Loading candidate dataset...")
        bundles, bundle_cands = load_external_dataset(cand_path, max_count=args.max_count)
        total_bundles = len(bundles)
        print(f"  - Loaded {total_bundles:,} sentence bundles.")

        all_ko_phrases: List[str] = []
        all_en_phrases: List[str] = []
        for ko_c, en_c in bundle_cands:
            all_ko_phrases.extend(ko_c)
            all_en_phrases.extend(en_c)

        unique_phrases = list(set(all_ko_phrases + all_en_phrases))
        print(
            f"  - Loaded {len(all_ko_phrases):,} KO / {len(all_en_phrases):,} EN candidates "
            f"({len(unique_phrases):,} unique phrases across dataset)"
        )

        print("\n▶ [2/4] Initializing BGE-M3 and Precomputing Phrase Embeddings...")
        term_mapper = TermMapper(model_name=args.bge_model, device=device)
        t0_enc = time.time()
        term_mapper.precompute_embeddings(unique_phrases, batch_size=args.batch_size)
        print(f"  - Embeddings precomputed in {time.time() - t0_enc:.2f}s")

        print("\n▶ [3/4] Executing Mapping, MNN Filtering, and Local NMS...")
        t0_map = time.time()
        std_pairs, poly_pairs, defective_records = execute_mapping(
            bundles=bundles, bundle_cands=bundle_cands, term_mapper=term_mapper
        )
        elapsed_map = time.time() - t0_map
        throughput = total_bundles / elapsed_map if elapsed_map > 0 else 0
        print(f"  - Mapping complete in {elapsed_map:.2f}s ({throughput:.1f} sent/s)")
        print(f"  - [4-2] Multi-word Compound Pairs: {len(std_pairs):,}")
        print(f"  - [4-1] Single/Asymmetric Pairs:   {len(poly_pairs):,}")

        save_aligned_bundles_jsonl(bundles, saved_bundles_path)

        print("\n▶ [4/4] Building Unified Dictionaries and Artifacts...")
        compound_records = ClusterBuilder.save_compound_terms(std_pairs, compounds_out)
        poly_records = ClusterBuilder.save_unified_polysemy_clusters(poly_pairs, poly_out)
        ClusterBuilder.extract_and_save_acronym_terms(bundles, acronym_out)
        acronym_records, removed_acros = filter_acronym_file(acronym_out)
        if removed_acros:
            print(f"  - Filtered out {len(removed_acros)} operator/masking acronyms: {removed_acros}")

        unmapped_out = out_dir / "unmapped_sentences.jsonl"
        ClusterBuilder.save_unmapped_sentences(defective_records, unmapped_out)

        raw_out = out_dir / "raw_occurrences.jsonl"
        if raw_out.exists():
            raw_out.unlink()
        ClusterBuilder.save_raw_occurrences(std_pairs, raw_out, category="compound")
        ClusterBuilder.save_raw_occurrences(poly_pairs, raw_out, category="single_noun")

        report_out = out_dir / "report.md"
        ClusterBuilder.generate_report(
            out_path=report_out,
            elapsed_sec=elapsed_map,
            throughput=throughput,
            compound_records=compound_records,
            unified_cluster_records=poly_records,
            acronym_records=acronym_records,
            defective_sentence_records=defective_records,
            total_sentences=total_bundles,
            filtered_tier_count=0,
            include_gray_zone=True,
        )

        print("================================================================================")
        print("✅ Stage 4 & 4-1 Dictionary Building Complete!")
        print(f"  - [4-2] Compound Terms:           {compounds_out} ({len(compound_records):,} terms)")
        print(f"  - [4-1] Unified Polysemy Clusters: {poly_out} ({len(poly_records):,} clusters)")
        print(f"  - Acronym Terms:                  {acronym_out} ({len(acronym_records):,} acronyms)")
        print(f"  - Unmapped Sentences:             {unmapped_out} ({len(defective_records):,} sentences)")
        print(f"  - Summary Report:                 {report_out}")
        print("================================================================================")

    if not args.skip_compositionality_filter:
        if args.comp_out_dir:
            comp_dir = Path(args.comp_out_dir)
            if not comp_dir.is_absolute():
                comp_dir = out_dir / comp_dir
        else:
            comp_dir = out_dir / "compositionality"
        comp_dir.mkdir(parents=True, exist_ok=True)

        if args.external_polysemy_path and args.merge_external_polysemy:
            ext_poly = Path(args.external_polysemy_path).resolve()
            active_poly_path = merge_polysemy_clusters(
                [poly_out, ext_poly], comp_dir / "merged_polysemy_clusters.jsonl"
            )
            if args.external_acronym_path:
                ext_acro = Path(args.external_acronym_path).resolve()
                active_acro_path = merge_acronym_terms(
                    [acronym_out, ext_acro], comp_dir / "merged_acronym_terms.jsonl"
                )
            else:
                active_acro_path = acronym_out
        else:
            active_poly_path = Path(args.external_polysemy_path).resolve() if args.external_polysemy_path else poly_out
            active_acro_path = Path(args.external_acronym_path).resolve() if args.external_acronym_path else acronym_out

        print("\n▶ [Stage 4-2] Running Recursive Hierarchical Compositionality Filter...")
        print(f"  - Reference Polysemy Lexicon: {active_poly_path}")
        print(f"  - Reference Acronym Lexicon:  {active_acro_path}")
        print(f"  - Output Subdirectory:        {comp_dir}")
        comp_filter = CompositionalityFilter(
            min_trans_freq=args.min_trans_freq, threshold=0.80
        )
        comp_filter.run(
            compound_path=compounds_out,
            polysemy_path=active_poly_path,
            out_dir=comp_dir,
            candidates_path=saved_bundles_path,
            acronym_path=active_acro_path,
        )
        print("================================================================================")
        print("🎉 Stage 4-2 Compositionality Filter Finished!")
        print(f"  - Rich Domain Compounds:         {comp_dir / 'rich_compounds.jsonl'}")
        print(f"  - Literal Compounds:             {comp_dir / 'literal_compounds.jsonl'}")
        print(f"  - Over-extension Quarantined:    {comp_dir / 'over_extensions.jsonl'}")
        print("================================================================================")

        if poly_out.is_file() and saved_bundles_path.is_file():
            domain_out = out_dir / "combined_for_domain_check.jsonl"
            count = export_domain_input(comp_dir / "rich_compounds.jsonl", poly_out, saved_bundles_path, domain_out)
            print(f"  - Word-domain-check Input:       {domain_out} ({count:,} pairs)")
        else:
            print("  - 도메인 분류 입력 저장 생략: 자체 다의어 사전과 aligned_bundles.jsonl이 필요합니다.")


if __name__ == "__main__":
    main()
