"""
Lightweight Pipeline Module 4: Recursive Hierarchical Compositionality Filter.

Evaluates compound terms using a 2-stage hierarchical compositionality engine:
1. Stage 1 (Base 1-Word Lexicon):
   - Builds generic cross-lingual lexicon from single-word polysemy clusters.
   - Separates base literal compounds (overlap >= threshold, default 0.80) from candidate rich compounds.
2. Stage 2 (Recursive Hierarchical Literal & Over-extension Filtering):
   - Pass 1 (2-word keys): Evaluates 3+ word terms containing 2-word anchors.
     - Quarantines over-extensions (English matches anchor exactly without translating extra Korean words).
     - Promotes hierarchical literals (A + B+C, B+C + A where both contribute stems, overlap >= hier_threshold).
   - Pass 2 (3-word keys): Evaluates 4+ word terms containing 3-word anchors.
     - Promotes hierarchical literals.
   - Pass 3 (4-word keys): Evaluates 5+ word terms containing 4-word anchors.
     - Promotes hierarchical literals.
3. Outputs:
   - literal_compounds.jsonl (Base 1-word + Hierarchical 2/3/4-word literals)
   - rich_compounds.jsonl (Final pure domain / specialized compounds)
   - over_extensions.jsonl (Quarantined defective / over-extended mappings)
   - summary.json & sentence-linked provenance files
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from kiwipiepy import Kiwi
from nltk.stem import PorterStemmer

STEMMER = PorterStemmer()
EN_STOPWORDS: Set[str] = {
    "of", "the", "in", "for", "and", "to", "a", "an", "on", "with", "by", "at",
    "from", "as", "into", "through", "during", "including"
}


def clean_en_tokens(text: str) -> List[str]:
    """Extracts non-stopword alphabetic tokens in lowercase."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    content = [t for t in tokens if t not in EN_STOPWORDS]
    return content if content else tokens


def extract_stems(text: str) -> List[str]:
    """Extracts Porter-stemmed non-stopword tokens."""
    tokens = clean_en_tokens(text)
    return [STEMMER.stem(t) for t in tokens]


class CompositionalityFilter:
    """Production recursive hierarchical compositionality evaluator and artifact generator."""

    def __init__(
        self,
        min_trans_freq: int = 5,
        threshold: float = 0.80,
        hier_threshold: float = 0.75,
    ) -> None:
        self.min_trans_freq = min_trans_freq
        self.threshold = threshold
        self.hier_threshold = hier_threshold
        self.kiwi = Kiwi()

    def build_generic_lexicon(
        self, polysemy_path: Path, acronym_path: Optional[Path] = None
    ) -> Tuple[Dict[str, Set[str]], int, int, int]:
        """Builds a bidirectional dictionary linking Korean words/morphemes/acronyms to expected English stems."""
        ko_dict: Dict[str, Set[str]] = defaultdict(set)
        valid_ko_anchors = 0
        valid_en_anchors = 0
        total_valid_trs = 0

        with polysemy_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                lang = data.get("anchor_lang")
                anchor = data.get("anchor_term", "").strip()
                translations = data.get("translations", [])
                valid_trs = [t for t in translations if t.get("frequency", 0) >= self.min_trans_freq]
                if not valid_trs:
                    continue

                total_valid_trs += len(valid_trs)
                if lang == "ko":
                    valid_ko_anchors += 1
                    for tr in valid_trs:
                        tr_stems = extract_stems(tr.get("term", ""))
                        for s in tr_stems:
                            ko_dict[anchor].add(s)
                        for t in self.kiwi.tokenize(anchor):
                            if t.tag.startswith(("NN", "XR")) or t.tag == "SL":
                                for s in tr_stems:
                                    ko_dict[t.form].add(s)
                elif lang == "en":
                    valid_en_anchors += 1
                    en_stems = extract_stems(anchor)
                    for tr in valid_trs:
                        ko_text = tr.get("term", "").strip()
                        for s in en_stems:
                            ko_dict[ko_text].add(s)
                        for t in self.kiwi.tokenize(ko_text):
                            if t.tag.startswith(("NN", "XR")) or t.tag == "SL":
                                for s in en_stems:
                                    ko_dict[t.form].add(s)

        if acronym_path and acronym_path.exists():
            with acronym_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    acr = data.get("acronym", "").strip()
                    if not acr:
                        continue
                    for s in extract_stems(acr):
                        ko_dict[acr].add(s)
                        ko_dict[acr.upper()].add(s)
                        ko_dict[acr.lower()].add(s)

        total_links = sum(len(v) for v in ko_dict.values())
        return dict(ko_dict), valid_ko_anchors, valid_en_anchors, total_links

    def load_sentence_dictionary(self, candidates_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
        """Preloads sentence text lookup map from candidates.jsonl."""
        sent_dict: Dict[str, Dict[str, str]] = {}
        if not candidates_path or not candidates_path.exists():
            return sent_dict

        with candidates_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                ko_id = data.get("a_sent_id", "")
                ko_text = data.get("a_text", "")
                candidates = data.get("candidates", [])
                en_id = candidates[0].get("b_sent_id", "") if candidates else ""
                en_text = candidates[0].get("b_text", "") if candidates else ""

                sent_dict[ko_id] = {
                    "ko_text": ko_text,
                    "en_id": en_id,
                    "en_text": en_text,
                }
        return sent_dict

    def _evaluate_1word_record(
        self, item: Dict[str, Any], poly_dict: Dict[str, Set[str]]
    ) -> Tuple[Dict[str, Any], float]:
        """Computes constituent 1-word lexical overlap for a single compound term."""
        ko_term = item["surface_a"]
        en_term = item["surface_b"]

        ws_tokens = ko_term.split()
        kiwi_tokens = [t.form for t in self.kiwi.tokenize(ko_term) if t.tag.startswith(("NN", "XR", "VA")) or t.tag == "SL"]
        all_ko_tokens = set(ws_tokens) | set(kiwi_tokens)

        expected_stems: Set[str] = set()
        for kt in all_ko_tokens:
            if kt in poly_dict:
                expected_stems.update(poly_dict[kt])
            if kt.upper() in poly_dict:
                expected_stems.update(poly_dict[kt.upper()])
            if kt.lower() in poly_dict:
                expected_stems.update(poly_dict[kt.lower()])

        en_stems = extract_stems(en_term)
        if not en_stems:
            overlap_ratio = 0.0
            matched_stems: List[str] = []
        else:
            matched_stems = [s for s in en_stems if s in expected_stems]
            overlap_ratio = round(len(matched_stems) / len(en_stems), 4)

        record = dict(item)
        record["literal_overlap"] = overlap_ratio
        record["matched_stems"] = matched_stems
        record["en_stems"] = en_stems
        return record, overlap_ratio

    def _extract_anchors(
        self, records: List[Dict[str, Any]], target_len: int
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """Extracts n-word candidate anchors, their constituent stems, and exact English surface translations."""
        anchor_stems: Dict[str, Set[str]] = defaultdict(set)
        anchor_exact: Dict[str, Set[str]] = defaultdict(set)

        for r in records:
            tokens = r["surface_a"].split()
            if len(tokens) == target_len:
                anc = r["surface_a"]
                sb_clean = r["surface_b"].lower().strip()
                anchor_exact[anc].add(sb_clean)
                for s in extract_stems(sb_clean):
                    anchor_stems[anc].add(s)

        return anchor_stems, anchor_exact

    def _evaluate_hierarchical_pass(
        self,
        candidate_records: List[Dict[str, Any]],
        anchor_stems: Dict[str, Set[str]],
        anchor_exact: Dict[str, Set[str]],
        known_stems_dict: Dict[str, Set[str]],
        target_len: int,
        pass_name: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Executes one hierarchical pass:
        - Quarantines over-extensions (extra Korean word unaligned in English)
        - Identifies hierarchical literals (anchor + remaining words combine to cover English >= hier_threshold)
        """
        pass_literals: List[Dict[str, Any]] = []
        pass_over_extensions: List[Dict[str, Any]] = []
        remaining_records: List[Dict[str, Any]] = []

        for r in candidate_records:
            tokens = r["surface_a"].split()
            n_tokens = len(tokens)

            # Skip terms shorter than target_len + 1 (cannot contain an n-word anchor plus extra words)
            if n_tokens <= target_len:
                remaining_records.append(r)
                continue

            found_anc = None
            for i in range(n_tokens - target_len + 1):
                sub = " ".join(tokens[i : i + target_len])
                if sub in anchor_stems:
                    found_anc = sub
                    break

            if not found_anc:
                remaining_records.append(r)
                continue

            # Check Over-extension: English translation matches anchor translation exactly
            sb_clean = r["surface_b"].lower().strip()
            if sb_clean in anchor_exact.get(found_anc, set()):
                rec_oe = dict(r)
                rec_oe["composition_type"] = "over_extension_quarantined"
                rec_oe["quarantine_reason"] = (
                    f"English matches {target_len}-word anchor '{found_anc}' exactly without translating extra words"
                )
                rec_oe["anchor_key"] = found_anc
                pass_over_extensions.append(rec_oe)
                continue

            # Check Hierarchical Literal Composition
            rem_str = r["surface_a"].replace(found_anc, "").strip()
            rem_tokens = rem_str.split()
            rem_stems: Set[str] = set()
            for rt in rem_tokens:
                rem_stems.update(known_stems_dict.get(rt, set()))
                rem_stems.update(known_stems_dict.get(rt.upper(), set()))
                rem_stems.update(known_stems_dict.get(rt.lower(), set()))
                rem_stems.update(anchor_stems.get(rt, set()))

            en_s = extract_stems(r["surface_b"])
            if not en_s:
                remaining_records.append(r)
                continue

            anc_s = anchor_stems[found_anc]
            has_anc = any(s in anc_s for s in en_s)
            has_rem = any(s in rem_stems for s in en_s)
            matched = [s for s in en_s if s in anc_s or s in rem_stems]
            ratio = len(matched) / len(en_s)

            if has_anc and has_rem and ratio >= self.hier_threshold:
                rec_lit = dict(r)
                rec_lit["composition_type"] = "hierarchical_literal"
                rec_lit["hierarchical_stage"] = pass_name
                rec_lit["anchor_key"] = found_anc
                rec_lit["hierarchical_overlap"] = round(ratio, 4)
                pass_literals.append(rec_lit)
            else:
                remaining_records.append(r)

        return pass_literals, pass_over_extensions, remaining_records

    GENERIC_PREFIXES = {
        "주요", "특정", "실제", "전체", "최종", "해당", "기존", "일반", "잠재적",
        "상대적", "포괄적", "전반적", "기본적", "직접적", "간접적", "추가적", "일부",
        "개별", "기타", "평균", "핵심", "초기", "단기", "다양한", "적절한", "지속적",
        "일차적", "이차적", "정기적", "일상적", "대표적"
    }

    DOMAIN_PREFIXES = {
        "건강", "연구", "사회적", "병원", "환자", "간호", "임상", "의료", "치료",
        "교육", "운동", "학교", "지역", "신체적", "심리적", "정신적", "생리적", "인구학적",
        "생활", "가족", "부모", "학생", "여성", "남성", "노인", "아동", "청소년"
    }

    TARGET_PREFIXES = GENERIC_PREFIXES | DOMAIN_PREFIXES

    def _filter_modifier_compositional_terms(
        self,
        rich_candidates: List[Dict[str, Any]],
        poly_dict: Dict[str, Set[str]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Filters transparent compositional phrases formed by generic or domain modifiers."""
        mod_literals: List[Dict[str, Any]] = []
        pure_rich: List[Dict[str, Any]] = []

        both_matched_cnt = 0
        prefix_matched_cnt = 0
        head_matched_cnt = 0

        for r in rich_candidates:
            tokens = r["surface_a"].split()
            if len(tokens) != 2:
                pure_rich.append(r)
                continue

            w1, w2 = tokens[0], tokens[1]
            en_stems = set(extract_stems(r["surface_b"]))
            w1_stems = (
                poly_dict.get(w1, set())
                | poly_dict.get(w1.upper(), set())
                | poly_dict.get(w1.lower(), set())
            )
            w2_stems = (
                poly_dict.get(w2, set())
                | poly_dict.get(w2.upper(), set())
                | poly_dict.get(w2.lower(), set())
            )

            match_w1 = any(s in en_stems for s in w1_stems)
            match_w2 = any(s in en_stems for s in w2_stems)

            # Rule 1: Both tokens matched in 1-word dict
            if match_w1 and match_w2:
                rec_lit = dict(r)
                rec_lit["composition_type"] = "modifier_both_tokens_matched"
                mod_literals.append(rec_lit)
                both_matched_cnt += 1
                continue

            # Rule 2: Target Prefix in set AND Prefix matched in EN
            if w1 in self.TARGET_PREFIXES and match_w1:
                rec_lit = dict(r)
                rec_lit["composition_type"] = f"modifier_prefix_matched_{w1}"
                mod_literals.append(rec_lit)
                prefix_matched_cnt += 1
                continue

            # Rule 3: Target Prefix in set AND Head matched in EN
            if w1 in self.TARGET_PREFIXES and match_w2:
                rec_lit = dict(r)
                rec_lit["composition_type"] = f"modifier_head_matched_{w1}"
                mod_literals.append(rec_lit)
                head_matched_cnt += 1
                continue

            pure_rich.append(r)

        stats = {
            "total_modifier_literals": len(mod_literals),
            "both_matched": both_matched_cnt,
            "prefix_matched": prefix_matched_cnt,
            "head_matched": head_matched_cnt,
            "pure_rich_count": len(pure_rich),
        }
        return mod_literals, pure_rich, stats

    def _run_hierarchical_refinement(
        self,
        candidate_rich: List[Dict[str, Any]],
        poly_dict: Dict[str, Set[str]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Orchestrates 3-pass recursive hierarchical filtering across 2-word, 3-word, and 4-word keys."""
        all_hierarchical_literals: List[Dict[str, Any]] = []
        all_over_extensions: List[Dict[str, Any]] = []

        # Pass 1: 2-word keys
        anc2_stems, anc2_exact = self._extract_anchors(candidate_rich, target_len=2)
        p1_lit, p1_oe, rem_1 = self._evaluate_hierarchical_pass(
            candidate_records=candidate_rich,
            anchor_stems=anc2_stems,
            anchor_exact=anc2_exact,
            known_stems_dict=poly_dict,
            target_len=2,
            pass_name="2word",
        )
        all_hierarchical_literals.extend(p1_lit)
        all_over_extensions.extend(p1_oe)

        # Pass 2: 3-word keys
        anc3_stems, anc3_exact = self._extract_anchors(rem_1, target_len=3)
        known_stems_p2 = dict(poly_dict)
        known_stems_p2.update(anc2_stems)
        p2_lit, p2_oe, rem_2 = self._evaluate_hierarchical_pass(
            candidate_records=rem_1,
            anchor_stems=anc3_stems,
            anchor_exact=anc3_exact,
            known_stems_dict=known_stems_p2,
            target_len=3,
            pass_name="3word",
        )
        all_hierarchical_literals.extend(p2_lit)
        all_over_extensions.extend(p2_oe)

        # Pass 3: 4-word keys
        anc4_stems, anc4_exact = self._extract_anchors(rem_2, target_len=4)
        known_stems_p3 = dict(known_stems_p2)
        known_stems_p3.update(anc3_stems)
        p3_lit, p3_oe, rem_3 = self._evaluate_hierarchical_pass(
            candidate_records=rem_2,
            anchor_stems=anc4_stems,
            anchor_exact=anc4_exact,
            known_stems_dict=known_stems_p3,
            target_len=4,
            pass_name="4word",
        )
        all_hierarchical_literals.extend(p3_lit)
        all_over_extensions.extend(p3_oe)

        # Mark final rich records
        final_rich: List[Dict[str, Any]] = []
        for r in rem_3:
            rec = dict(r)
            rec["composition_type"] = "rich_domain_term"
            final_rich.append(rec)

        stats = {
            "pass1_2word_literals": len(p1_lit),
            "pass2_3word_literals": len(p2_lit),
            "pass3_4word_literals": len(p3_lit),
            "total_hierarchical_literals": len(all_hierarchical_literals),
            "over_extensions_quarantined": len(all_over_extensions),
            "final_rich_count": len(final_rich),
        }
        return all_hierarchical_literals, all_over_extensions, final_rich, stats

    def _attach_sentence_provenance(
        self, record: Dict[str, Any], sent_dict: Dict[str, Dict[str, str]]
    ) -> Dict[str, Any]:
        """Appends sample sentence texts for contextual provenance if available."""
        rec = dict(record)
        sample_sources = rec.get("sample_sources", [])
        sample_sentences = []
        if sample_sources and sent_dict:
            first_src = sample_sources[0]
            ko_id = first_src.split(":")[0]
            s_info = sent_dict.get(ko_id)
            if s_info:
                sample_sentences.append({
                    "source_pair": first_src,
                    "ko_text": s_info["ko_text"],
                    "en_text": s_info["en_text"],
                })
        rec["sample_sentences"] = sample_sentences
        return rec

    def run(
        self,
        compound_path: Path,
        polysemy_path: Path,
        out_dir: Path,
        candidates_path: Optional[Path] = None,
        acronym_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Executes full recursive hierarchical compositionality filtering and serializes production artifacts."""
        out_dir.mkdir(parents=True, exist_ok=True)

        print("[CompositionalityFilter] Building generic lexicon...")
        poly_dict, valid_ko, valid_en, total_links = self.build_generic_lexicon(
            polysemy_path, acronym_path=acronym_path
        )
        print(f"  - Anchors: {valid_ko:,} KO, {valid_en:,} EN | Keys: {len(poly_dict):,} | Links: {total_links:,}")

        sent_dict = self.load_sentence_dictionary(candidates_path)
        if sent_dict:
            print(f"  - Loaded {len(sent_dict):,} sample sentence pairs for contextual provenance.")

        # Stage 1: Base 1-word lexicon evaluation
        print("\n▶ [Stage 1] Evaluating base 1-word literal compositionality...")
        score_brackets = {
            "1.0 (Exact 100% Literal)": 0,
            "0.75 - 0.99 (Mostly Literal)": 0,
            "0.50 - 0.74 (Partially Compositional)": 0,
            "0.01 - 0.49 (Strongly Non-compositional)": 0,
            "0.0 (Emergent / Unknown Lexicon)": 0,
        }

        base_literals: List[Dict[str, Any]] = []
        candidate_rich: List[Dict[str, Any]] = []
        all_scored_file = out_dir / "compound_terms_scored.jsonl"

        with compound_path.open("r", encoding="utf-8") as f_in, all_scored_file.open("w", encoding="utf-8") as f_all:
            for line in f_in:
                if not line.strip():
                    continue
                item = json.loads(line)
                rec, overlap = self._evaluate_1word_record(item, poly_dict)
                f_all.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if overlap >= 1.0:
                    score_brackets["1.0 (Exact 100% Literal)"] += 1
                elif overlap >= 0.75:
                    score_brackets["0.75 - 0.99 (Mostly Literal)"] += 1
                elif overlap >= 0.50:
                    score_brackets["0.50 - 0.74 (Partially Compositional)"] += 1
                elif overlap > 0.0:
                    score_brackets["0.01 - 0.49 (Strongly Non-compositional)"] += 1
                else:
                    score_brackets["0.0 (Emergent / Unknown Lexicon)"] += 1

                if overlap >= self.threshold:
                    rec["composition_type"] = "base_1word_literal"
                    base_literals.append(rec)
                else:
                    candidate_rich.append(rec)

        print(f"  - Base 1-Word Literals: {len(base_literals):,} | Candidate Rich: {len(candidate_rich):,}")

        # Stage 2: Recursive Hierarchical Refinement
        print("\n▶ [Stage 2] Running recursive hierarchical refinement (Pass 1~3)...")
        hier_literals, over_exts, hier_rich, hier_stats = self._run_hierarchical_refinement(
            candidate_rich=candidate_rich,
            poly_dict=poly_dict,
        )
        print(f"  - Pass 1 (2-word keys): {hier_stats['pass1_2word_literals']:,} literals")
        print(f"  - Pass 2 (3-word keys): {hier_stats['pass2_3word_literals']:,} literals")
        print(f"  - Pass 3 (4-word keys): {hier_stats['pass3_4word_literals']:,} literals")
        print(f"  - Quarantined Over-extensions: {hier_stats['over_extensions_quarantined']:,}")
        print(f"  - Candidate Rich after Stage 2: {len(hier_rich):,}")

        # Stage 3: Domain & Generic Modifier Compositionality Filter
        print("\n▶ [Stage 3] Running Domain & Generic Modifier Compositionality Filter...")
        mod_literals, final_rich, mod_stats = self._filter_modifier_compositional_terms(
            rich_candidates=hier_rich,
            poly_dict=poly_dict,
        )
        print(f"  - Converted Modifier Literals: {mod_stats['total_modifier_literals']:,} terms")
        print(f"    (Both matched: {mod_stats['both_matched']:,}, Prefix matched: {mod_stats['prefix_matched']:,}, Head matched: {mod_stats['head_matched']:,})")
        print(f"  - Final Pure Rich Technical Terms: {len(final_rich):,}")

        # Serialization
        all_literals = base_literals + hier_literals + mod_literals
        literal_file = out_dir / "literal_compounds.jsonl"
        rich_file = out_dir / "rich_compounds.jsonl"
        oe_file = out_dir / "over_extensions.jsonl"
        lit_sents_file = out_dir / "literal_compounds_with_sentences.jsonl"
        rich_sents_file = out_dir / "rich_compounds_with_sentences.jsonl"

        with literal_file.open("w", encoding="utf-8") as f_lit, lit_sents_file.open("w", encoding="utf-8") as f_lits:
            for r in all_literals:
                f_lit.write(json.dumps(r, ensure_ascii=False) + "\n")
                r_sent = self._attach_sentence_provenance(r, sent_dict)
                f_lits.write(json.dumps(r_sent, ensure_ascii=False) + "\n")

        with rich_file.open("w", encoding="utf-8") as f_rich, rich_sents_file.open("w", encoding="utf-8") as f_richs:
            for r in final_rich:
                f_rich.write(json.dumps(r, ensure_ascii=False) + "\n")
                r_sent = self._attach_sentence_provenance(r, sent_dict)
                f_richs.write(json.dumps(r_sent, ensure_ascii=False) + "\n")

        with oe_file.open("w", encoding="utf-8") as f_oe:
            for r in over_exts:
                f_oe.write(json.dumps(r, ensure_ascii=False) + "\n")

        total_input = len(base_literals) + len(candidate_rich)
        summary = {
            "experiment": "Production Recursive Hierarchical Compositionality Filter with Domain/Generic Modifiers",
            "threshold_1word": self.threshold,
            "threshold_hierarchical": self.hier_threshold,
            "min_translation_frequency": self.min_trans_freq,
            "total_compound_records": total_input,
            "base_1word_literals": len(base_literals),
            "hierarchical_literals_added": len(hier_literals),
            "modifier_literals_added": len(mod_literals),
            "total_literals_count": len(all_literals),
            "total_literals_ratio": round(len(all_literals) / total_input, 4) if total_input else 0,
            "over_extensions_quarantined": len(over_exts),
            "final_rich_preserved_count": len(final_rich),
            "final_rich_preserved_ratio": round(len(final_rich) / total_input, 4) if total_input else 0,
            "hierarchical_breakdown": hier_stats,
            "modifier_breakdown": mod_stats,
            "distribution_1word": score_brackets,
            "artifacts": {
                "compound_terms_scored": str(all_scored_file.name),
                "literal_compounds": str(literal_file.name),
                "rich_compounds": str(rich_file.name),
                "over_extensions": str(oe_file.name),
                "rich_compounds_with_sentences": str(rich_sents_file.name),
                "literal_compounds_with_sentences": str(lit_sents_file.name),
            },
        }

        with (out_dir / "summary.json").open("w", encoding="utf-8") as f_sum:
            json.dump(summary, f_sum, ensure_ascii=False, indent=2)

        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Production Recursive Hierarchical Compositionality Filter.")
    parser.add_argument("--compound_path", type=str, default="output/lightweight_full/compound_terms.jsonl")
    parser.add_argument("--polysemy_path", type=str, default="output/lightweight_full/polysemy_clusters.jsonl")
    parser.add_argument("--acronym_path", type=str, default="output/lightweight_full/acronym_terms.jsonl")
    parser.add_argument("--candidates_path", type=str, default="output/candidates.jsonl")
    parser.add_argument("--output_dir", type=str, default="output/lightweight_full/compositionality")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--hier_threshold", type=float, default=0.75)
    parser.add_argument("--min_trans_freq", type=int, default=5)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    c_filter = CompositionalityFilter(
        min_trans_freq=args.min_trans_freq,
        threshold=args.threshold,
        hier_threshold=args.hier_threshold,
    )
    summary = c_filter.run(
        compound_path=base_dir / args.compound_path,
        polysemy_path=base_dir / args.polysemy_path,
        out_dir=base_dir / args.output_dir,
        candidates_path=base_dir / args.candidates_path,
        acronym_path=(base_dir / args.acronym_path) if args.acronym_path else None,
    )
    print("\n🎉 Recursive Hierarchical Compositionality Filter Complete!")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
