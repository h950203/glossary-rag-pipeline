"""
Lightweight Pipeline Module 3: Cluster Builder & Artifact Exporter.
Builds deduplicated terminology dictionaries and unifies polysemy/synonym pairs:
1. compound_terms.jsonl: Pure Symmetric Multi-word Technical Compounds (>=2 words in both languages)
2. polysemy_clusters.jsonl: Unified 1:N Polysemy & Synonym Clusters (Single consolidated file for both KO and EN anchors)
3. acronym_terms.jsonl: Standalone cross-lingual acronyms (uppercase/digit/hyphen tokens present in both sentences)
4. unmapped_sentences.jsonl: Defective sentences containing unmapped multi-word compounds
5. raw_occurrences.jsonl: Sentence-level provenance stream
6. report.md: Quantitative benchmark and quality summary report
"""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common.text_utils import (
    ACRONYM_TOKEN_RE,
    PUNCT_STRIP,
    clean_phrase,
    has_ascii_alpha,
    has_hangul,
)
from models import CandidateBundle, RawTermPair, UnmappedSentence


class ClusterBuilder:
    """Aggregates raw matches, eliminates duplicates, and builds unified output dictionaries."""

    @staticmethod
    def save_compound_terms(
        pairs: List[RawTermPair], out_path: Path
    ) -> List[Dict[str, Any]]:
        """Deduplicates pure symmetric multi-word compound pairs and sorts descending by frequency."""
        agg_map = defaultdict(lambda: {"count": 0, "sources": []})
        for p in pairs:
            ka = clean_phrase(p.surface_a)
            kb = clean_phrase(p.surface_b).lower()
            if not ka or not kb:
                continue
            key = (ka, kb)
            agg_map[key]["count"] += 1
            src = f"{p.source_a_id}:{p.source_b_id}"
            if src not in agg_map[key]["sources"] and len(agg_map[key]["sources"]) < 5:
                agg_map[key]["sources"].append(src)

        sorted_items = sorted(agg_map.items(), key=lambda x: (x[1]["count"], x[0][0]), reverse=True)

        records = []
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for (ka, kb), data in sorted_items:
                rec = {
                    "surface_a": ka,
                    "surface_b": kb,
                    "canonical_a": ka,
                    "canonical_b": kb,
                    "frequency": data["count"],
                    "sample_sources": data["sources"],
                    "category": "compound_term",
                }
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return records

    @classmethod
    def save_unified_polysemy_clusters(
        cls, pairs: List[RawTermPair], out_path: Path
    ) -> List[Dict[str, Any]]:
        """Consolidates single/asymmetric polysemy & synonym pairs into a single unified JSONL file."""
        clusters = cls._aggregate_polysemy_clusters(pairs)
        sorted_clusters = sorted(
            clusters.items(), key=lambda x: (x[1]["total_freq"], x[0][0]), reverse=True
        )
        return cls._write_polysemy_clusters(sorted_clusters, out_path)

    @staticmethod
    def _aggregate_polysemy_clusters(
        pairs: List[RawTermPair],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Accumulates translations under KO or EN single-word anchors."""
        clusters: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {
                "display_anchor": "",
                "total_freq": 0,
                "translations": defaultdict(lambda: {"count": 0, "sources": []}),
            }
        )

        for p in pairs:
            sa = clean_phrase(p.surface_a)
            sb = clean_phrase(p.surface_b)
            ca = clean_phrase(p.canonical_a)
            cb = clean_phrase(p.canonical_b)
            if not sa or not sb:
                continue

            words_a = len(sa.split())
            words_b = len(sb.split())
            src = f"{p.source_a_id}:{p.source_b_id}"

            if words_b == 1 and has_ascii_alpha(cb):
                anchor_en = cb.lower()
                key_en = (anchor_en, "en")
                clusters[key_en]["display_anchor"] = anchor_en
                clusters[key_en]["total_freq"] += 1
                clusters[key_en]["translations"][sa]["count"] += 1
                srcs = clusters[key_en]["translations"][sa]["sources"]
                if src not in srcs and len(srcs) < 5:
                    srcs.append(src)

            if words_a == 1 and has_hangul(ca):
                key_ko = (ca, "ko")
                clusters[key_ko]["display_anchor"] = sa
                clusters[key_ko]["total_freq"] += 1
                trans_en = sb.lower()
                clusters[key_ko]["translations"][trans_en]["count"] += 1
                srcs = clusters[key_ko]["translations"][trans_en]["sources"]
                if src not in srcs and len(srcs) < 5:
                    srcs.append(src)

        return clusters

    @staticmethod
    def _write_polysemy_clusters(
        sorted_clusters: List[Tuple[Tuple[str, str], Dict[str, Any]]],
        out_path: Path,
    ) -> List[Dict[str, Any]]:
        """Formats and writes polysemy cluster records to JSONL."""
        records = []
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for (anchor_canon, lang), data in sorted_clusters:
                sorted_trans = sorted(
                    data["translations"].items(), key=lambda x: x[1]["count"], reverse=True
                )
                trans_list = [
                    {
                        "term": t,
                        "frequency": t_data["count"],
                        "sample_sources": t_data["sources"],
                    }
                    for t, t_data in sorted_trans
                ]
                rec = {
                    "anchor_term": data["display_anchor"],
                    "anchor_lang": lang,
                    "canonical_term": anchor_canon,
                    "total_frequency": data["total_freq"],
                    "distinct_translations": len(trans_list),
                    "translations": trans_list,
                }
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return records

    @staticmethod
    def is_valid_acronym(token: str) -> bool:
        """
        Validates whether a token is a standalone acronym:
        - Length >= 2 characters
        - Must contain at least two uppercase ASCII letters (preserves 2-letter acronyms like IT, AI, UI)
        - Strictly uppercase letters, digits, or hyphen (excludes pure digits)
        """
        t = token.strip(PUNCT_STRIP)
        if len(t) < 2:
            return False
        if sum(1 for c in t if c.isupper()) < 2:
            return False
        if not all(c.isupper() or c.isdigit() or c == "-" for c in t):
            return False
        if t.isdigit():
            return False
        return True

    @classmethod
    def extract_and_save_acronym_terms(
        cls, bundles: List[CandidateBundle], out_path: Path
    ) -> List[Dict[str, Any]]:
        """
        Extracts identical cross-lingual standalone acronyms present in both sentences.
        Saves to acronym_terms.jsonl sorted descending by frequency.
        """
        acronym_agg: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "sources": []})

        for b in bundles:
            if not b.candidates:
                continue
            top1 = b.candidates[0]
            tokens_a = ACRONYM_TOKEN_RE.findall(b.a_text)
            tokens_b = ACRONYM_TOKEN_RE.findall(top1.b_text)

            valid_b = {t.lower(): t for t in set(tokens_b) if cls.is_valid_acronym(t)}
            unique_valid_a = {ta for ta in set(tokens_a) if cls.is_valid_acronym(ta)}

            for ta in unique_valid_a:
                if ta.lower() in valid_b:
                    canon = ta.upper()
                    acronym_agg[canon]["count"] += 1
                    src = f"{b.a_sent_id}:{top1.b_sent_id}"
                    if src not in acronym_agg[canon]["sources"] and len(acronym_agg[canon]["sources"]) < 5:
                        acronym_agg[canon]["sources"].append(src)

        sorted_acronyms = sorted(acronym_agg.items(), key=lambda x: (x[1]["count"], x[0]), reverse=True)

        records = []
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for canon, data in sorted_acronyms:
                rec = {
                    "acronym": canon,
                    "surface_a": canon,
                    "surface_b": canon,
                    "frequency": data["count"],
                    "sample_sources": data["sources"],
                    "category": "acronym",
                }
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return records

    @staticmethod
    def save_unmapped_sentences(
        records: List[UnmappedSentence], out_path: Path
    ) -> None:
        """Saves defective sentence records with uncovered multi-word compounds."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                rec = {
                    "source_a_id": r.source_a_id,
                    "source_b_id": r.source_b_id,
                    "a_text": r.a_text,
                    "b_text": r.b_text,
                    "labse_score": round(r.labse_score, 4),
                    "uncovered_compounds": r.uncovered_compounds,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def save_raw_occurrences(pairs: List[RawTermPair], out_path: Path, category: str) -> None:
        """Appends sentence-level occurrence stream to JSONL."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            for p in pairs:
                record = {
                    "source_a_id": p.source_a_id,
                    "source_b_id": p.source_b_id,
                    "surface_a": p.surface_a,
                    "canonical_a": p.canonical_a,
                    "surface_b": p.surface_b,
                    "canonical_b": p.canonical_b,
                    "category": category,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def generate_report(
        cls,
        out_path: Path,
        elapsed_sec: float,
        throughput: float,
        compound_records: List[Dict[str, Any]],
        unified_cluster_records: List[Dict[str, Any]],
        acronym_records: List[Dict[str, Any]],
        defective_sentence_records: List[UnmappedSentence],
        total_sentences: int,
        filtered_tier_count: int = 0,
        include_gray_zone: bool = False,
    ) -> None:
        """Generates a comprehensive markdown summary report."""
        processed_count = max(0, total_sentences - filtered_tier_count)
        defective_pct = (len(defective_sentence_records) / processed_count * 100) if processed_count > 0 else 0.0
        en_hubs = [c for c in unified_cluster_records if c["anchor_lang"] == "en"]
        ko_hubs = [c for c in unified_cluster_records if c["anchor_lang"] == "ko"]
        tier_label = "UNMATCHED 필터링" if include_gray_zone else "GRAY_ZONE / UNMATCHED 필터링"

        lines = [
            "# 경량화 전문용어 추출 파이프라인 최종 결과 리포트",
            "",
            "## 1. 아키텍처 및 산출물 파일 구조",
            "- **`compound_terms.jsonl`**: 양쪽 언어 모두 2어절 이상인 순수 대칭 복합 전문용어 사전",
            "- **`polysemy_clusters.jsonl`**: 단일 통합 다의어/동의어 1:N 클러스터 사전",
            "- **`acronym_terms.jsonl`**: 2자 이상 영문 대문자/숫자 조합 단독 전문 약어 사전",
            "- **`unmapped_sentences.jsonl`**: 복합어 잔여물이 매칭되지 않은 결함 문장 격리 보관 목록",
            "- **`raw_occurrences.jsonl`**: 문장별 원본 출현 기록 전수 보존 스트림",
            "",
            "## 2. 정량적 성능 지표",
            "",
            "| 지표 | 측정값 | 설명 |",
            "| :--- | :---: | :--- |",
            f"| **총 입력 문장 수** | **{total_sentences:,} 문장** | 입력 병렬 코퍼스 |",
            f"| **저신뢰도 제외 문장 수** | **{filtered_tier_count:,} 문장** | {tier_label} |",
            f"| **실제 처리 문장 수** | **{processed_count:,} 문장** | 용어 추출 대상 문장 |",
            f"| **처리 소요 시간 / 속도** | **{elapsed_sec:.2f}초 ({throughput:.1f} 문장/초)** | 실시간 고속 임베딩 매핑 |",
            f"| **고유 대칭 복합 전문용어** | **{len(compound_records):,}개** | 순수 복합어 표제어 |",
            f"| **통합 다의어/동의어 클러스터** | **{len(unified_cluster_records):,}개 허브** | KO {len(ko_hubs):,}개 + EN {len(en_hubs):,}개 |",
            f"| **단독 전문 약어 (Acronyms)** | **{len(acronym_records):,}개** | HPV, ADHD, BMI 등 전용 수록 |",
            f"| **결함/미매칭 격리 문장 수** | **{len(defective_sentence_records):,}문장 ({defective_pct:.1f}%)** | 처리 문장 대비 복합어 잔류 비율 |",
            "",
        ]

        lines.extend(cls._build_acronym_table(acronym_records))
        lines.extend(cls._build_polysemy_table("4. Top-15 영어 앵커 다의어 허브", "EN", "KO", en_hubs))
        lines.extend(cls._build_polysemy_table("5. Top-15 한국어 앵커 다의어 허브", "KO", "EN", ko_hubs))
        lines.extend(cls._build_compound_table(compound_records))

        out_path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _build_acronym_table(acronym_records: List[Dict[str, Any]]) -> List[str]:
        """Builds markdown table for Top-15 acronyms."""
        lines = [
            "## 3. Top-15 단독 전문 약어 (`acronym_terms.jsonl`)",
            "",
            "| 순위 | 전문 약어 (Acronym) | 총 출현빈도 | 대표 샘플 출처 |",
            "| :---: | :--- | :---: | :--- |",
        ]
        for idx, acr in enumerate(acronym_records[:15], 1):
            sample_str = ", ".join(acr["sample_sources"][:2])
            lines.append(f"| {idx} | **`{acr['acronym']}`** | **{acr['frequency']}회** | {sample_str} |")
        lines.append("")
        return lines

    @staticmethod
    def _build_polysemy_table(
        title: str, anchor_lang: str, trans_lang: str, hubs: List[Dict[str, Any]]
    ) -> List[str]:
        """Builds markdown table for polysemy hubs."""
        lines = [
            f"## {title} (`polysemy_clusters.jsonl`)",
            "",
            f"| 순위 | 앵커 어휘 ({anchor_lang}) | 총 출현빈도 | 번역어 분기 ({trans_lang}) | 대표 샘플 출처 |",
            "| :---: | :--- | :---: | :--- | :--- |",
        ]
        for idx, hub in enumerate(hubs[:15], 1):
            trans_str = ", ".join([f"`{t['term']}` ({t['frequency']}회)" for t in hub["translations"][:4]])
            sample_str = hub["translations"][0]["sample_sources"][0] if hub["translations"] and hub["translations"][0]["sample_sources"] else ""
            lines.append(f"| {idx} | **`{hub['anchor_term']}`** | **{hub['total_frequency']}회** | {trans_str} | {sample_str} |")
        lines.append("")
        return lines

    @staticmethod
    def _build_compound_table(compound_records: List[Dict[str, Any]]) -> List[str]:
        """Builds markdown table for Top-20 compound terms."""
        lines = [
            "## 6. Top-20 고유 대칭 복합 전문용어 (`compound_terms.jsonl`)",
            "",
            "| 순위 | 한국어 (KO) | 영어 (EN) | 코퍼스 등장빈도 | 대표 샘플 출처 |",
            "| :---: | :--- | :--- | :---: | :--- |",
        ]
        for idx, rec in enumerate(compound_records[:20], 1):
            samples_str = ", ".join(rec["sample_sources"][:2])
            lines.append(f"| {idx} | `{rec['surface_a']}` | `{rec['surface_b']}` | **{rec['frequency']}회** | {samples_str} |")
        return lines
