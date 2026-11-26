"""
Lightweight Pipeline Module 2: BGE-M3 Dense Embedding Term Mapper.
Maps candidate phrases using dense multilingual embeddings with:
- Contiguous matrix embedding cache & vectorized Mutual Nearest Neighbor (MNN) filtering
- Symmetric multi-word compound detection & conditional boost (+0.04)
- Identical loanword exclusion (score >= 0.9999)
- Single / asymmetric local NMS (highest semantic fidelity claims slot)
- English token-sequence overlap & Korean boundary + constituent-token coverage checking
"""

from typing import Dict, List, Optional, Tuple
import re

import torch
from sentence_transformers import SentenceTransformer

from common.text_utils import has_hangul
from models import CandidateBundle, RawTermPair


class TermMapper:
    """Computes cross-lingual similarity between term candidates and performs NMS selection."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        compound_sim_threshold: float = 0.60,
        single_noun_suspect_min: float = 0.60,
        compound_boost: float = 0.04,
        device: Optional[str] = None,
        shared_model: Optional[SentenceTransformer] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compound_sim_threshold = compound_sim_threshold
        self.single_noun_suspect_min = single_noun_suspect_min
        self.compound_boost = compound_boost

        self.model = shared_model
        if self.model is None:
            try:
                self.model = SentenceTransformer(model_name, device=self.device)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"⚠️ [TermMapper] GPU initialization failed ({e}). Falling back to CPU.")
                self.model = SentenceTransformer(model_name, device="cpu")
                self.device = "cpu"

        self._phrase_to_idx: Dict[str, int] = {}
        self._emb_matrix: Optional[torch.Tensor] = None

    def precompute_embeddings(self, texts: List[str], batch_size: int = 64) -> None:
        """Encodes unique candidates into a contiguous normalized embedding matrix on CPU host memory."""
        missing = [t for t in set(texts) if t and t not in self._phrase_to_idx]
        if not missing:
            return
        with torch.inference_mode():
            embs_np = self.model.encode(
                missing,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
                device=self.device,
            )
        embs = torch.from_numpy(embs_np).to(torch.float32)
        start_idx = 0 if self._emb_matrix is None else int(self._emb_matrix.size(0))
        for offset, text in enumerate(missing):
            self._phrase_to_idx[text] = start_idx + offset
        self._emb_matrix = (
            embs if self._emb_matrix is None else torch.cat([self._emb_matrix, embs], dim=0)
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _token_contains_en(outer: str, inner: str) -> bool:
        """Checks if inner English tokens appear as a contiguous sub-sequence in outer English tokens."""
        o_tokens = outer.lower().split()
        i_tokens = inner.lower().split()
        n_o = len(o_tokens)
        n_i = len(i_tokens)
        if n_i == 0 or n_i > n_o:
            return False
        return any(o_tokens[k : k + n_i] == i_tokens for k in range(n_o - n_i + 1))

    @classmethod
    def _en_overlap(cls, text1: str, text2: str) -> bool:
        """Token-level bidirectional overlap for English phrases."""
        return cls._token_contains_en(text1, text2) or cls._token_contains_en(text2, text1)

    @staticmethod
    def _ko_overlap(text1: str, text2: str) -> bool:
        """Bidirectional substring overlap for Korean phrases."""
        return (text1 in text2) or (text2 in text1)

    @staticmethod
    def _find_spans(text: str, sub: str) -> List[Tuple[int, int]]:
        """Finds all non-overlapping character spans of sub in text (case-insensitive for English)."""
        if not sub or not text:
            return []
        spans = []
        pattern = re.escape(sub)
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            spans.append((m.start(), m.end()))
        return spans

    def _find_mnn_pairs(
        self,
        cands_a: List[str],
        cands_b: List[str],
        min_score: float,
    ) -> List[Tuple[float, str, str]]:
        """Vectorized cosine similarity and Mutual Nearest Neighbor (MNN) filtering between two pools."""
        if not cands_a or not cands_b:
            return []
        assert self._emb_matrix is not None
        ko_idx = [self._phrase_to_idx[k] for k in cands_a]
        en_idx = [self._phrase_to_idx[e] for e in cands_b]
        ko_vecs = self._emb_matrix[ko_idx]
        en_vecs = self._emb_matrix[en_idx]
        sim_matrix = torch.matmul(ko_vecs, en_vecs.T)

        best_e_for_k = sim_matrix.argmax(dim=1)
        best_k_for_e = sim_matrix.argmax(dim=0)
        ko_arange = torch.arange(len(cands_a), device=sim_matrix.device)
        mnn_mask = best_k_for_e[best_e_for_k] == ko_arange

        valid_i = torch.nonzero(mnn_mask, as_tuple=False).squeeze(1)
        if valid_i.numel() == 0:
            return []

        valid_j = best_e_for_k[valid_i]
        valid_scores = sim_matrix[valid_i, valid_j]

        pairs: List[Tuple[float, str, str]] = []
        for i, j, score in zip(valid_i.tolist(), valid_j.tolist(), valid_scores.tolist()):
            k_cand = cands_a[i]
            if score >= 0.9999 or not has_hangul(k_cand) or score < min_score:
                continue
            e_cand = cands_b[j]
            pairs.append((score, k_cand, e_cand))
        return pairs

    def map_bundle(
        self,
        bundle: CandidateBundle,
        ko_cands: List[str],
        en_cands: List[str],
        is_ko_compound_fn,
    ) -> Tuple[List[RawTermPair], List[RawTermPair], List[str]]:
        """
        Maps candidates in a sentence bundle using a 2-stage hierarchical pipeline:
        Stage 1: Multi-word Compound MNN -> Token Span Masking
        Stage 2: Single-word / Residual MNN on unconsumed spans
        """
        if not bundle.candidates:
            return [], [], []

        top1 = bundle.candidates[0]
        if not ko_cands or not en_cands:
            uncovered_c = [k for k in ko_cands if is_ko_compound_fn(k)]
            return [], [], uncovered_c

        needed = [t for t in ko_cands + en_cands if t not in self._phrase_to_idx]
        if needed:
            self.precompute_embeddings(needed)

        ko_sent = bundle.a_text
        en_sent = top1.b_text
        ko_occupied = [False] * len(ko_sent)
        en_occupied = [False] * len(en_sent)

        # ----------------------------------------------------------------------
        # Stage 1: Compound-only Pool (Length >= 2 on both sides)
        # ----------------------------------------------------------------------
        ko_comp = [k for k in ko_cands if is_ko_compound_fn(k)]
        en_comp = [e for e in en_cands if len(e.strip().split()) >= 2]

        selected_pairs: List[RawTermPair] = []

        if ko_comp and en_comp:
            comp_pairs = self._find_mnn_pairs(
                ko_comp, en_comp, min_score=self.compound_sim_threshold
            )
            # Sort: Length descending first, then similarity score with boost
            comp_pairs.sort(
                key=lambda x: (
                    len(x[1].split()) + len(x[2].split()),
                    x[0] + self.compound_boost,
                    x[0],
                ),
                reverse=True,
            )

            for score, k_cand, e_cand in comp_pairs:
                k_spans = self._find_spans(ko_sent, k_cand)
                e_spans = self._find_spans(en_sent, e_cand)

                valid_k = next((sp for sp in k_spans if not any(ko_occupied[sp[0] : sp[1]])), None)
                valid_e = next((sp for sp in e_spans if not any(en_occupied[sp[0] : sp[1]])), None)

                if valid_k is not None and valid_e is not None:
                    selected_pairs.append(
                        RawTermPair(
                            surface_a=k_cand,
                            canonical_a=k_cand,
                            surface_b=e_cand,
                            canonical_b=e_cand.lower(),
                            source_a_id=bundle.a_sent_id,
                            source_b_id=top1.b_sent_id,
                        )
                    )
                    for idx in range(valid_k[0], valid_k[1]):
                        ko_occupied[idx] = True
                    for idx in range(valid_e[0], valid_e[1]):
                        en_occupied[idx] = True

        # ----------------------------------------------------------------------
        # Stage 2: Single-word / Residual Pool (Only unconsumed token spans)
        # ----------------------------------------------------------------------
        ko_residual = [
            k for k in ko_cands
            if any(not any(ko_occupied[sp[0] : sp[1]]) for sp in self._find_spans(ko_sent, k))
        ]
        en_residual = [
            e for e in en_cands
            if any(not any(en_occupied[sp[0] : sp[1]]) for sp in self._find_spans(en_sent, e))
        ]

        selected_polysemy: List[RawTermPair] = []

        if ko_residual and en_residual:
            single_pairs = self._find_mnn_pairs(
                ko_residual, en_residual, min_score=self.single_noun_suspect_min
            )
            single_pairs.sort(key=lambda x: x[0], reverse=True)

            for score, k_cand, e_cand in single_pairs:
                k_spans = self._find_spans(ko_sent, k_cand)
                e_spans = self._find_spans(en_sent, e_cand)

                valid_k = next((sp for sp in k_spans if not any(ko_occupied[sp[0] : sp[1]])), None)
                valid_e = next((sp for sp in e_spans if not any(en_occupied[sp[0] : sp[1]])), None)

                if valid_k is not None and valid_e is not None:
                    selected_polysemy.append(
                        RawTermPair(
                            surface_a=k_cand,
                            canonical_a=k_cand,
                            surface_b=e_cand,
                            canonical_b=e_cand.lower(),
                            source_a_id=bundle.a_sent_id,
                            source_b_id=top1.b_sent_id,
                        )
                    )
                    for idx in range(valid_k[0], valid_k[1]):
                        ko_occupied[idx] = True
                    for idx in range(valid_e[0], valid_e[1]):
                        en_occupied[idx] = True

        uncovered_compounds = self._find_uncovered_compounds(
            ko_cands=ko_cands,
            compounds=selected_pairs,
            polysemy=selected_polysemy,
            is_ko_compound_fn=is_ko_compound_fn,
            ko_sent=ko_sent,
            ko_occupied=ko_occupied,
        )

        return selected_pairs, selected_polysemy, uncovered_compounds

    def _find_uncovered_compounds(
        self,
        ko_cands: List[str],
        compounds: List[RawTermPair],
        polysemy: List[RawTermPair],
        is_ko_compound_fn,
        ko_sent: str,
        ko_occupied: List[bool],
    ) -> List[str]:
        """
        Finds genuine multi-word compounds that are entirely unmapped and unconsumed in the sentence.
        Filters out candidates that have:
        1. Substring overlap with matched surface terms
        2. Stem / constituent token containment (e.g. '탄력' in '탄력성')
        3. Physical span overlap with already occupied sentence characters
        """
        all_matched_ko = {p.surface_a for p in compounds + polysemy}
        matched_tokens = set()
        for m in all_matched_ko:
            matched_tokens.update(m.split())

        uncovered: List[str] = []
        for k in ko_cands:
            if not is_ko_compound_fn(k):
                continue

            # 1. Surface string overlap with any matched Korean phrase
            if any((k in m) or (m in k) for m in all_matched_ko):
                continue

            # 2. Token / stem containment overlap (handles affixes like '탄력' <-> '탄력성')
            k_tokens = k.split()
            if any(any((w in m_tok) or (m_tok in w) for m_tok in matched_tokens) for w in k_tokens):
                continue

            # 3. Sentence character span occupancy check
            spans = self._find_spans(ko_sent, k)
            if spans and any(any(ko_occupied[s:e]) for s, e in spans):
                continue

            uncovered.append(k)

        return uncovered
