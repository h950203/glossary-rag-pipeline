"""Contextual BGE-M3 span scoring and explicit, uncalibrated lab policies.

Loads the official encoder and colbert_linear.pt without FlagEmbedding's
training dependencies. See FlagEmbedding's EncoderOnlyEmbedderM3Model.
"""
from pathlib import Path

from . import core
from .backends import TooLong
from .lab_common import pair_key, stage_row


class CandidateLimit(ValueError):
    pass


class BGEEncoder:
    def __init__(self, args, resolved_revision=None):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.torch, self.device = torch, args.device
        torch.set_num_threads(args.threads)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA를 사용할 수 없습니다. --device cpu 또는 CUDA 환경을 확인하세요.")
        options = {"cache_dir": str(args.model_cache), "local_files_only": args.local_files_only,
                   "revision": resolved_revision or args.bge_revision, "trust_remote_code": False}
        config = AutoConfig.from_pretrained(args.bge_model, **options)
        self.revision = getattr(config, "_commit_hash", None)
        if self.revision:
            options["revision"] = self.revision
        self.tokenizer = AutoTokenizer.from_pretrained(args.bge_model, use_fast=True, **options)
        if not self.tokenizer.is_fast:
            raise ValueError("BGE 원문 위치 매핑에는 fast tokenizer가 필요합니다.")
        dtype = torch.float16 if args.fp16 else torch.float32
        self.model = AutoModel.from_pretrained(args.bge_model, torch_dtype=dtype, **options).to(args.device).eval()
        local = Path(args.bge_model).expanduser()
        if local.is_dir():
            head = local / "colbert_linear.pt"
        else:
            head = Path(hf_hub_download(args.bge_model, "colbert_linear.pt",
                                       revision=options["revision"], cache_dir=str(args.model_cache),
                                       local_files_only=args.local_files_only))
        # A missing projection is an error, never a random or dense-head fallback.
        state = torch.load(head, map_location="cpu", weights_only=True)
        weight = state["weight"]
        if weight.ndim != 2 or weight.shape[1] != config.hidden_size:
            raise ValueError("colbert_linear.pt와 encoder 차원이 일치하지 않습니다.")
        self.projection = torch.nn.Linear(weight.shape[1], weight.shape[0], bias="bias" in state)
        self.projection.load_state_dict(state, strict=True)
        self.projection = self.projection.to(device=args.device, dtype=dtype).eval()
        position_limit = getattr(config, "max_position_embeddings", args.max_model_tokens)
        if config.model_type in ("xlm-roberta", "roberta"):
            position_limit -= (config.pad_token_id or 0) + 1
        self.limit = min(args.max_model_tokens, self.tokenizer.model_max_length, position_limit)
        from .lab_common import file_hash
        self.metadata = {"model": args.bge_model, "resolved_revision": self.revision,
                         "head_sha256": file_hash(head), "representation": "normalized colbert projection",
                         "pooling": "character-overlap weighted span mean, L2 normalized",
                         "device": args.device, "fp16": args.fp16, "max_model_tokens": self.limit,
                         "prefix": None, "scores_are_probabilities": False}

    def encode_pair(self, english, korean):
        torch = self.torch
        encoded = self.tokenizer([english, korean], padding=True, truncation=False,
                                 return_offsets_mapping=True, return_special_tokens_mask=True,
                                 return_tensors="pt")
        lengths = encoded["attention_mask"].sum(1).tolist()
        if any(length > self.limit for length in lengths):
            raise TooLong(f"BGE 입력 {lengths} tokens, 상한 {self.limit}; 자동 절단하지 않음")
        offsets = encoded.pop("offset_mapping").tolist()
        specials = encoded.pop("special_tokens_mask").tolist()
        inputs = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            vectors = torch.nn.functional.normalize(self.projection(hidden), dim=-1).float().cpu().numpy()
            dense = torch.nn.functional.normalize(hidden[:, 0].float(), dim=-1)
            similarity = float((dense[0] * dense[1]).sum().cpu())
        features = []
        for index, text in enumerate((english, korean)):
            positions = [i for i, (start, end) in enumerate(offsets[index])
                         if i < lengths[index] and not specials[index][i] and end > start]
            spans = [offsets[index][i] for i in positions]
            if any(not 0 <= start < end <= len(text) for start, end in spans):
                raise ValueError("BGE tokenizer offset이 원문 범위를 벗어났습니다.")
            features.append({"text": text, "offsets": spans, "vectors": vectors[index, positions]})
        return features, similarity


def pool_span(features, start, end):
    import numpy as np
    text = features["text"]
    positions, weights, covered, shared = [], [], set(), False
    for index, (left, right) in enumerate(features["offsets"]):
        overlap = [p for p in range(max(start, left), min(end, right)) if not text[p].isspace()]
        if overlap:
            positions.append(index)
            weights.append(len(overlap))
            covered.update(overlap)
            shared |= bool(text[left:start].strip() if left < start else "")
            shared |= bool(text[end:right].strip() if right > end else "")
    required = {p for p in range(start, end) if not text[p].isspace()}
    if not required or not required <= covered:
        return None, shared
    pooled = np.average(features["vectors"][positions], axis=0, weights=weights).astype("float32")
    norm = float(np.linalg.norm(pooled))
    return (pooled / norm if norm > 0 else None), shared


def punctuation(text):
    return not any(c.isalnum() for c in text)


def spans_for(row, language, max_tokens, cap):
    tokens = row["tokens"][language]
    spans = []
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + max_tokens) + 1):
            if end > start + 1 and any(punctuation(t["text"]) for t in tokens[start:end]):
                break
            spans.append((start, end))
            if len(spans) > cap:
                raise CandidateLimit(f"{language} 구간 후보가 상한 {cap}개 초과; --max-span-candidates 조정")
    return spans


def span_vectors(row, language, spans, features):
    import numpy as np
    vectors, valid, shared = [], [], []
    tokens = row["tokens"][language]
    for span in spans:
        start, end = tokens[span[0]]["start"], tokens[span[1] - 1]["end"]
        vector, boundary = pool_span(features, start, end)
        if vector is not None:
            vectors.append(vector)
            valid.append(span)
            shared.append(boundary)
    return (np.stack(vectors) if vectors else None), valid, shared


def overlap(left, right):
    return left[0] < right[1] and right[0] < left[1]


def alternative_margin(scores, spans, chosen):
    alternatives = [float(score) for index, score in enumerate(scores)
                    if not overlap(spans[index], spans[chosen]) and score > -2]
    return float(scores[chosen]) - max(alternatives) if alternatives else 2.0


def mask_punctuation(row, es, ks, matrix):
    import numpy as np
    texts = [np.array([row[lang][row["tokens"][lang][a]["start"]:row["tokens"][lang][b - 1]["end"]]
                       for a, b in spans], dtype=object) for lang, spans in (("en", es), ("ko", ks))]
    ep, kp = (np.array([punctuation(text) for text in values]) for values in texts)
    incompatible = (ep[:, None] | kp[None, :]) & (texts[0][:, None] != texts[1][None, :])
    result = matrix.copy()
    result[incompatible] = -2
    return result


def make_proposals(row, features, args):
    """Mutual-best span matches, followed by disjoint-span greedy decoding.

    Overlapping variants are not counted as independent margin competitors.
    One-to-many is represented as a span, not invented Cartesian token links.
    """
    en_spans = spans_for(row, "en", args.bge_span_tokens, args.max_span_candidates)
    ko_spans = spans_for(row, "ko", args.bge_span_tokens, args.max_span_candidates)
    ev, es, eb = span_vectors(row, "en", en_spans, features[0])
    kv, ks, kb = span_vectors(row, "ko", ko_spans, features[1])
    diagnostic = {"span_candidates": {"en": len(en_spans), "ko": len(ko_spans)},
                  "uncovered_spans": {"en": len(en_spans) - len(es), "ko": len(ko_spans) - len(ks)},
                  "shared_subword_spans": {"en": sum(eb), "ko": sum(kb)}}
    if ev is None or kv is None:
        return [], diagnostic
    matrix = mask_punctuation(row, es, ks, ev @ kv.T)
    forward, reverse = matrix.argmax(axis=1), matrix.argmax(axis=0)
    proposals = []
    for i, j in enumerate(forward.tolist()):
        score = float(matrix[i, j])
        if int(reverse[j]) != i or score < args.min_similarity:
            continue
        margin = min(alternative_margin(matrix[i], ks, j), alternative_margin(matrix[:, j], es, i))
        if margin < args.margin:
            continue
        a, b = es[i]
        c, d = ks[j]
        pair = core.candidate(row, a, b, c, d, "word" if b - a == d - c == 1 else "phrase")
        pair.update(score=score, evidence={"source": "bge_mutual_best_span", "margin": margin,
                                         "shared_subword_boundary": eb[i] or kb[j]})
        proposals.append(pair)
    proposals.sort(key=lambda p: (-p["score"], len(p["en_token_ids"]) + len(p["ko_token_ids"]), pair_key(p)))
    return proposals, diagnostic


def decode_disjoint(proposals):
    used_en, used_ko, accepted = set(), set(), []
    for pair in proposals:
        en, ko = set(pair["en_token_ids"]), set(pair["ko_token_ids"])
        if used_en & en or used_ko & ko:
            continue
        accepted.append(pair)
        used_en.update(en)
        used_ko.update(ko)
    return accepted


def align_row(row, encoder, args, encoded=None):
    """Independent C alignment; no statistical links or fast_align dependency."""
    method = row["method"]
    if row["status"] not in {"ready", "aligned", "no_alignment"}:
        return stage_row(row, method, [], [])
    try:
        if isinstance(encoded, Exception):
            raise encoded
        features, sentence_score = encoded if encoded is not None else encoder.encode_pair(row["en"], row["ko"])
        proposals, diagnostics = make_proposals(row, features, args)
    except (TooLong, CandidateLimit) as exc:
        result = stage_row(row, method, [], [])
        result.update(status="too_long" if isinstance(exc, TooLong) else "too_many_candidates", reason=str(exc))
        return result
    accepted = decode_disjoint(proposals)
    links = [(p["en_token_ids"][0], p["ko_token_ids"][0]) for p in accepted if p["kind"] == "word"]
    result = stage_row(row, method, links, accepted)
    result["span_alignments"] = [{"en_token_ids": p["en_token_ids"], "ko_token_ids": p["ko_token_ids"]}
                                for p in accepted if p["kind"] == "phrase"]
    result["bge_diagnostics"] = {**diagnostics, "sentence_dense_cosine": sentence_score}
    from .term_alignment import align_term_units
    result["term_pairs"], result["term_diagnostics"] = align_term_units(row, features, args)
    from .candidate_review import alternatives
    candidates, pooled = alternatives(row, features, args, result["term_pairs"])
    result["term_candidate_review"] = {"candidates": candidates, "pooled": pooled}
    return result
