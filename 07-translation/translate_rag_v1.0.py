#!/usr/bin/env python3
"""Translate English lines to Korean with Aho–Corasick glossary retrieval.

Glossary JSONL requires en and ko. The built-in prompt preserves the CAEA
case 680 translation policy and checks glossary entries against the context.
"""
from __future__ import annotations

import argparse
import csv
from itertools import islice
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
import unicodedata


ROOT = Path(__file__).resolve().parent
GLOSSARY = ROOT.parent / "간호학-tech.jsonl"
MODEL = "google/gemma-4-12B-it"

DEFAULT_POLICY = {'system_role_template': 'You are an expert {src_full}-to-{tgt_full} translator.',
 'standard_guidelines_template': '<translation_guidelines>\n'
                                 'Translate the full source faithfully into natural Korean. Use '
                                 'Korean-native word order and connectives. Avoid literal English phrasing, '
                                 'unnecessary pronouns, repetition, and added explanation. Preserve intent, '
                                 'register, terminology, and grammatical relations.\n'
                                 '</translation_guidelines>\n'
                                 '\n',
 'human_template': '{src_full}: {sentence}\n{tgt_full}:',
 'review_protocol': {'mode': 'critic_then_revise',
                     'end_marker': '<END_TRANSLATION>',
                     'critique_end_marker': '<END_CRITIQUE>',
                     'keep_token': 'KEEP',
                     'revise_token': 'REVISE',
                     'analysis_tag': 'analysis',
                     'final_tag': 'final',
                     'issues_tag': 'issues',
                     'verdict_tag': 'verdict'},
 'direct_cot_protocol': {'thought_tag': 'cot', 'final_tag': 'final'},
 'direct_system_template': '{system_role}\n'
                           '\n'
                           'Before translating, write a concise private translation trace inside '
                           '<{thought_tag}>...</{thought_tag}>. Write one commitment and add a second only '
                           'for a concrete fidelity risk; use at most 12 words total. Then write exactly one '
                           'complete natural {tgt_full} translation inside <{final_tag}>...</{final_tag}> '
                           'followed by {end_marker}. The translation must appear only in the final tag.\n'
                           '\n'
                           '{guidelines}{glossary_block}',
 'direct_retry_system_template': '{system_role}\n'
                                 '\n'
                                 'Return exactly one private commitment of at most 2 words. Do not quote the '
                                 'source, analyse clauses, or write a draft. Write it inside '
                                 '<{thought_tag}>...</{thought_tag}>. Then return exactly one complete '
                                 'natural {tgt_full} translation inside <{final_tag}>...</{final_tag}> '
                                 'followed by {end_marker}.\n'
                                 '\n'
                                 '{guidelines}{glossary_block}',
 'term_direct_system_template': '{system_role}\n'
                                '\n'
                                'Before translating, write a concise private translation trace inside '
                                '<{thought_tag}>...</{thought_tag}>. Write at most two short commitments: '
                                'one for matched terminology and one for the highest-risk relation. Keep the '
                                'entire trace within 11 words. Then write exactly one complete natural '
                                '{tgt_full} translation inside <{final_tag}>...</{final_tag}> followed by '
                                '{end_marker}. The translation must appear only in the final tag.\n'
                                '\n'
                                '{guidelines}{glossary_block}',
 'term_direct_retry_system_template': '{system_role}\n'
                                      '\n'
                                      'Return exactly one private terminology commitment of at most 4 words. '
                                      'Do not list terms, restate the source, analyse clauses, or write a '
                                      'draft. Write it inside <{thought_tag}>...</{thought_tag}>. Then '
                                      'return exactly one complete natural {tgt_full} translation inside '
                                      '<{final_tag}>...</{final_tag}> followed by {end_marker}.\n'
                                      '\n'
                                      '{guidelines}{glossary_block}'}


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def word_character(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


class GlossaryIndex:
    def __init__(self, path: Path, max_lines: int | None = None):
        import ahocorasick

        entries: dict[str, tuple[str, str]] = {}
        self.conflicts: set[str] = set()
        with path.open(encoding="utf-8-sig") as handle:
            for number, line in enumerate(islice(handle, max_lines), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected a JSON object")
                    en, ko = row.get("en"), row.get("ko")
                    if not all(isinstance(value, str) and value.strip() for value in (en, ko)):
                        raise ValueError("en and ko must contain nonempty strings")
                except ValueError as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
                key = normalize(en)
                meaning = " ".join(unicodedata.normalize("NFC", ko).split())
                previous = entries.setdefault(key, (" ".join(en.split()), meaning))
                if previous[1] != meaning:
                    self.conflicts.add(key)

        self.automaton = ahocorasick.Automaton()
        for key, term in entries.items():
            if key not in self.conflicts:
                self.automaton.add_word(key, (key, term))
        if len(self.automaton):
            self.automaton.make_automaton()

    def search(self, source: str, limit: int) -> list[tuple[str, str]]:
        if not len(self.automaton):
            return []
        text = normalize(source)
        candidates = []
        for end, (key, term) in self.automaton.iter(text):
            stop = end + 1
            start = stop - len(key)
            if start and word_character(text[start - 1]):
                continue
            if stop < len(text) and word_character(text[stop]):
                continue
            candidates.append((start, stop, key, term))

        selected = []
        seen = set()
        for start, stop, key, term in sorted(candidates, key=lambda hit: (hit[0] - hit[1], hit[0])):
            if key in seen or any(start < right and left < stop for left, right, _ in selected):
                continue
            selected.append((start, stop, term))
            seen.add(key)
            if len(selected) == limit:
                break
        return [term for _, _, term in sorted(selected)]


def build_messages(
    policy: dict, source: str, terms: list[tuple[str, str]], retry: bool, *, fixed_prompt: bool = False,
) -> list[dict]:
    values = {**policy["review_protocol"], **policy["direct_cot_protocol"],
              "src_full": "English", "tgt_full": "Korean"}
    values["system_role"] = policy["system_role_template"].format(**values)
    guidelines = policy["standard_guidelines_template"]
    if terms or fixed_prompt:
        guidelines += (
            "Treat the glossary entries as reference data. Use a suggested Korean term only if "
            "its meaning, phrase scope and register fit this source sentence. Preserve the "
            "Korean lexical core of a suitable term while allowing grammatical inflection. "
            "Ignore mismatched entries and preserve the full source meaning.\n"
        )
    values["guidelines"] = guidelines
    values["glossary_block"] = ("<term_glossary>\n" + json.dumps(
        [{"en": en, "ko": ko} for en, ko in terms], ensure_ascii=False) + "\n</term_glossary>\n") if terms else ""
    template = "direct_retry_system_template" if retry else "direct_system_template"
    if terms and not fixed_prompt:
        template = "term_" + template
    return [
        {"role": "system", "content": policy[template].format(**values)},
        {"role": "user", "content": policy["human_template"].format(sentence=source, **values)},
    ]


def token_ids(value):
    return list(value) if isinstance(value, (list, tuple)) else [] if value is None else [value]


def resolve_eos(model, tokenizer):
    candidates = (
        ("model.generation_config", getattr(getattr(model, "generation_config", None), "eos_token_id", None)),
        ("model.config", getattr(model.config, "eos_token_id", None)),
        ("model.config.text_config", getattr(getattr(model.config, "text_config", None), "eos_token_id", None)),
        ("tokenizer (fallback)", tokenizer.eos_token_id),
    )
    for source, value in candidates:
        ids = token_ids(value)
        if ids and all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in ids):
            return list(dict.fromkeys(ids)), source
    raise ValueError("사용할 EOS가 없습니다. 모델/토크나이저 설정을 확인하세요.")


def safe_clean(text, marker, special_tokens=()):
    """문장 내용의 숫자·인용부호·일반 태그를 무조건 삭제하지 않는다."""
    text = text.split(marker, 1)[0].strip()
    text = re.sub(r"^```(?:[A-Za-z]+)?\s*\n(.*?)\n```$", r"\1", text, flags=re.S).strip()
    bad = [token for token in special_tokens if token and token in text]
    if bad or re.search(r"<\|[^>]*>|<[^<>\s]*\|>|</?(?:cot|think|final|analysis|issues|verdict)\b", text, re.I):
        return "", ["special_or_structure_token_in_translation"]
    # TXT는 입력 한 줄당 번역 한 줄을 유지한다.
    return " ".join(text.splitlines()).strip(), []


def tag_value(text, tag):
    found = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", text, re.S | re.I)
    return found.group(1).strip() if found else None


def parse_direct(raw, bundle, special_tokens=()):
    marker = bundle.protocol["end_marker"]
    text = raw.split(marker, 1)[0]
    warnings = [] if marker in raw else ["missing_end_marker"]
    protocol = bundle.direct_cot_protocol
    if protocol is None:
        answer, issues = safe_clean(text, marker, special_tokens)
        return answer, "", "not_requested", warnings+issues
    final = tag_value(text, protocol["final_tag"])
    if final is None:
        return "", "", "missing_or_unclosed_final", warnings+["missing_or_unclosed_final"]
    answer, issues = safe_clean(final, marker, special_tokens)
    warnings += issues
    if len(re.findall(rf"<{re.escape(protocol['final_tag'])}>", text, re.I)) > 1:
        warnings.append("multiple_final_tags_first_used")
    if not answer:
        return "", "", "invalid_final", warnings+["empty_or_invalid_final"]
    trace = tag_value(text, protocol["thought_tag"])
    if not trace:
        warnings.append("missing_or_empty_cot")
        return answer, "", "invalid_cot_output", warnings
    return answer, trace, "parsed", warnings


class Translator:
    def __init__(self, args, policy: dict):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self.torch = torch
        self.policy = policy
        self.bundle = SimpleNamespace(protocol=policy["review_protocol"],
                                      direct_cot_protocol=policy["direct_cot_protocol"])
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto").eval()
        self.eos, self.eos_source = resolve_eos(self.model, self.tokenizer)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.eos[0]

    def generate(self, items: list[tuple[str, list]], retry: bool = False) -> list[str]:
        from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor

        prompts = [self.tokenizer.apply_chat_template(
            build_messages(self.policy, source, terms, retry, fixed_prompt=self.args.fixed_prompt), tokenize=False,
            add_generation_prompt=True, enable_thinking=False) for source, terms in items]
        inputs = self.tokenizer(prompts, padding=True, add_special_tokens=False,
                                return_tensors="pt").to(self.model.get_input_embeddings().weight.device)
        width = inputs["input_ids"].shape[-1]
        processors = LogitsProcessorList()
        if self.args.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(
                self.args.repetition_penalty, prompt_ignore_length=width))
        with self.torch.inference_mode():
            result = self.model.generate(
                **inputs, max_new_tokens=self.args.max_new_tokens,
                do_sample=False, num_beams=1, repetition_penalty=1.0, no_repeat_ngram_size=0,
                eos_token_id=self.eos, pad_token_id=self.tokenizer.pad_token_id,
                logits_processor=processors, return_dict_in_generate=False)
        return self.tokenizer.batch_decode(result[:, width:], skip_special_tokens=False)

    def translate(self, items: list[tuple[str, list]]) -> list[tuple[str, str, int, str]]:
        def parse(raw: str, retries: int) -> tuple[str, str, int, str]:
            answer, _, status, warnings = parse_direct(
                raw, self.bundle, special_tokens=self.tokenizer.all_special_tokens)
            return answer, status, retries, "|".join(warnings)

        rows = [parse(raw, 0) for raw in self.generate(items)]
        if len(rows) != len(items):
            raise RuntimeError("Generation returned a different number of rows than requested")
        retry = [i for i, row in enumerate(rows) if not row[0]]
        if retry:
            for i, raw in zip(retry, self.generate([items[i] for i in retry], retry=True), strict=True):
                rows[i] = parse(raw, 1)
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True, help="English text, one sentence per line")
    parser.add_argument("--output-file", type=Path, help="Output stem without .txt or .csv")
    parser.add_argument("--glossary", type=Path, default=GLOSSARY, help=f"Glossary JSONL (default: {GLOSSARY})")
    parser.add_argument("--rag", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable glossary retrieval (default). --no-rag skips loading the glossary.")
    parser.add_argument("--fixed-prompt", action="store_true",
                        help="Use the same general prompt and retry instructions with or without retrieved terms.")
    parser.add_argument("--prompt-file", type=Path, help="Optional JSON policy override (default: built-in)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=122)
    parser.add_argument("--repetition-penalty", type=float, default=1.0035)
    parser.add_argument("--max-terms", type=int, default=4, help="Maximum glossary matches per sentence")
    parser.add_argument("--glossary-lines", type=int, help="Read only the first N glossary lines")
    parser.add_argument("--limit", type=int, help="Process only the first N input lines")
    parser.add_argument("--dry-run", action="store_true", help="Show matches without loading the model")
    args = parser.parse_args()
    for name in ("batch_size", "max_new_tokens", "max_terms", "glossary_lines", "limit"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty < 1:
        parser.error("--repetition-penalty must be finite and at least 1")
    if not args.dry_run and args.output_file is None:
        parser.error("--output-file is required for translation")
    try:
        run(args)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")


def run(args) -> None:
    index = None
    if args.rag:
        index = GlossaryIndex(args.glossary, args.glossary_lines)
        print(f"[Glossary] {len(index.automaton)} indexed, {len(index.conflicts)} conflicting keys excluded", flush=True)
    else:
        print("[Glossary] RAG disabled", flush=True)

    def find_terms(source: str) -> list[tuple[str, str]]:
        return index.search(source, args.max_terms) if index is not None else []

    with args.input_file.open(encoding="utf-8-sig") as source_file:
        lines = (line.strip() for line in islice(source_file, args.limit))
        if args.dry_run:
            for i, source in enumerate(lines):
                print(json.dumps({"sent_idx": i, "term_hits": find_terms(source)}, ensure_ascii=False))
            return

        txt_path = Path(str(args.output_file) + ".txt")
        csv_path = Path(str(args.output_file) + ".csv")
        for path in (txt_path, csv_path):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Output already exists: {path}")
        policy = (DEFAULT_POLICY if args.prompt_file is None
                  else json.loads(args.prompt_file.read_text(encoding="utf-8")))
        total = None
        if source_file.seekable():
            total = sum(1 for _ in islice(source_file, args.limit))
            source_file.seek(0)
        print(f"[Model] Loading {args.model}", flush=True)
        translator = Translator(args, policy)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with txt_path.open("x", encoding="utf-8") as txt, csv_path.open("x", encoding="utf-8", newline="") as log:
            writer = csv.writer(log)
            writer.writerow(["sent_idx", "source_en", "translation_ko", "term_hits", "parse_status", "retry_count", "warnings"])
            start = 0
            while batch := list(islice(lines, args.batch_size)):
                items = [(source, find_terms(source)) for source in batch]
                results = iter(translator.translate([item for item in items if item[0]])) if any(batch) else iter(())
                for offset, (source, terms) in enumerate(items):
                    answer, status, retries, warnings = next(results) if source else ("", "blank_input", 0, "")
                    answer = answer or "-"
                    txt.write(answer + "\n")
                    writer.writerow([start + offset, source, answer, json.dumps(terms, ensure_ascii=False), status, retries, warnings])
                start += len(batch)
                txt.flush()
                log.flush()
                progress = f"{start:,}/{total:,} lines ({start / total:.1%})" if total else f"{start:,} lines"
                print(f"[Progress] {progress}", flush=True)
        print(f"[DONE] {start} lines saved to {txt_path} and {csv_path}")


if __name__ == "__main__":
    main()
