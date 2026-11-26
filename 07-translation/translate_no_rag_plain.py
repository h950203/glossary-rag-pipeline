#!/usr/bin/env python3
"""Translate English lines into Korean with a short, direct prompt."""

import argparse
import csv
import importlib.util
from itertools import islice
import math
from pathlib import Path


BASE_PATH = Path(__file__).with_name("translate_rag_v1.1.py")
BASE_SPEC = importlib.util.spec_from_file_location("plain_translation_backend", BASE_PATH)
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)

TRANSLATION_PROMPT = "Translate the English sentence into natural Korean. Return only the Korean translation."


def build_messages(source):
    return [
        {"role": "system", "content": TRANSLATION_PROMPT},
        {"role": "user", "content": source},
    ]


class PlainTranslator(base.Translator):
    def __init__(self, args):
        super().__init__(args, base.DEFAULT_POLICY)

    def translate(self, sources):
        responses = self.generate_messages([build_messages(source) for source in sources], skip_special_tokens=True)
        if len(responses) != len(sources):
            raise RuntimeError("Generation returned a different number of rows than requested")
        return [
            base.safe_clean(response, "<END_TRANSLATION>", self.tokenizer.all_special_tokens)
            for response in responses
        ]


def run(args):
    txt_path = Path(str(args.output_file) + ".txt")
    csv_path = Path(str(args.output_file) + ".csv")
    for path in (txt_path, csv_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}")

    with args.input_file.open(encoding="utf-8-sig") as source_file:
        total = sum(1 for _ in islice(source_file, args.limit))
        if not total:
            raise ValueError(f"Input has no lines: {args.input_file}")
        source_file.seek(0)
        lines = (line.strip() for line in islice(source_file, args.limit))
        print(f"[Model] Loading {args.model}", flush=True)
        translator = PlainTranslator(args)

        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with txt_path.open("x", encoding="utf-8") as txt, csv_path.open("x", encoding="utf-8", newline="") as log:
            writer = csv.writer(log)
            writer.writerow(["sent_idx", "source_en", "translation_ko", "term_hits", "parse_status", "retry_count", "warnings"])
            processed = 0
            while batch := list(islice(lines, args.batch_size)):
                nonempty = [line for line in batch if line]
                answers = iter(translator.translate(nonempty) if nonempty else ())
                for source in batch:
                    answer, warnings = next(answers) if source else ("", [])
                    status = "parsed" if answer else "invalid_plain_output" if source else "blank_input"
                    output = answer or "-"
                    txt.write(output + "\n")
                    writer.writerow([processed, source, output, "[]", status, 0, "|".join(warnings)])
                    processed += 1
                txt.flush()
                log.flush()
                print(f"[Progress] {processed:,}/{total:,} lines ({processed / total:.1%})", flush=True)

    print(f"[DONE] {processed:,} lines saved to {txt_path} and {csv_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True, help="English TXT with one sentence per line")
    parser.add_argument("--output-file", type=Path, required=True, help="Output stem without .txt or .csv")
    parser.add_argument("--model", default=base.MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=122)
    parser.add_argument("--repetition-penalty", type=float, default=1.0035)
    parser.add_argument("--limit", type=int, help="Process only the first N lines")
    args = parser.parse_args()
    if min(args.batch_size, args.max_new_tokens) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size, --max-new-tokens and --limit must be positive")
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty < 1:
        parser.error("--repetition-penalty must be finite and at least 1")
    try:
        run(args)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
