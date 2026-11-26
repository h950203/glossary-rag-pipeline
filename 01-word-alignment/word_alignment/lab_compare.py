"""Model-free comparison and extraction using original span identities."""
import csv
from itertools import zip_longest
import json
from pathlib import Path
import random
import sys

from .lab_common import atomic_text, pair_key, read_rows, write_line


FIELDS = ("line_id", "sentence_id", "row_id", "stage", "method", "status", "pair_id", "kind", "en", "ko",
          "en_start", "en_end", "ko_start", "ko_end", "score", "change", "present",
          "en_sentence", "ko_sentence")
COMPARABLE = {"aligned", "no_alignment", "empty_text", "empty_tokens"}


def load_manifest(directory):
    path = directory / "manifest.json"
    with path.open(encoding="utf-8") as reader:
        data = json.load(reader)
    if data.get("schema") not in {"alignment-lab-v1", "alignment-lab-bge-v2", "word-alignment-v1.1", "word-alignment-v1.2"}:
        raise ValueError(f"alignment-lab manifest가 아닙니다: {path}")
    return data


def select_stages(manifest, requested=None):
    available = [name for name, data in manifest["stages"].items() if data.get("status") == "complete"]
    if not requested:
        if not available:
            raise ValueError("완료된 단계가 없습니다.")
        return available
    selected = []
    for value in requested:
        if value == "C" and "C_bge" in available:
            value = "C_bge"
        matches = [stage for stage in available if stage == value or stage.split("_", 1)[0] == value]
        if len(matches) != 1:
            raise ValueError(f"완료된 단계를 정확히 지정하세요: {value}; 가능: {', '.join(available)}")
        if matches[0] not in selected:
            selected.append(matches[0])
    return selected


def aligned_rows(directory, manifest, stages):
    streams = [read_rows(directory / manifest["stages"][name]["path"]) for name in stages]
    for rows in zip_longest(*streams):
        if any(row is None for row in rows):
            raise ValueError("단계별 결과 줄 수가 다릅니다.")
        if len({(row["row_id"], row["line_id"], row["en"], row["ko"]) for row in rows}) != 1:
            raise ValueError("단계별 원문/문장 ID가 다릅니다. 서로 다른 실행을 섞을 수 없습니다.")
        yield dict(zip(stages, rows))


def parse_line_ids(values):
    ids = set()
    for value in values or ():
        try:
            if "-" in value:
                left, right = map(int, value.split("-", 1))
                if not 0 < left <= right or right - left > 1000000:
                    raise ValueError
                ids.update(range(left, right + 1))
            else:
                number = int(value)
                if number < 1:
                    raise ValueError
                ids.add(number)
        except ValueError as exc:
            raise ValueError(f"잘못된 line ID/range: {value}") from exc
    return ids


def matches(pair, args):
    for language in ("en", "ko"):
        value = getattr(args, language + "_text", None)
        if value is not None:
            expected, actual = value.casefold(), pair[language].casefold()
            matched = expected == actual if args.exact else expected in actual
            if not matched:
                return False
    return args.kind == "all" or pair["kind"] == args.kind


def comparison(rows, reference, args):
    first = next(iter(rows.values()))
    selected = {stage: [p for p in row["pairs"] if matches(p, args)] for stage, row in rows.items()}
    keys = {stage: {pair_key(p): p for p in pairs} for stage, pairs in selected.items()}
    comparable = all(row["status"] in COMPARABLE for row in rows.values())
    common = set.intersection(*(set(value) for value in keys.values())) if comparable else set()
    reference_keys = set(keys[reference])
    result = {key: first[key] for key in ("line_id", "sentence_id", "row_id", "en", "ko")}
    result.update(comparable=comparable, reference=reference,
                  common=[keys[reference][key] for key in sorted(common)],
                  stages={stage: {"status": rows[stage]["status"], "pairs": pairs} for stage, pairs in selected.items()},
                  differences={stage: {"added": [keys[stage][k] for k in sorted(set(keys[stage]) - reference_keys)],
                                       "removed": [keys[reference][k] for k in sorted(reference_keys - set(keys[stage]))]}
                               for stage in rows if comparable and stage != reference})
    return result


def flatten(result, mode):
    common = {pair_key(pair) for pair in result["common"]}
    reference = {pair_key(p) for p in result["stages"][result["reference"]]["pairs"]}
    for stage, data in result["stages"].items():
        pairs = data["pairs"]
        removed = result["differences"].get(stage, {}).get("removed", [])
        if mode == "common":
            pairs = [p for p in pairs if pair_key(p) in common]
        elif mode == "changed":
            if not result["comparable"] or stage == result["reference"]:
                pairs, removed = [], []
            else:
                pairs = [p for p in pairs if pair_key(p) not in reference]
        for pair, present in [(p, True) for p in pairs] + ([(p, False) for p in removed] if mode == "changed" else []):
            key = pair_key(pair)
            change = ("not_comparable" if not result["comparable"] else "removed" if not present else
                      "common" if key in common else "added" if key not in reference else "shared_with_reference")
            yield {"line_id": result["line_id"], "sentence_id": result["sentence_id"], "row_id": result["row_id"],
                   "stage": stage, "method": stage, "status": data["status"], "pair_id": pair["pair_id"], "kind": pair["kind"],
                   "en": pair["en"], "ko": pair["ko"], "en_start": pair["en_span"][0], "en_end": pair["en_span"][1],
                   "ko_start": pair["ko_span"][0], "ko_end": pair["ko_span"][1],
                   "score": pair["score"] if present else None, "change": change, "present": present,
                   "en_sentence": result["en"], "ko_sentence": result["ko"]}


def iter_comparisons(args, directory, manifest, stages):
    ids = parse_line_ids(args.line_ids)
    if args.sample_size:
        count = manifest["input_counts"]["selected_lines"]
        ids = set(random.Random(args.seed).sample(range(1, count + 1), min(count, args.sample_size)))
    highest_id = max(ids) if ids else None
    reference = select_stages(manifest, [args.reference])[0] if args.reference else stages[0]
    if reference not in stages:
        raise ValueError("--reference는 --stages에 포함되어야 합니다.")
    emitted = 0
    for rows in aligned_rows(directory, manifest, stages):
        first = next(iter(rows.values()))
        if highest_id is not None and first["line_id"] > highest_id:
            break
        if ids and first["line_id"] not in ids:
            continue
        if args.sentence_contains and not any(args.sentence_contains.casefold() in first[lang].casefold() for lang in ("en", "ko")):
            continue
        result = comparison(rows, reference, args)
        if (args.en_text is not None or args.ko_text is not None) and not any(data["pairs"] for data in result["stages"].values()):
            continue
        if args.mode == "common" and not result["common"]:
            continue
        if args.mode == "changed" and not any(data["added"] or data["removed"] for data in result["differences"].values()):
            continue
        yield result
        emitted += 1
        if args.limit and emitted >= args.limit:
            break


def export_comparisons(args):
    directory = args.lab_dir.expanduser().resolve()
    manifest = load_manifest(directory)
    stages = select_stages(manifest, args.stages)
    fields = args.fields or list(FIELDS)
    if set(fields) - set(FIELDS):
        raise ValueError(f"알 수 없는 필드: {set(fields) - set(FIELDS)}; 가능: {', '.join(FIELDS)}")
    if args.fields and args.layout != "pairs" and args.format != "csv":
        raise ValueError("--fields는 --layout pairs 또는 --format csv와 함께 사용하세요.")
    if args.output:
        output = args.output.expanduser().resolve()
        protected = {directory / "manifest.json", directory / "summary.json", directory / "comparison.csv"}
        protected.update((directory / item["path"]).resolve() for item in manifest["stages"].values() if "path" in item)
        # A query must never overwrite inputs, checkpoints, or training evidence.
        managed = output.is_relative_to(directory) and any(
            part in ("stages", "cache", "tokens", "fast", "methods", "tokenizations") for part in output.relative_to(directory).parts[:-1])
        if output in protected or managed:
            raise ValueError("실험 관리 파일에 비교 결과를 덮어쓸 수 없습니다.")
        for key in ("en", "ko"):
            if output == Path(manifest["input"][key]).resolve():
                raise ValueError("입력 파일에 비교 결과를 덮어쓸 수 없습니다.")
        with atomic_text(output, args.overwrite) as writer:
            _export(args, writer, fields, iter_comparisons(args, directory, manifest, stages))
    else:
        _export(args, sys.stdout, fields, iter_comparisons(args, directory, manifest, stages))


def _export(args, writer, fields, results):
    flat = args.layout == "pairs" or args.format == "csv"
    if args.format == "csv":
        csv_writer = csv.DictWriter(writer, fieldnames=fields, extrasaction="ignore")
        csv_writer.writeheader()
    elif args.format == "json":
        writer.write("[\n")
    first = True
    for result in results:
        if flat:
            records = ({field: row[field] for field in fields} for row in flatten(result, args.mode))
        elif args.mode == "common":
            records = [{key: result[key] for key in ("line_id", "sentence_id", "row_id", "en", "ko", "comparable", "common")}]
        elif args.mode == "changed":
            records = [{key: result[key] for key in ("line_id", "sentence_id", "row_id", "en", "ko", "comparable", "reference", "differences")}]
        else:
            records = [result]
        for row in records:
            if args.format == "csv":
                csv_writer.writerow(row)
            elif args.format == "jsonl":
                write_line(writer, row)
            else:
                if not first:
                    writer.write(",\n")
                json.dump(row, writer, ensure_ascii=False, allow_nan=False)
                first = False
    if args.format == "json":
        writer.write("\n]\n")
