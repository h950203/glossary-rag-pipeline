"""Streaming artifacts and identities shared by alignment-lab stages."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

from . import core


STAGES = ("C_bge",)
VERSION = 2


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@contextmanager
def atomic_text(path, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix="." + path.name + ".", delete=False) as writer:
            temporary = Path(writer.name)
            yield writer
            writer.flush()
            os.fsync(writer.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, data, overwrite=False):
    with atomic_text(path, overwrite) as writer:
        json.dump(data, writer, ensure_ascii=False, indent=2, allow_nan=False)
        writer.write("\n")


def write_line(writer, row):
    writer.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as reader:
        for number, line in enumerate(reader, 1):
            try:
                yield json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"손상된 JSONL: {path}:{number}") from exc


def identify(row, corpus_id):
    row["sentence_id"] = digest([row["en"], row["ko"]])[:24]
    row["row_id"] = digest([corpus_id, row["line_id"], row["sentence_id"]])[:24]
    return row


def pair_key(pair):
    # Token IDs and kind intentionally do not participate: segmentations differ.
    return tuple(pair["en_span"] + pair["ko_span"])


def finish_pairs(row, pairs):
    unique = {}
    for pair in pairs:
        for lang in ("en", "ko"):
            start, end = pair[lang + "_span"]
            if not 0 <= start < end <= len(row[lang]) or row[lang][start:end] != pair[lang]:
                raise ValueError(f"line {row['line_id']}: 원문과 후보 구간 불일치")
        key = pair_key(pair)
        pair["pair_id"] = digest([row["row_id"], *key])[:24]
        unique.setdefault(key, pair)
    return sorted(unique.values(), key=pair_key)




def stage_row(row, stage, links, pairs):
    result = dict(row)
    result.update(stage=stage, alignments=[list(p) for p in sorted(set(map(tuple, links)))],
                  pairs=finish_pairs(row, pairs))
    if row["status"] in ("ready", "aligned", "no_alignment"):
        result["status"] = "aligned" if result["pairs"] else "no_alignment"
    return result
