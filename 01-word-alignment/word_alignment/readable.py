"""Streaming, model-free views of every sentence in an alignment JSONL."""
from collections import Counter
from contextlib import ExitStack, contextmanager
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path

from .lab_common import atomic_text, file_hash, write_json, write_line
from .lab_compare import load_manifest, select_stages


@contextmanager
def result_lock(root):
    """Share the runner lock so exports cannot race another run/export."""
    root = Path(root)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"같은 출력 폴더에서 작업이 실행 중입니다: {root}") from exc
        yield


def export_file(source, destination, include_punctuation=False, expected_hash=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    names = ("alignments-readable.txt", "alignments-simple.jsonl", "pairs.csv", "readable-export.json")
    paths = [destination / name for name in names]
    if any(path.resolve() == source for path in paths):
        raise ValueError("읽기용 출력으로 원본 JSONL을 덮어쓸 수 없습니다")
    totals, statuses = Counter(), Counter()
    checksum = hashlib.sha256()
    initial = source.stat()
    # Each output is published atomically only after every input row validates.
    # Metadata is written last; it records the exact input and file checksums.
    with ExitStack() as stack:
        reader = stack.enter_context(source.open("rb"))
        text = stack.enter_context(atomic_text(paths[0], overwrite=True))
        simple = stack.enter_context(atomic_text(paths[1], overwrite=True))
        csv_file = stack.enter_context(atomic_text(paths[2], overwrite=True))
        columns = ["line_id", "method", "status", "ko_term", "en_term", "score", "kind", "ko_sentence", "en_sentence"]
        table = csv.DictWriter(csv_file, fieldnames=columns)
        table.writeheader()
        for number, raw in enumerate(reader, 1):
            checksum.update(raw)
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("행이 JSON 객체가 아닙니다")
                line_id, korean, english = row["line_id"], row["ko"], row["en"]
                method = row.get("method", row.get("stage", ""))
                status = row.get("status", "unknown")
                if not isinstance(korean, str) or not isinstance(english, str):
                    raise ValueError("원문은 문자열이어야 합니다")
                pairs = []
                for pair in row.get("pairs", []):
                    ko, en, kind = pair["ko"], pair["en"], pair["kind"]
                    score = pair["score"]
                    if not isinstance(ko, str) or not isinstance(en, str):
                        raise ValueError("용어는 문자열이어야 합니다")
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        raise ValueError("유사도는 유한한 숫자여야 합니다")
                    totals["raw_pairs"] += 1
                    if not include_punctuation and not any(c.isalnum() for c in ko + en):
                        totals["hidden_punctuation_pairs"] += 1
                        continue
                    pairs.append({"ko": ko, "en": en, "score": score, "kind": kind})
                record = {"line_id": line_id, "method": method, "status": status,
                          "ko": korean, "en": english, "pairs": pairs}
                write_line(simple, record)
                text.write(f"\n[{line_id}] {method} | status={status}\nKO: {korean}\nEN: {english}\n")
                for pair in pairs:
                    text.write(f"  {pair['ko']} ↔ {pair['en']}  | 유사도 {pair['score']:.3f}  | {pair['kind']}\n")
                    table.writerow({"line_id": line_id, "method": method, "status": status,
                                    "ko_term": pair["ko"], "en_term": pair["en"],
                                    "score": pair["score"], "kind": pair["kind"],
                                    "ko_sentence": korean, "en_sentence": english})
                if not pairs:
                    text.write("  (표시할 대응쌍 없음)\n")
                    # Retain sentences with no visible pairs in the CSV too.
                    table.writerow({"line_id": line_id, "method": method, "status": status,
                                    "ko_sentence": korean, "en_sentence": english})
                    totals["sentences_without_visible_pairs"] += 1
                totals["sentences"] += 1
                totals["visible_pairs"] += len(pairs)
                statuses[status] += 1
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"읽기용 변환 실패: {source}:{number}: {exc}") from exc
            if number % 5000 == 0:
                print(f"읽기용 변환: {number}문장", flush=True)
        actual_hash = checksum.hexdigest()
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(f"원본 해시가 manifest와 다릅니다: {source}")
        current = source.stat()
        if (initial.st_size, initial.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
            raise ValueError("변환 중 원본이 변경되었습니다. 원본 생성 완료 후 다시 실행하세요")
    report = {"source": str(source), "source_sha256": actual_hash,
              "include_punctuation": include_punctuation, "counts": dict(totals),
              "statuses": dict(statuses), "files": {p.name: file_hash(p) for p in paths[:3]}}
    write_json(paths[3], report, overwrite=True)
    print(f"읽기용 저장: {destination}; {totals['sentences']}문장 / {totals['visible_pairs']}대응쌍", flush=True)
    return report


def export_methods(root, methods=None, include_punctuation=False, locked=False):
    root = Path(root).resolve()
    if not locked:
        with result_lock(root):
            return export_methods(root, methods, include_punctuation, locked=True)
    manifest = load_manifest(root)
    for name in select_stages(manifest, methods):
        entry = manifest["stages"][name]
        source = (root / entry["path"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("manifest의 원본 경로가 결과 폴더 밖을 가리킵니다")
        export_file(source, source.parent, include_punctuation, expected_hash=entry["sha256"])


def export_cli(args):
    if args.input:
        target = args.output_dir or Path(__file__).resolve().parents[1] / "result" / "readable"
        target.mkdir(parents=True, exist_ok=True)
        with result_lock(target):
            export_file(args.input, target, args.include_punctuation)
    else:
        export_methods(args.result_dir, args.methods, args.include_punctuation)
    return 0
