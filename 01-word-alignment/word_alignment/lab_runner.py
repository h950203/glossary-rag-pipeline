"""Independent BGE experiments with separately saved tokenizer profiles."""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from itertools import zip_longest
import csv
import json
from pathlib import Path
import sqlite3
import time
import uuid
import zlib

from . import core
from .lab_bge import BGEEncoder, TooLong, align_row
from .lab_common import atomic_text, digest, file_hash, identify, read_rows, write_json, write_line
from .lab_compare import COMPARABLE, aligned_rows
from .tokenization import TokenizerSuite, load_methods, method_signature
from .tokenization.common import validate_tokens
from .term_units import term_settings

SCHEMA = "word-alignment-v1.2"


def now():
    return datetime.now(timezone.utc).isoformat()


def packages():
    values = {}
    for name in ("kiwipiepy", "spacy", "nltk", "torch", "numpy", "transformers", "huggingface-hub", "tokenizers"):
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            pass
    return values


def verify_artifact(item, root):
    path = root / item["path"]
    if not path.is_file() or file_hash(path) != item["sha256"]:
        raise ValueError(f"완료된 파일이 변경되었거나 없습니다: {path}")
    return path


def artifact(path, root, **extra):
    return {"status": "complete", "path": str(path.relative_to(root)), "sha256": file_hash(path), **extra}


def save_manifest(root, manifest):
    manifest["updated_at"] = now()
    write_json(root / "manifest.json", manifest, overwrite=True)


def source_plan(args):
    inputs = {lang: str(getattr(args, lang).expanduser().resolve()) for lang in ("en", "ko")}
    hashes = {lang: file_hash(path) for lang, path in inputs.items()}
    total = selected = empty = 0
    for total, en, ko in core.parallel_lines(Path(inputs["en"]), Path(inputs["ko"])):
        if not args.max_sentences or total <= args.max_sentences:
            selected += 1
            empty += not bool(en.strip() and ko.strip())
    return {"input": inputs, "input_hashes": hashes, "corpus_id": digest(hashes),
            "source_signature": digest([hashes, args.max_sentences]),
            "input_counts": {"total_lines": total, "selected_lines": selected, "empty_pairs": empty}}


def model_identity(args):
    result = {"model": args.bge_model, "revision": args.bge_revision}
    local = Path(args.bge_model).expanduser()
    if local.is_dir():
        result["local_files"] = {str(p.relative_to(local)): file_hash(p) for p in sorted(local.rglob("*"))
                                 if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".pt", ".model", ".txt"}}
    return result


def encoder_settings(args):
    keys = ("device", "fp16", "max_model_tokens", "bge_span_tokens", "max_span_candidates", "min_similarity", "margin")
    return {**{key: getattr(args, key) for key in keys}, **model_identity(args),
            "engine_code": file_hash(Path(__file__).with_name("lab_bge.py")),
            "term_policy": term_settings(args),
            "term_alignment_code": file_hash(Path(__file__).with_name("term_alignment.py")),
            "candidate_review_code": file_hash(Path(__file__).with_name("candidate_review.py"))}


def counts(path):
    values = Counter()
    for row in read_rows(path):
        values["sentences"] += 1
        values["status_" + row["status"]] += 1
        for lang in ("en", "ko"):
            values[lang + "_tokens"] += len(row["tokens"][lang])
        pairs = row.get("pairs", [])
        values["pairs"] += len(pairs)
        values["word_pairs"] += sum(p["kind"] == "word" for p in pairs)
        values["phrase_pairs"] += sum(p["kind"] == "phrase" for p in pairs)
    return dict(values)


def verify_orphan(path, manifest, method, token_signature, alignment_signature=None):
    number = 0
    for number, row in enumerate(read_rows(path), 1):
        if (row.get("experiment_id") != manifest["run_id"] or row.get("method") != method
                or row.get("line_id") != number or row.get("tokenizer_signature") != token_signature):
            raise ValueError(f"재사용할 수 없는 미등록 결과: {path}")
        if alignment_signature and row.get("alignment_signature") != alignment_signature:
            raise ValueError(f"정렬 설정이 다른 미등록 결과: {path}")
        expected = identify({k: row[k] for k in ("line_id", "en", "ko")}, manifest["corpus_id"])
        if row["row_id"] != expected["row_id"]:
            raise ValueError(f"원문 ID가 다른 결과: {path}")
        if row["status"] != "empty_text":
            for lang in ("en", "ko"):
                validate_tokens(row[lang], row["tokens"][lang])
    if number != manifest["input_counts"]["selected_lines"]:
        raise ValueError(f"결과 문장 수가 다릅니다: {path}")


def tokenization_metadata(root, method, profile, signature):
    write_json(root / "methods" / method / "tokenization.json",
               {"method": method, "profile": profile, "tokenizer_signature": signature,
                "packages": packages()}, overwrite=True)


def ensure_tokenizations(root, manifest, args, specs, suite):
    from .term_units import TermAnalyzer, term_settings
    pending = {}
    for method in args.methods:
        profile = specs[method]
        signature = digest([method_signature(profile, suite.lexicon), term_settings(args)])
        if "bge_subword" in (profile["en"], profile["ko"]):
            signature = digest([signature, model_identity(args)])
        old = manifest["tokenizations"].get(method)
        path = root / "tokenizations" / (method + ".jsonl")
        if old:
            if old["tokenizer_signature"] != signature:
                raise ValueError(f"토큰화 설정/라이브러리가 변경됐습니다: {method}. 새 출력 폴더를 사용하세요.")
            verify_artifact(old, root)
            if not (root / "methods" / method / "tokenization.json").exists():
                tokenization_metadata(root, method, profile, signature)
            continue
        if path.exists():
            verify_orphan(path, manifest, method, signature)
            manifest["tokenizations"][method] = artifact(path, root, profile=profile,
                tokenizer_signature=signature, counts=counts(path))
            tokenization_metadata(root, method, profile, signature)
            save_manifest(root, manifest)
        else:
            pending[method] = (profile, signature, path)
    if not pending:
        return
    started = time.monotonic()
    analyzer = TermAnalyzer(args)
    with ExitStack() as stack:
        writers = {method: stack.enter_context(atomic_text(item[2])) for method, item in pending.items()}
        for line_id, en, ko in core.parallel_lines(Path(manifest["input"]["en"]), Path(manifest["input"]["ko"])):
            if args.max_sentences and line_id > args.max_sentences:
                break
            suite.begin_row()
            for method, (profile, signature, _) in pending.items():
                row = identify({"line_id": line_id, "en": en, "ko": ko}, manifest["corpus_id"])
                row.update(experiment_id=manifest["run_id"], method=method, tokenizer_profile=profile,
                           tokenizer_signature=signature)
                if en.strip() and ko.strip():
                    row["tokens"] = {lang: suite.tokens(row[lang], lang, profile[lang]) for lang in ("en", "ko")}
                    row["status"] = "ready" if all(row["tokens"].values()) else "empty_tokens"
                    if row["status"] == "ready":
                        analyzer.analyze(row, suite)
                else:
                    row.update(tokens={"en": [], "ko": []}, status="empty_text")
                write_line(writers[method], row)
            if line_id % args.progress_every == 0:
                print(f"tokenize: {line_id} sentences / {len(pending)} methods", flush=True)
    if suite.resolved_revision:
        manifest["tokenizer_model_revision"] = suite.resolved_revision
    for method, (profile, signature, path) in pending.items():
        manifest["tokenizations"][method] = artifact(path, root, profile=profile,
            tokenizer_signature=signature, counts=counts(path),
            elapsed_seconds_shared=round(time.monotonic() - started, 3))
        tokenization_metadata(root, method, profile, signature)
    save_manifest(root, manifest)


def tokenized_rows(root, manifest, methods):
    streams = [read_rows(root / manifest["tokenizations"][method]["path"]) for method in methods]
    for group in zip_longest(*streams):
        if any(row is None for row in group) or len({(r["row_id"], r["line_id"], r["en"], r["ko"]) for r in group}) != 1:
            raise ValueError("방식별 토큰화 문장 대응이 다릅니다")
        yield dict(zip(methods, group))


def ensure_alignments(root, manifest, args):
    settings = encoder_settings(args)
    if manifest.get("encoder_settings") and manifest["encoder_settings"] != settings:
        raise ValueError("기존 BGE 모델/정렬 설정이 다릅니다. 새 출력 폴더를 사용하세요.")
    manifest["encoder_settings"] = settings
    pending = []
    signatures = {}
    for method in args.methods:
        token = manifest["tokenizations"][method]
        signature = digest([settings, token["tokenizer_signature"]])
        signatures[method] = signature
        path = root / "methods" / method / "alignments.jsonl"
        if method in manifest["stages"]:
            if manifest["stages"][method]["alignment_signature"] != signature:
                raise ValueError(f"정렬 설정이 다릅니다: {method}")
            verify_artifact(manifest["stages"][method], root)
        elif path.exists():
            verify_orphan(path, manifest, method, token["tokenizer_signature"], signature)
            manifest["stages"][method] = artifact(path, root, alignment_signature=signature,
                                                  profile=token["profile"], counts=counts(path))
        else:
            pending.append(method)
    save_manifest(root, manifest)
    if not pending:
        return
    cache = root / "cache" / "alignment.sqlite3"
    cache.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache)
    encoder = None
    started = time.monotonic()
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS results (method TEXT, line_id INTEGER, row_id TEXT, signature TEXT, payload BLOB, PRIMARY KEY(method,line_id))")
        for position, rows in enumerate(tokenized_rows(root, manifest, pending), 1):
            missing = []
            for method, row in rows.items():
                item = connection.execute("SELECT row_id,signature FROM results WHERE method=? AND line_id=?",
                                          (method, row["line_id"])).fetchone()
                if item:
                    if item != (row["row_id"], signatures[method]):
                        raise ValueError("BGE 체크포인트와 원문/설정이 다릅니다")
                else:
                    missing.append(method)
            encoded = None
            if any(rows[method]["status"] == "ready" for method in missing):
                if encoder is None:
                    resolved = (manifest.get("bge_model", {}).get("resolved_revision")
                                or manifest.get("tokenizer_model_revision"))
                    print("BGE encoder 로딩; 선택한 방식 간 문장 인코딩 공유", flush=True)
                    encoder = BGEEncoder(args, resolved)
                    actual = encoder.metadata.get("resolved_revision")
                    if resolved and actual != resolved:
                        raise RuntimeError("체크포인트 모델 revision과 다릅니다")
                    manifest["bge_model"] = encoder.metadata
                    save_manifest(root, manifest)
                first = next(rows[method] for method in missing if rows[method]["status"] == "ready")
                try:
                    encoded = encoder.encode_pair(first["en"], first["ko"])
                except TooLong as exc:
                    encoded = exc
            for method in missing:
                row = rows[method]
                row["alignment_signature"] = signatures[method]
                result = align_row(row, encoder, args, encoded)
                payload = zlib.compress(json.dumps(result, ensure_ascii=False, allow_nan=False).encode())
                connection.execute("INSERT INTO results VALUES (?,?,?,?,?)",
                    (method, row["line_id"], row["row_id"], signatures[method], payload))
            if position % args.checkpoint_every == 0:
                connection.commit()
            if position % args.progress_every == 0:
                print(f"BGE: {position} sentences / {len(pending)} methods", flush=True)
        connection.commit()
        for method in pending:
            path = root / "methods" / method / "alignments.jsonl"
            count = 0
            with atomic_text(path) as writer:
                for line_id, row_id, signature, payload in connection.execute(
                        "SELECT line_id,row_id,signature,payload FROM results WHERE method=? ORDER BY line_id", (method,)):
                    row = json.loads(zlib.decompress(payload))
                    count += 1
                    if line_id != count or row["row_id"] != row_id or signature != signatures[method]:
                        raise ValueError("저장된 BGE 결과가 손상됐습니다")
                    write_line(writer, row)
                if count != manifest["input_counts"]["selected_lines"]:
                    raise ValueError("BGE 결과 문장 수가 다릅니다")
            manifest["stages"][method] = artifact(path, root, alignment_signature=signatures[method],
                profile=manifest["tokenizations"][method]["profile"], counts=counts(path),
                elapsed_seconds_shared=round(time.monotonic() - started, 3))
            save_manifest(root, manifest)
    finally:
        connection.close()


def write_summary(root, manifest):
    token_columns = ["method", "english", "korean", "sentences", "en_tokens", "ko_tokens",
                     "mean_en_tokens", "mean_ko_tokens"]
    with atomic_text(root / "tokenization-summary.csv", overwrite=True) as writer:
        out = csv.DictWriter(writer, fieldnames=token_columns)
        out.writeheader()
        for method, item in manifest["tokenizations"].items():
            values, spec = item["counts"], item["profile"]
            n = values.get("sentences", 0)
            out.writerow({"method": method, "english": spec["en"], "korean": spec["ko"],
                          **{k: values.get(k, 0) for k in ("sentences", "en_tokens", "ko_tokens")},
                          "mean_en_tokens": values.get("en_tokens", 0) / n if n else 0,
                          "mean_ko_tokens": values.get("ko_tokens", 0) / n if n else 0})
    with atomic_text(root / "tokenization-preview.jsonl", overwrite=True) as writer:
        for position, rows in enumerate(tokenized_rows(root, manifest, list(manifest["tokenizations"])), 1):
            if position > 20:
                break
            first = next(iter(rows.values()))
            write_line(writer, {"line_id": first["line_id"], "en": first["en"], "ko": first["ko"],
                "methods": {method: {"profile": row["tokenizer_profile"],
                    "en": [t["text"] for t in row["tokens"]["en"]],
                    "ko": [t["text"] for t in row["tokens"]["ko"]]} for method, row in rows.items()}})
    stages = [name for name, item in manifest["stages"].items() if item["status"] == "complete"]
    write_json(root / "summary.json", {"run_id": manifest["run_id"], "engine": "C_bge",
               "counts_are_accuracy_metrics": False,
               "methods": {name: item for name, item in manifest["stages"].items()}}, overwrite=True)
    if not stages:
        return
    reference = "C_bge" if "C_bge" in stages else stages[0]
    columns = ["line_id", "row_id", "comparable", "common_pairs"]
    for name in stages:
        columns += [name + "_pairs", name + "_added", name + "_removed"]
    with atomic_text(root / "comparison.csv", overwrite=True) as writer:
        output = csv.DictWriter(writer, fieldnames=columns)
        output.writeheader()
        for rows in aligned_rows(root, manifest, stages):
            keys = {name: {p["pair_id"] for p in row["pairs"]} for name, row in rows.items()}
            comparable = all(row["status"] in COMPARABLE for row in rows.values())
            first = next(iter(rows.values()))
            record = {"line_id": first["line_id"], "row_id": first["row_id"], "comparable": comparable,
                      "common_pairs": len(set.intersection(*keys.values())) if comparable else ""}
            for name in stages:
                record[name + "_pairs"] = len(keys[name])
                record[name + "_added"] = len(keys[name] - keys[reference]) if comparable else ""
                record[name + "_removed"] = len(keys[reference] - keys[name]) if comparable else ""
            output.writerow(record)


def run(args):
    specs = load_methods(args.methods_file)
    plan = source_plan(args)
    if args.dry_run:
        print(json.dumps({"engine": "C_bge", "command": args.command, "methods": {m: specs[m] for m in args.methods},
                          "output_dir": str(args.output_dir), "input_counts": plan["input_counts"],
                          "loads_encoder": False, "fast_align_required": False}, ensure_ascii=False, indent=2))
        return 0
    root = args.output_dir.expanduser().resolve()
    reserved = {".run.lock"}
    if getattr(args, "managed_layout", False):
        reserved.update({".pipeline.lock", "run.json", "run.log"})
    if root.exists() and any(p.name not in reserved for p in root.iterdir()) and not args.resume:
        raise ValueError(f"출력이 있습니다: {root}. --resume 또는 새 --output-dir를 사용하세요.")
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("같은 출력 폴더에서 정렬/토큰화가 실행 중입니다") from exc
        if args.resume:
            manifest = json.loads((root / "manifest.json").read_text())
            if manifest.get("schema") != SCHEMA or manifest["source_signature"] != plan["source_signature"]:
                raise ValueError("입력/선택 문장 수 또는 실험 구조가 다릅니다. 새 출력 폴더를 사용하세요.")
            manifest["input"] = plan["input"]
        else:
            manifest = {"schema": SCHEMA, "engine": "C_bge", "run_id": uuid.uuid4().hex,
                        "created_at": now(), "tokenizations": {}, "stages": {}, **plan}
        manifest.update(status="tokenizing", packages=packages())
        save_manifest(root, manifest)
        try:
            suite = TokenizerSuite(args, manifest.get("bge_model", {}).get("resolved_revision")
                                   or manifest.get("tokenizer_model_revision"))
            ensure_tokenizations(root, manifest, args, specs, suite)
            if args.command == "run":
                manifest["status"] = "aligning"
                save_manifest(root, manifest)
                ensure_alignments(root, manifest, args)
            write_summary(root, manifest)
            if args.command == "run":
                from .readable import export_methods
                export_methods(root, args.methods, args.include_punctuation, locked=True)
                from .term_export import export_terms
                export_terms(root, manifest, args.methods)
            manifest["status"] = "complete" if args.command == "run" else "tokenized"
            manifest.pop("last_error", None)
            save_manifest(root, manifest)
            print(f"저장: {root}; methods={', '.join(args.methods)}", flush=True)
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            last_error=f"{type(exc).__name__}: {exc}")
            save_manifest(root, manifest)
            raise
